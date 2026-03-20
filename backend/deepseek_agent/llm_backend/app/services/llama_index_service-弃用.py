import os
import re
import json
import asyncio
import json_repair
from typing import Any, List, Callable, Optional, Union, Dict

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.readers.json import JSONReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.prompts import PromptTemplate
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from app.core.logger import get_logger

logger = get_logger(service="LlamaIndexService")

# ==========================================
# 1. 提示词与解析器 (Prompt & Parser)
# ==========================================
TACTICAL_EXTRACT_PROMPT = PromptTemplate("""
- 目标 -
你是一名专业的军事以此情报分析师。给定一份战术简报（文本），请从中识别军事实体及其相互关系。
你需要严格基于文本内容，提取实体的详细属性以及它们之间包含“四维信息”的战术关系。

- 步骤 -
1. 识别实体 (Entities)：
   对于每一个识别出的实体，提取以下信息：
   - name: 实体名称 (如 "B-1B", "永暑礁")。
   - label: 实体类型 (如 "Platform", "Location", "Unit", "Signal")。
   - description: 实体的详细画像。请根据实体类型，包含以下关键特征：
	 * 如果是 [Platform] (平台): 包含国别、型号、主要功能、携带武器或传感器类型 (例如: "美国空军超音速重型轰炸机，配备L波段雷达，具有低空突防能力")。
	 * 如果是 [Location] (地点): 包含地理属性、主权归属、战略地位或设施情况 (例如: "南海重要岛礁，建有3000米跑道及雷达站")。
	 * 如果是 [Unit] (部队): 包含隶属关系、级别、作战任务 (例如: "美国海军第七舰队下属航母打击群，负责西太平洋防务")。
	 * 如果是 [Signal] (信号): 包含频段、用途、技术体制 (例如: "L波段长程对空警戒雷达信号")。
2. 识别关系 (Relationships/Actions)：
   识别实体之间的交互或事件。对于每一个关系，必须提取以下战术参数。如果文中未明确提及，请使用 null。
   
   - relation: 动作类型 (使用大写英文，如 PATROLLING, EMITTING, DETECTED_AT, EXERCISING, APPROACHING)。
   - description: 事件的简短中文总结 (例如: "B-1B 在低空进行威慑巡航")。
   - timestamp: 具体时间 (格式: YYYY-MM-DD HH:MM)。非常重要！如果只说了"4月25日"，请补全年份。
   - coordinates: GIS 坐标 (经纬度) 或具体的地点名称。
   - signal_info: 任何雷达/通信信号细节 (例如: "L波段", "12.5GHz", "AIS信号")。
   - tactical_intent: 推断的战术意图 (例如: "威慑", "侦察", "航行自由", "演习")。

- 输出格式 -
返回一个合法的 JSON 对象，包含两个键："entities" 和 "relationships"。
- "entities": 实体对象列表。
- "relationships": 关系对象列表。在每个关系中，包含一个嵌套对象 "tactical_data" 来存储 timestamp, coordinates, signal_info, tactical_intent。
""")

def parse_fn(response_str: str) -> Any:
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities, relationships = [], []
    if not match: return entities, relationships
    try:
        data = json_repair.loads(match.group(0))
        if isinstance(data, dict):
            raw_entities = data.get("entities", [])
            if isinstance(raw_entities, list):
                for entity in raw_entities:
                    if not entity or not isinstance(entity, dict): continue
                    name = entity.get("name")
                    label = entity.get("label")
                    if not name: continue
                    entities.append((str(name).strip(), str(label).strip() if label else "Unknown", str(entity.get("description") or "无描述").strip()))

            raw_relations = data.get("relationships", [])
            if isinstance(raw_relations, list):
                for rel in raw_relations:
                    if not rel or not isinstance(rel, dict): continue
                    source, target = rel.get("source"), rel.get("target")
                    if not source or not target: continue
                    rel_type = str(rel.get("relation") or "RELATED_TO")
                    tac = rel.get("tactical_data", {})
                    if not isinstance(tac, dict): tac = {}
                    relationships.append((
                        str(source), str(target), rel_type, rel.get("description"),
                        {"timestamp": tac.get("timestamp"), "geo": tac.get("coordinates"), "signal": tac.get("signal_info"), "intent": tac.get("tactical_intent")}
                    ))
        return entities, relationships
    except Exception as e:
        logger.error(f"JSON 解析错误: {e}")
        return entities, relationships

# ==========================================
# 2. 自定义核心组件 (Extractor & Store)
# ==========================================
from llama_index.core.llms import LLM
class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(self, llm=None, extract_prompt=None, parse_fn=default_parse_triplets_fn, max_paths_per_chunk=2, num_workers=4):
        if isinstance(extract_prompt, str): extract_prompt = PromptTemplate(extract_prompt)
        super().__init__(
            llm=llm or Settings.llm, 
            extract_prompt=extract_prompt or TACTICAL_EXTRACT_PROMPT, 
            parse_fn=parse_fn, num_workers=num_workers, max_paths_per_chunk=max_paths_per_chunk
        )

    async def acall(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(jobs, workers=self.num_workers, show_progress=show_progress, desc="Extracting paths from text")

    async def _aextract(self, node: BaseNode) -> BaseNode:
        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(self.extract_prompt, text=text, max_knowledge_triplets=self.max_paths_per_chunk)
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities, entities_relationship = [], []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            existing_nodes.append(EntityNode(name=entity, label=entity_type, properties=entity_metadata.copy()))

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description, tactical_data = triple
            relation_metadata["relationship_description"] = description
            if tactical_data:
                clean_tactical = {k: v for k, v in tactical_data.items() if v}
                relation_metadata.update(clean_tactical)
            existing_relations.append(Relation(label=rel, source_id=subj, target_id=obj, properties=relation_metadata.copy()))

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

class TacticalGraphStore(Neo4jPropertyGraphStore):
    def execute_cypher(self, cypher_sql):
        try:
            records = self.structured_query(cypher_sql)
            return records, None
        except Exception as e:
            return [], str(e)
            
    def close(self):
        self.driver.close()

# ==========================================
# 3. 封装为 FastAPI 服务类 (Service)
# ==========================================
class LlamaIndexRAGService:
    _instance = None
    
    # 保证整个应用生命周期中，模型和数据库连接只初始化一次
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaIndexRAGService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("正在初始化 LlamaIndex RAG 服务...")
        
        # 1. 设置全局大模型与嵌入模型
        self.llm = Ollama(model="qwen3:8b", request_timeout=360.0)
        self.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # 2. 设置图数据库连接
        self.graph_store = TacticalGraphStore(
            username="neo4j", 
            password="12345678", 
            url="bolt://localhost:7688"
        )
        
        # 3. 检查并构建/加载图谱索引
        records, error = self.graph_store.execute_cypher("MATCH (n) RETURN count(n) AS node_count")
        if not error and records and records[0].get('node_count', 0) > 0:
            logger.info(f"✅ 检测到 Neo4j 中已存在 {records[0]['node_count']} 个节点，直接加载图谱！")
            self.index = PropertyGraphIndex.from_existing(property_graph_store=self.graph_store)
        else:
            logger.info("⚠️ Neo4j 数据库为空，开始读取本地文件构建图谱...")
            self.index = self._build_index_from_documents()

        # 4. 初始化查询引擎 (注意：增加 streaming=True)
        local_rerank_path = "/data/hitszdzh/models/rerank_models/BAAI/bge-reranker-v2-m3"
        self.reranker = SentenceTransformerRerank(model=local_rerank_path, top_n=3)
        
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            node_postprocessors=[self.reranker],
            include_text=True,
            verbose=True,
            streaming=False  # 流式输出，当作为中间节点的时候不需要流式输出
        )

    def _build_index_from_documents(self):
        """从本地目录加载文档并进行知识抽取"""
        from llama_index.core.node_parser import SentenceSplitter
        
        briefing_loader = SimpleDirectoryReader(
            input_dir="/data/hitszdzh/RAG/LlamaIndex/data_military", 
            recursive=True,
            required_exts=[".pdf", ".docx", ".doc", ".txt", ".md", ".csv"],
        )
        briefing_docs = briefing_loader.load_data()
      
        json_docs = []
        json_dir_path = "/data/hitszdzh//RAG/LlamaIndex/data_military"
        if os.path.exists(json_dir_path):
            json_reader = JSONReader(levels_back=0, clean_json=True)
            for root, dirs, files in os.walk(json_dir_path):
                for file in files:
                    if file.endswith(".json"):
                        loaded_docs = json_reader.load_data(input_file=os.path.join(root, file))
                        json_docs.extend(loaded_docs)
                        
        documents = briefing_docs + json_docs
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents)
        
        kg_extractor = GraphRAGExtractor(llm=self.llm, extract_prompt=TACTICAL_EXTRACT_PROMPT, max_paths_per_chunk=2, parse_fn=parse_fn)
        
        return PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[kg_extractor],
            property_graph_store=self.graph_store,
            show_progress=True,
        )

    # 中间节点不需要流式输出了，这个生成器可以不用了
    async def generate_stream(self, query: str):
        """提供给 FastAPI 接口调用的流式生成器"""
        try:
            logger.info(f"LlamaIndex 开始查询: {query}")
            # 执行异步查询
            response = await self.query_engine.aquery(query)
            
            # 迭代流式响应，吐出给前端
            async for text in response.async_response_gen():
                # 按照前端需要的 SSE (Server-Sent Events) 格式打包
                yield f"data: {json.dumps(text, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            logger.error(f"LlamaIndex 查询失败: {str(e)}")
            yield f"data: {json.dumps('知识库查询发生异常，请检查后端日志。', ensure_ascii=False)}\n\n"