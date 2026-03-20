import os
import re
import json
import json_repair
from typing import Any, List, Callable, Dict, Coroutine

# LlamaIndex 相关导入
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
from llama_index.core.llms import LLM

# 项目及状态导入
from app.core.logger import get_logger
from ..state import TaskState
from .prompts import TACTICAL_EXTRACT_PROMPT

logger = get_logger(service="llama_index_kg_rag_node")


# 解析器
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
# 3. LlamaIndex RAG 类 (Singleton Service)
# ==========================================
class LlamaIndexKnowledgeGraphRAG:
    _instance = None
    
    # 保证整个应用生命周期中，模型和数据库连接只初始化一次
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaIndexKnowledgeGraphRAG, cls).__new__(cls)
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

        # 4. 初始化查询引擎 (作为中间节点，此处不需要流式输出)
        local_rerank_path = "/data/hitszdzh/models/rerank_models/BAAI/bge-reranker-v2-m3"
        self.reranker = SentenceTransformerRerank(model=local_rerank_path, top_n=3)
        
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            node_postprocessors=[self.reranker],
            include_text=True,
            verbose=True,
            streaming=False
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

def create_kg_rag_node() -> Callable[[TaskState], Coroutine[Any, Any, Dict[str, Any]]]:
    """创建知识图谱推理节点"""
    async def kg_rag_node(state: TaskState) -> Dict[str, Any]:
        logger.info(f"🕸️ [KG RAG] 正在查知识图谱: {state['question']}")
        llamaindex_rag = LlamaIndexKnowledgeGraphRAG()
        response = await llamaindex_rag.query_engine.aquery(state["question"])
        
        # 兼容流式引擎和非流式引擎的数据提取
        # 通过这种兼容，即便在底层 LlamaIndexKnowledgeGraphRAG 开启了流式输出，依然能拿到非流式输出的结果
        # 拿到的永远都是完整的一大段话（非流式结果），而绝对不会拿到半截数据或者报错
        final_text = response.get_response().response if hasattr(response, "get_response") else str(response)
        return {"task_results": [{"question": state["question"], "result": final_text}]}

    return kg_rag_node

# TODO 后续把这个删掉
# def create_hybrid_rag_node() -> Callable[[TaskState], Coroutine[Any, Any, Dict[str, Any]]]:
#     """创建混合检索事实查询节点"""
#     async def hybrid_rag_node(state: TaskState) -> Dict[str, Any]:
#         logger.info(f"🔍 [Hybrid RAG] 正在查混合向量: {state['question']}")
#         llamaindex_rag = LlamaIndexKnowledgeGraphRAG()
#         response = await llamaindex_rag.query_engine.aquery(state["question"])
        
#         final_text = response.get_response().response if hasattr(response, "get_response") else str(response)
#         return {"task_results": [{"question": state["question"], "result": final_text}]}

#     return hybrid_rag_node