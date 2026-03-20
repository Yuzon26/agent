import os
import jieba
from typing import Any, Callable, Coroutine, Dict, List

# LlamaIndex 核心导入
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# LlamaIndex 检索与融合导入
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 项目状态导入
from app.core.logger import get_logger
from ..state import TaskState

logger = get_logger(service="llama_index_hybrid_rag_node")

# ==========================================
# 1. BM25 中文分词器 (核心：解决 BM25 不支持中文的问题)
# ==========================================
def chinese_tokenizer(text: str) -> List[str]:
    return list(jieba.cut(text))

# ==========================================
# 2. 混合检索 RAG 类 (Singleton Service)
# ==========================================
class LlamaIndexHybridRAG:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaIndexHybridRAG, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("正在初始化 LlamaIndex Hybrid (混合检索) 服务...")
        
        # 1. 全局模型设置
        self.llm = Ollama(model="qwen3:8b", request_timeout=360.0)
        self.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # 🌟 关键：使用 ThinkRAG 项目基线推荐的军工简报切块黄金参数
        Settings.chunk_size = 512
        Settings.chunk_overlap = 128

        # 持久化存储路径 (避免每次启动都重新切分文档)
        # 获取当前 node.py 文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 在当前目录下创建一个名为 hybrid_storage 的子文件夹
        persist_dir = os.path.join(current_dir, "hybrid_storage")
        
        # 2. 加载或构建纯向量索引
        if os.path.exists(persist_dir):
            logger.info("✅ 检测到本地 Hybrid 向量存储，直接加载...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            logger.info("⚠️ 本地不存在 Hybrid 存储，开始读取本地文件构建纯向量索引...")
            self.index = self._build_index_from_documents(persist_dir)

        # 3. 构建混合检索双路召回与融合组件
        # a. 向量检索器
        vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=5)
        
        # b. BM25 关键词检索器
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore, 
            similarity_top_k=5, 
            tokenizer=chinese_tokenizer
        )
        
        # c. QueryFusion 融合检索器 (根据基线项目，使用 dist_based_score 距离分数融合)
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            retriever_weights=[0.6, 0.4], # 向量占 60%，BM25 占 40%
            similarity_top_k=5,
            num_queries=1, 
            mode="dist_based_score", 
            use_async=True
        )
        
        # 4. 初始化带有重排器的查询引擎 (无需流式)
        local_rerank_path = "/data/hitszdzh/models/rerank_models/BAAI/bge-reranker-v2-m3"
        self.reranker = SentenceTransformerRerank(model=local_rerank_path, top_n=3)
        
        self.hybrid_query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[self.reranker],
            llm=self.llm,
            streaming=False
        )

    def _build_index_from_documents(self, persist_dir: str):
        """从本地加载文档并建立纯向量索引"""
        briefing_loader = SimpleDirectoryReader(
            input_dir="/data/hitszdzh/RAG/LlamaIndex/data_military", 
            recursive=True,
            required_exts=[".pdf", ".docx", ".doc", ".txt", ".md", ".csv"],
        )
        documents = briefing_loader.load_data()
        
        # 切块并转为 Nodes
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
        nodes = splitter.get_nodes_from_documents(documents)
        
        # 构建索引并持久化到本地
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info(f"✅ Hybrid 索引构建完成，并已持久化至 {persist_dir}")
        
        return index

# ==========================================
# 3. LangGraph 节点工厂函数
# ==========================================
def create_hybrid_rag_node() -> Callable[[TaskState], Coroutine[Any, Any, Dict[str, Any]]]:
    """创建混合检索事实查询节点"""
    async def hybrid_rag_node(state: TaskState) -> Dict[str, Any]:
        logger.info(f"🔍 [Hybrid RAG] 正在查混合向量: {state['question']}")
        
        # 实例化单例并调用融合引擎
        hybrid_rag = LlamaIndexHybridRAG()
        response = await hybrid_rag.hybrid_query_engine.aquery(state["question"])
        
        # 兼容流式引擎和非流式引擎的数据提取
        final_text = response.get_response().response if hasattr(response, "get_response") else str(response)
        
        return {"task_results": [{"question": state["question"], "result": final_text}]}

    return hybrid_rag_node