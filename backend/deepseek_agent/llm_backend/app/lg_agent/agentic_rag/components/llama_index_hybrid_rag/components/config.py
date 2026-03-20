import os

STORAGE_DIR = "storage"  # directory to cache the generated index
DATA_DIR = "data"  # directory containing the documents to index
MODEL_DIR = "localmodels"  # directory containing the model files, use None if use remote model
CONFIG_STORE_FILE = "config_store.json" # local storage for configurations

# The device that used for running the model. 
# Set it to 'auto' will automatically detect (with warnings), or it can be manually set to one of 'cuda', 'mps', 'cpu', or 'xpu'.
LLM_DEVICE = "auto"
EMBEDDING_DEVICE = "auto"

# LLM Settings

HISTORY_LEN = 3

MAX_TOKENS = 2048

TEMPERATURE = 0.1

TOP_K = 5

SYSTEM_PROMPT = "You are an AI assistant that helps users to find accurate information. You can answer questions, provide explanations, and generate text based on the input. Please answer the user's question exactly in the same language as the question or follow user's instructions. For example, if user's question is in Chinese, please generate answer in Chinese as well. If you don't know the answer, please reply the user that you don't know. If you need more information, you can ask the user for clarification. Please be professional to the user."

# 响应合成模式（ResponseMode） 决定了系统在检索到多个相关的文本节点（Nodes）后，如何将这些散落的信息交给大模型（LLM）来组合成最终答案
RESPONSE_MODE = [   # Configure the response mode of the query engine
            "compact",
            "refine",
            "tree_summarize",
            "simple_summarize",
            "accumulate",
            "compact_accumulate",
]
DEFAULT_RESPONSE_MODE = "compact"  # 【chris更改】这里原来是simple_summarize，但是经过之前的测试，采用默认的compact效果好一定

# 【chris】查询方法：
# 直接在服务器的终端里运行以下命令：curl http://localhost:11434
# 如果终端返回了 Ollama is running 这行白底黑字，说明你的 Ollama 正在正常运行，且地址就是 http://localhost:11434。你在 config.py 中直接填入这个地址即可。
OLLAMA_API_URL = "http://localhost:11434"

# Models' API configuration，set the KEY in environment variables
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

LLM_API_LIST = {
    # Ollama API
    "Ollama": {
        "api_base": OLLAMA_API_URL,
        "models": [],
        "provider": "Ollama",
    },
    # OpenAI API
    "OpenAI": {
        "api_key": OPENAI_API_KEY,
        "api_base": "https://api.openai.com/v1/",
        "models": ["gpt-4", "gpt-3.5", "gpt-4o"],
        "provider": "OpenAI",
    },
    # DeepSeek API
    "DeepSeek": {
        "api_key": DEEPSEEK_API_KEY,
        "api_base": "https://api.deepseek.com/v1/",
        "models": ["deepseek-chat","deepseek-reasoner"],
        "provider": "DeepSeek",
    },
    # Moonshot API
    "Moonshot": {
        "api_key": MOONSHOT_API_KEY,
        "api_base": "https://api.moonshot.cn/v1/",
        "models": ["moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k"],
        "provider": "Moonshot",
    },
    # ZhiPu API
    "Zhipu": {
        "api_key": ZHIPU_API_KEY,
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "models": ["glm-4-plus", "glm-4-0520", "glm-4", "glm-4-air", "glm-4-airx", "glm-4-long", "glm-4-flashx", "glm-4-flash", "glm-4v-plus", "glm-4v"],
        "provider": "Zhipu",
    },
}

# Text splitter configuration

DEFAULT_CHUNK_SIZE = 2048
DEFAULT_CHUNK_OVERLAP = 512
ZH_TITLE_ENHANCE = True # Chinese title enhance  【chris更改】这里原来是 False

# Storage configuration

MONGO_URI = "mongodb://localhost:27017"
REDIS_URI = "redis://localhost:6379"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
ES_URI = "http://localhost:9200"

# Default vector database type, including "es" and "chroma"
DEFAULT_VS_TYPE = "es"

# Chat store type，including "simple" and "redis"
DEFAULT_CHAT_STORE = "redis"
CHAT_STORE_FILE_NAME = "chat_store.json"
CHAT_STORE_KEY = "user1"

# Use HuggingFace model，Configure domestic mirror
HF_ENDPOINT = "https://hf-mirror.com" # Default to be "https://huggingface.co"

# Configure Embedding model
DEFAULT_EMBEDDING_MODEL = "bge-small-zh-v1.5"
# EMBEDDING_MODEL_PATH = {
#     "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
#     "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
# }
# 【chris更改】
EMBEDDING_MODEL_PATH = {"bge-small-zh-v1.5": "/home/data/hitszdzh/RAG/LlamaIndex/ThinkRAG-main/localmodels/BAAI/bge-small-zh-v1.5"}


# Configure Reranker model
DEFAULT_RERANKER_MODEL = "bge-reranker-v2-m3"
RERANKER_MODEL_PATH = {
    # "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    # 【chris更改】
    "bge-reranker-v2-m3": "/home/data/hitszdzh/models/rerank_models/BAAI/bge-reranker-v2-m3" ,
    "bge-reranker-base": "/home/data/hitszdzh/RAG/LlamaIndex/ThinkRAG-main/localmodels/BAAI/bge-reranker-base"
}



# Use reranker model or not
USE_RERANKER = True
RERANKER_MODEL_TOP_N = 3
RERANKER_MAX_LENGTH = 1024

# Evironment variable, default to be "development", set to "production" for production environment
THINKRAG_ENV = os.getenv("THINKRAG_ENV", "development")
DEV_MODE = THINKRAG_ENV == "development"

# For creating IndexManager
DEFAULT_INDEX_NAME = "knowledge_base"