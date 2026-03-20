from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

from app.core.config import settings
from ..components.state import InputState, OutputState, OverallState
# 导入你的组件包
from ..components.state import MainGraphState
from ..components.planner.node import create_planner_node
from ..components.intent_router.node import create_intent_recognition_node, create_intent_dispatcher_node
from ..components.llama_index_kg_rag.node import create_kg_rag_node
from ..components.llama_index_hybrid_rag.node import create_hybrid_rag_node
from ..components.aggregate.node import create_aggregate_node
from ..components.final_answer.node import create_final_answer_node
from .edges import map_tasks_to_intent, route_based_on_intent

from app.core.logger import get_logger

logger = get_logger(service="agentic_rag_workflow")

# TODO 全局记忆体（在实际生产中，如果你多进程部署，这里需要换成 RedisSaver 或 PostgresSaver）
# 目前单例或单进程调试用 MemorySaver 足够
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

def create_agentic_rag_workflow():
    """工厂函数：构建并编译 RAG 工作流"""
    
    # 1. 资源初始化
    if settings.AGENT_SERVICE == "deepseek":
        llm = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=0.1)
    else:
        llm = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.1)


    # 2. 节点工厂实例化
    planner_node = create_planner_node(llm=llm)
    intent_recognition_node = create_intent_recognition_node(llm=llm)
    intent_dispatcher_node = create_intent_dispatcher_node()
    kg_rag_node = create_kg_rag_node()
    hybrid_rag_node = create_hybrid_rag_node()
    aggregate_node = create_aggregate_node()
    final_answer_node = create_final_answer_node(llm=llm)
    # 3. 开始画图
    builder = StateGraph(
        OverallState,        # 图内部流转使用的总黑板
        input=InputState,    # 外部传入数据的检查口
        output=OutputState   # 最终输出数据的过滤器
        )

    builder.add_node("planner", planner_node)
    builder.add_node("intent_recognition_node", intent_recognition_node)
    builder.add_node("intent_dispatcher_node", intent_dispatcher_node)
    builder.add_node("kg_rag_node", kg_rag_node)
    builder.add_node("hybrid_rag_node", hybrid_rag_node)
    builder.add_node("aggregate_node", aggregate_node)
    builder.add_node("final_answer_node", final_answer_node)
    # 添加连线
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", map_tasks_to_intent, ["intent_recognition_node"])
    builder.add_edge("intent_recognition_node", "intent_dispatcher_node") # 并行的意图识别结果汇聚到intent_dispatcher_node中
    builder.add_conditional_edges("intent_dispatcher_node", route_based_on_intent, ["kg_rag_node", "hybrid_rag_node"])



    builder.add_edge("kg_rag_node", "aggregate_node")
    builder.add_edge("hybrid_rag_node", "aggregate_node")
    builder.add_edge("aggregate_node", "final_answer_node")
    builder.add_edge("final_answer_node", END)

    return builder.compile(checkpointer=checkpointer)

# 暴露给外部调用的实例
agentic_rag_workflow = create_agentic_rag_workflow()

# 暴露 llm 供接口做最终的打字机输出使用
# ⚠️ 这里只是简单的导出一个实例，如果在正式大型项目中，你可以在外层 Service 里再次定义大模型。
if settings.AGENT_SERVICE == "deepseek":
    llm = ChatDeepSeek(api_key=settings.DEEPSEEK_API_KEY, model_name=settings.DEEPSEEK_MODEL, temperature=0.1)
else:
    llm = ChatOllama(model=settings.OLLAMA_AGENT_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.1)
