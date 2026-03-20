from typing import Any, Callable, Coroutine, Dict, Awaitable
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command, Send
from app.core.logger import get_logger

from ..state import TaskState, IntentOutput,OverallState
# 确保你的 PROMPT 文件路径正确
from app.lg_agent.agentic_rag.prompts.prompts import INTENT_RECOGNITION_SYSTEM_PROMPT

logger = get_logger(service="intent_recognition_node")

def create_intent_recognition_node(llm: BaseChatModel) -> Callable[[TaskState], Awaitable[Dict[str, Any]]]:
    """创建意图识别与路由分发节点"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_RECOGNITION_SYSTEM_PROMPT),
        ("human", "问题：{question}")
    ])
    
    chain = prompt | llm.with_structured_output(IntentOutput)

    async def intent_recognition_node(state: TaskState) -> Dict[str, Any]:
        logger.info(f"🎯 [Intent] 正在识别意图: {state['question']}")
        response = await chain.ainvoke({"question": state["question"]})
        
        # return {"intent": response.intent, "question": state["question"]}
        # ✅ 正确的返回格式：返回一个列表，让状态机使用 add 操作合并
        return {
            "intent_results": [
                {
                    "question": state["question"], 
                    "intent": response.intent
                }
            ]
        }
        # # 核心：根据分类结果执行 Command 动态跳转
        # # Command 如果没有 update 参数，它传递给下一个节点的状态更新就是空的。于是 hybrid_rag_node 收到了一个空状态 {}，当它试图读取 state['question'] 时，自然就触发了 KeyError

        # # 由于没有状态需要更新，因此可以不附加状态
        # if response.intent in ["causality", "summary"]:
        #     return Command(goto="kg_rag_node", update={"question": state["question"]})

        # else:
        #     return Command(goto="hybrid_rag_node", update={"question": state["question"]})


    return intent_recognition_node


# ==========================================
# 2. 意图收束分发节点 (Reduce/Barrier 同步节点)
# ==========================================
def create_intent_dispatcher_node() -> Callable[[OverallState], Awaitable[Dict[str, Any]]]:
    """创建意图收束节点（充当并发执行的同步屏障）"""
    
    async def intent_dispatcher_node(state: OverallState) -> Dict[str, Any]:
        # 这个节点不需要调用 LLM，它的唯一作用是等待所有 intent_recognition_node 结束
        # 此时控制权交还给主线程，收集箱 (intent_results) 已经装满了所有并发结果
        logger.info(f"🚦 [Dispatcher] 所有意图识别完毕，共收集到 {len(state.get('intent_results', []))} 个意图，准备下发至 RAG 矩阵...")
        
        # 不更新任何状态，直接放行，把控制权交给后方的条件边 (route_based_on_intent)
        return {}

    return intent_dispatcher_node