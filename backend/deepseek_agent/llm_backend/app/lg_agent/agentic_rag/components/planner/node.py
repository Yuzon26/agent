from typing import Any, Callable, Coroutine, Dict, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from app.core.logger import get_logger

from ..state import OverallState, PlannerOutput, HistoryRecord
from .prompts import PLANNER_SYSTEM_PROMPT,HUMAN_MESSAGE_TEMPLATE
logger = get_logger(service="planner_node")


# 🌟 格式化历史记录为字符串
def format_history(history: List[HistoryRecord]) -> str:
    if not history:
        return "无历史对话"
    return "\n".join([f"用户: {h['question']}\nAI: {h['answer']}" for h in history])

def create_planner_node(llm: BaseChatModel) -> Callable[[OverallState], Coroutine[Any, Any, Dict[str, Any]]]:
    """创建任务拆解节点"""
    

    prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", HUMAN_MESSAGE_TEMPLATE)
        ])
    
    chain = prompt | llm.with_structured_output(PlannerOutput)

    async def planner_node(state: OverallState) -> Dict[str, Any]:
        logger.info(f"🧠 [Planner] 正在拆解问题: {state['original_question']}")
        # History 是怎么从 service中的 MemorySaver “瞬移”到 Planner 节点里的？
        # 你看到的 state.get('history', []) 确实只有两行代码，仿佛历史数据凭空出现了。这背后其实是 LangGraph 框架帮你做了**“暗箱操作”**。
        # 它的底层作用机制是这样的一个生命周期：
        # 图启动前（查档）：
            # 当你触发 astream 时，LangGraph 会拿着你传进去的 thread_id，去 MemorySaver（内存里的一个大字典）中查找。
        # 状态合并（读档）：
            # 假设它找到了这个 thread_id 对应的旧存档（也就是上一轮跑完后的 OverallState ，里面包含了满满的 history）。
            # LangGraph 会在后台偷偷地把旧存档拿出来，作为当前图的初始状态。
        # 节点执行（使用）：
            # 图开始顺着边（Edge）往下走，第一个来到的就是 Planner 节点。
            # 此时传给 planner_node(state: OverallState) 的这个 state，已经是被 LangGraph 自动填满了旧存档数据的 state 了。
            # 所以，当你在 Planner 里写下 state.get('history') 时，你其实是在读取 LangGraph 刚刚从 MemorySaver 里帮你拿出来并塞进 state 里的旧数据。
            # 简而言之：MemorySaver 就像是游戏的“自动存档/读档机制”。每次进入第一关（Planner）时，你的背包（State）里就已经自动装好了上次玩剩下的道具（History）。
        history_records = state.get('history', [])
        logger.info(f"History messages (Raw): {history_records}")
        history_str = format_history(history_records)


        response = await chain.ainvoke({
            "history_str": history_str, 
            "question": state['original_question']
        })
        
        # 打印拆解出的任务数量和内容，方便调试
        logger.info(f"Total Sub Task: {len(response.tasks)}")
        for i, task in enumerate(response.tasks):
            logger.info(f"Sub Task[{i+1}]: {task.question}")
            
        return {"tasks": response.tasks}

    return planner_node