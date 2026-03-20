import json
from typing import Any, Callable, Coroutine, Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.core.logger import get_logger

from ..state import OverallState

logger = get_logger(service="final_answer_node")

def create_final_answer_node(llm: BaseChatModel) -> Callable[[OverallState], Coroutine[Any, Any, Dict[str, Any]]]:
    
    async def final_answer_node(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
        logger.info("✍️ [Final Answer] 正在生成最终回答...")
        
        aggregated_answers = state.get("aggregated_answers", "")
        original_question = state.get("original_question", "")
        history_records = state.get("history", [])

        # 1. 构建系统提示词      
        system_prompt = f"""你是一个顶级的军事与情报分析专家。
        为了准确回答用户的原始问题，我们已经将其拆分为多个子问题，并派发给底层的专业知识图谱与向量库进行深度检索与推理得到了多个子回答。
        以下是各个底层知识库传回的【详尽子回答】：
        
        ================================
        {aggregated_answers}
        ================================
              
        执行要求：
        1. 必须充分利用上述情报中的事实细节（如具体数值、坐标、型号、时间等），绝对不要遗漏关键数据。
        2. 不要机械地罗列，请将分散的情报融会贯通，形成逻辑严密、条理清晰的专业回复。
        3. 语气要保持专业、客观。如果底层解答中信息有缺失，请如实说明。
        4. 结合历史对话上下文，自然、连贯地回答用户的最新提问。
        """


        messages = [SystemMessage(content=system_prompt)]
        
        # 2. 注入滑动窗口保留的历史记录
        for h in history_records:
            messages.append(HumanMessage(content=h["question"]))
            messages.append(AIMessage(content=h["answer"]))
            
        # 3. 注入当前最新问题
        messages.append(HumanMessage(content=original_question))
        
        # 4. 调用大模型生成最终回答
        response = await llm.ainvoke(messages, config)
        final_text = response.content
        
        # 5. 🌟 关键：返回包含了新一轮问答的 history，触发 state.py 中的拦截器
        history_record = {
            "question": original_question,
            "answer": final_text
        }
        
        return {
            "final_answer": final_text,
            "history": [history_record] # LangGraph 会自动把它 append 到旧历史中
        }

    return final_answer_node