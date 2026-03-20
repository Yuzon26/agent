# app/services/agentic_rag_service.py

import json
from typing import List, Dict, AsyncGenerator, Optional, Callable
from app.core.logger import get_logger

# 引入你的 LangGraph 工作流和 LLM (请根据你的实际路径调整 import)
from app.lg_agent.agentic_rag.workflows.workflow import agentic_rag_workflow, llm
import uuid
def new_uuid():
    return str(uuid.uuid4())

logger = get_logger(service="AgenticRAGService")

class AgenticRAGService:
    def __init__(self):
        logger.info("Initializing Agentic RAG Service")

    # TODO 这个函数是不是可以移到别的文件去？
    async def generate_stream(
        self, 
        query: str,
        messages: List[Dict],
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        on_complete: Optional[Callable] = None
    ) -> AsyncGenerator[str, None]:
        """流式运行 LangGraph RAG 工作流，并生成最终回复"""
        # 检查前端发来的内容
        logger.info(f"📥 收到前端的 messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        try:
            # 1. 发送初始进度提示
            init_msg = json.dumps("⏳ 【系统】正在分析您的问题...\n", ensure_ascii=False)
            yield f"data: {init_msg}\n\n"
            
            # final_state = None
            
            # # 2. 运行工作流，监听中间节点状态
            # async for event in agentic_rag_workflow.astream(
            #     {
            #         "original_question": query, 
            #         "task_results": []
            #     }, 
            #     stream_mode="updates"
            # ):
            #     for node_name, node_state in event.items():
            #         status_msg = ""
            #         if node_name == "planner":
            #             status_msg = "🧩 【系统】已将问题拆解为多个子任务...\n"
            #         elif node_name in ["kg_rag_node", "hybrid_rag_node"]:
            #             status_msg = "📚 【系统】正在深度检索底层知识库...\n"
            #         elif node_name == "aggregate_node":
            #             status_msg = "✨ 【系统】情报提取完毕，正在生成最终回答：\n\n---\n\n"
                    
            #         if status_msg:
            #             json_data = json.dumps(status_msg, ensure_ascii=False)
            #             yield f"data: {json_data}\n\n"
                    
            #         final_state = node_state

            # # 3. 获取聚合后的无损上下文
            # aggregated_answers = final_state.get("aggregated_answers", "") if final_state else ""
            
            # # 4. 构造最终的 Prompt
            # prompt = f"""你是一个顶级的军事与情报分析专家。
            # 为了准确回答用户的原始问题，我们已经将其拆分为多个子问题，并派发给底层的专业知识图谱与向量库进行深度检索。
            # 以下是各个底层知识库传回的【详尽子解答】：
            
            # ================================
            # {aggregated_answers}
            # ================================
            
            # 现在，请你统筹全局，综合以上所有情报，全面、专业地回答用户的最终问题：
            # 【用户问题】：{query}
            
            # 执行要求：
            # 1. 必须充分利用上述情报中的事实细节（如具体数值、坐标、型号、时间等），绝对不要遗漏关键数据。
            # 2. 不要机械地罗列，请将分散的情报融会贯通，形成逻辑严密、条理清晰的专业回复。
            # 3. 语气要保持专业、客观。如果底层解答中信息有缺失，请如实说明。
            # """
            
            # 🌟 1. 配置对话线程 ID：LangGraph 靠这个区分不同用户的记忆
            # 这里不需要用user_id来作为thread_id的一部分吗？
            # 不需要，因为conversation_id是全局唯一的键，由会话管理那部分代码负责
            thread_id = conversation_id if conversation_id else new_uuid()
            config = {"configurable": {"thread_id": thread_id}}

            full_response = []
            
           # 🌟 2. 启动图，使用 stream_mode="messages" 捕获节点内部 LLM 的流式输出
            # 此时前端只传了 query（messages[-1]），我们直接用 query 即可
            # async for msg, metadata in agentic_rag_workflow.astream(
            #     {"original_question": query}, 
            #     config=config,
            #     stream_mode="messages"
            # ):
            #     # 过滤出是由 final_answer_node 里的 LLM 产生的消息块
            #     if metadata.get("langgraph_node") == "final_answer_node":
            #         if msg.content:
            #             full_response.append(msg.content)
            #             chunk_json = json.dumps(msg.content, ensure_ascii=False)
            #             yield f"data: {chunk_json}\n\n"

            # 同时开启 "updates" 和 "messages" 两种监听模式
            async for mode, payload in agentic_rag_workflow.astream(
                {"original_question": query}, 
                config=config,
                stream_mode=["updates", "messages"]
            ):
                # ==========================================
                # 模式 A：处理节点流转状态 (原先被注释掉的代码)
                # ==========================================
                if mode == "updates":
                    # 此时 payload 是一个字典，如 {"planner": {...}}
                    for node_name, node_state in payload.items():
                        status_msg = ""
                        if node_name == "planner":
                            status_msg = "🧩 【系统】已将问题拆解为多个子任务...\n"
                        elif node_name in ["kg_rag_node", "hybrid_rag_node"]:
                            status_msg = "📚 【系统】正在深度检索底层知识库...\n"
                        elif node_name == "aggregate_node":
                            status_msg = "✨ 【系统】情报提取完毕，正在生成最终回答：\n\n---\n\n"
                        
                        if status_msg:
                            json_data = json.dumps(status_msg, ensure_ascii=False)
                            yield f"data: {json_data}\n\n"

                # ==========================================
                # 模式 B：处理最终答案的流式打字机输出
                # ==========================================
                elif mode == "messages":
                    # 此时 payload 是一个元组 (msg, metadata)
                    msg, metadata = payload
                    
                    # 只过滤出 final_answer_node 里生成的内容
                    if metadata.get("langgraph_node") == "final_answer_node":
                        if msg.content:
                            full_response.append(msg.content)
                            chunk_json = json.dumps(msg.content, ensure_ascii=False)
                            yield f"data: {chunk_json}\n\n"
            
            # 6. 回调执行：保存历史记录
            if on_complete:
                complete_response = "".join(full_response)
                await on_complete(user_id, conversation_id, messages, complete_response)
                logger.info(f"Successfully saved RAG conversation history for user {user_id}")

        except Exception as e:
            # logger.error(f"RAG Stream generation error: {str(e)}", exc_info=True)
            logger.error("RAG Stream generation error: {}", str(e), exc_info=True)
            error_msg = json.dumps(f"\n\n❌ 生成回复时出错: {str(e)}", ensure_ascii=False)
            yield f"data: {error_msg}\n\n"
            # 抛出异常让上层捕获
            raise