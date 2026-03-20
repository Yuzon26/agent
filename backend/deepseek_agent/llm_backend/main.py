from unittest import result

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.llm_factory import LLMFactory
from app.services.search_service import SearchService
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from pathlib import Path

from app.core.logger import get_logger, log_structured
from app.core.middleware import LoggingMiddleware
from app.core.config import settings
from app.api import api_router
from app.core.database import AsyncSessionLocal
from app.models.conversation import Conversation, DialogueType
from app.models.message import Message
from sqlalchemy import select
from app.services.conversation_service import ConversationService
import uuid
import os
from app.services.indexing_service import IndexingService
import sys
from app.lg_agent.lg_states import AgentState, InputState
from app.lg_agent.utils import new_uuid
from app.lg_agent.lg_builder import graph
from langgraph.types import Command
import json


# 配置上传目录 - RAG 功能的
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# logger 变量就被初始化为一个日志记录器实例。
# 之后，便可以在当前文件中直接使用 logger.info()、logger.error() 等方法来记录日志，而不需要进行其他操作。
logger = get_logger(service="main")

# 创建 FastAPI 应用实例
app = FastAPI(title="AssistGen REST API")

# 添加日志中间件， 使用 LoggingMiddleware 来统一处理日志记录，从而替代 FastAPI 的原生打印日志。
app.add_middleware(LoggingMiddleware)

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中要设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 用户注册、登录路由通过 api_router 路由挂载到 /api 前缀
app.include_router(api_router, prefix="/api")

class ReasonRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: int

class ChatMessage(BaseModel):
    messages: List[Dict[str, str]]
    user_id: int
    conversation_id: int  # 添加会话ID字段

class RAGChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    index_id: str
    user_id: int

class CreateConversationRequest(BaseModel):
    user_id: int

class UpdateConversationNameRequest(BaseModel):
    name: str

class LangGraphRequest(BaseModel):
    query: str
    user_id: int
    conversation_id: Optional[str] = None
    image: Optional[UploadFile] = None

class LangGraphResumeRequest(BaseModel):
    query: str
    user_id: int
    conversation_id: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}

# -----------基础对话类接口-----------
# 【chris】这几个接口没有经过 LangGraph 智能体，而是直接调用了 LLMFactory 里的基础聊天服务，实现了流式响应 (StreamingResponse)

# TODO 看看这个llamafactory是怎么用的
@app.post("/api/chat")
async def chat_endpoint(request: ChatMessage):
    """聊天接口"""
    try:
        logger.info(f"Processing chat request for user {request.user_id} in conversation {request.conversation_id}")
        chat_service = LLMFactory.create_chat_service()
        
        return StreamingResponse(
            chat_service.generate_stream(
                messages=request.messages,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                on_complete=ConversationService.save_message
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# TODO 这里有点小问题
# 哪怕你是从 /api/chat 进来的，只要调用了这个流式生成的函数generate_stream，它都会强行把大模型切换成推理模型（self.reason_model）。
# 而在非流式的 generate 函数里，作者又写死了使用聊天模型
@app.post("/api/reason")
async def reason_endpoint(request: ReasonRequest):
    """推理接口"""
    try:
        logger.info(f"Processing reasoning request for user {request.user_id}")
        reasoner = LLMFactory.create_reasoner_service()
        
        log_structured("reason_request", {
            "user_id": request.user_id,
            "message_count": len(request.messages),
            "last_message": request.messages[-1]["content"][:100] + "..."
        })
        
        return StreamingResponse(
            reasoner.generate_stream(request.messages),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Reasoning error for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_endpoint(request: ChatMessage):
    """带搜索功能的聊天接口"""
    try:
        logger.info(f"Processing search request for user {request.user_id} in conversation {request.conversation_id}")
        logger.info(f"Request: {request}")
        search_service = LLMFactory.create_search_service()
        return StreamingResponse(
            search_service.generate_stream(
                query=request.messages[0]["content"],
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                # on_complete=ConversationService.save_message
            ),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------- 你的专属 LlamaIndex RAG 接口 -----------
# 确保在文件顶部导入你的服务

from app.services.agentic_rag_service import AgenticRAGService
from app.services.conversation_service import ConversationService # 用于保存历史记录
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
import json


# ----------- 你的专属 LlamaIndex RAG 接口 -----------
@app.post("/api/rag")
async def rag_endpoint(request: ChatMessage):
    """基于 LlamaIndex 的本地知识库问答专属接口"""
    try:
        logger.info(f"Processing RAG request for user {request.user_id} in conversation {request.conversation_id}")
        
        rag_service = AgenticRAGService()
        
        # 提取用户的最新提问
        query = request.messages[-1]["content"] if request.messages else ""
        if not query:
            raise ValueError("提问内容不能为空")

        # 2. 直接调用 Service，并将 ConversationService.save_message 作为回调函数传入
        # 这样就能完美复用原有 chat 接口的记忆保存功能！
        return StreamingResponse(
            rag_service.generate_stream(
                query=query,
                messages=request.messages,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                on_complete=ConversationService.save_message  # 核心：注入保存记录的回调方法
            ),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"LlamaIndex RAG error for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# -----------文件上传与 RAG 接口-----------
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: int = Form(...)
):
    """上传文件并准备 RAG 处理"""
    try:
        logger.info(f"Uploading file for user {user_id}: {file.filename}")
        # 为了防止不同用户传了同名文件互相覆盖，设计“两级目录结构”：
        # 1. 创建基于UUID的一级目录
        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))
        first_level_dir = UPLOAD_DIR / user_uuid
        
        # 2. 创建基于时间戳的二级目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        second_level_dir = first_level_dir / timestamp
        second_level_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 生成带时间戳的文件名
        # 把原来的文件名和时间戳拼在一起（比如 战术手册_20260310_153022.pdf），彻底杜绝文件名冲突
        original_name, ext = os.path.splitext(file.filename)
        new_filename = f"{original_name}_{timestamp}{ext}"
        file_path = second_level_dir / new_filename
        
        # 保存文件
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"file_path: {str(file_path)}")
            
        # 获取文件信息
        file_info = {
            "filename": new_filename,
            "original_name": file.filename,
            "size": len(content),
            "type": file.content_type,
            "path": str(file_path).replace('\\', '/'),
            "user_id": user_id,
            "user_uuid": user_uuid,
            "upload_time": timestamp,
            "directory": str(second_level_dir)
        }
        
        # # 4. 处理文件索引
        # indexing_service = IndexingService()
        # index_result = await indexing_service.process_file(file_info)
        
        
        #YUZON 
        #         file_info = {
        #     "filename": new_filename,
        #     "original_name": file.filename,
        #     "size": len(content),
        #     "type": file.content_type,
        #     "path": str(file_path),
        #     "user_id": user_id,
        #     "user_uuid": user_uuid,
        #     "upload_time": timestamp,
        #     "directory": str(second_level_dir)
        # }
        #index_result = {
        #     'original_file_path': file_path,
        #     'input_file_path': input_file_path,
        #     'file_type': file_type,
        #     'config_used': config_file,
        #     'is_update': is_update,
        #     'status': 'success',  #前端要看
        #     'user_id': user_id,
        #     'input_dir': user_input_dir,
        #     'output_dir': user_output_dir
        # }

        
        # # 合并结果
        # result = {**file_info, "index_result": index_result}
        result = {**file_info, "status": 'success'}
        
        return result
 
    
    except Exception as e:
        logger.error(f"Upload failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# TODO 这个好像是过时的代码部分,后面确认下,传统的RAG是怎么进去的
# 理论上它应该和“知识库问答”按钮一起作用的，但是前端的这个按钮没用，后面看看
@app.post("/chat-rag")
async def rag_chat_endpoint(request: RAGChatRequest):
    """基于文档的问答接口"""
    try:
        logger.info(f"Processing RAG chat request for user {request.user_id}")
        rag_chat_service = RAGChatService()
        
        return StreamingResponse(
            rag_chat_service.generate_stream(
                request.messages,
                request.index_id
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"RAG chat error for user {request.user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------会话历史管理接口 (CRUD)-----------
@app.post("/api/conversations")
async def create_conversation(request: CreateConversationRequest):
    """创建新会话"""
    try:
        conversation_id = await ConversationService.create_conversation(request.user_id)
        return {"conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/user/{user_id}")
async def get_user_conversations(user_id: int):
    """获取用户的所有会话"""
    try:
        conversations = await ConversationService.get_user_conversations(user_id)
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: int, user_id: int):
    """获取会话的所有消息"""
    try:
        messages = await ConversationService.get_conversation_messages(conversation_id, user_id)
        return messages
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """删除会话及其所有消息"""
    try:
        conversation_service = ConversationService()
        await conversation_service.delete_conversation(conversation_id)
        return {"message": "会话已删除"}
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/conversations/{conversation_id}/name")
async def update_conversation_name(
    conversation_id: int,
    request: UpdateConversationNameRequest
):
    """修改会话名称"""
    try:
        conversation_service = ConversationService()
        await conversation_service.update_conversation_name(conversation_id, request.name)
        return {"message": "会话名称已更新"}
    except Exception as e:
        logger.error(f"更新会话名称失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------最核心的 LangGraph 智能体接口-----------
@app.post("/api/langgraph/query")
async def langgraph_query(
    query: str = Form(...),
    user_id: int = Form(...),
    conversation_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """使用LangGraph处理用户查询，支持图片上传"""
    try:
        logger.info(f"Processing LangGraph query for user {user_id} and conversation {conversation_id}")
        
        # 处理图片上传
        image_path = None
        if image:
            # 创建图片存储目录
            image_dir = Path("uploads/images")
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name, ext = os.path.splitext(image.filename)
            new_filename = f"{original_name}_{timestamp}{ext}"
            image_path = image_dir / new_filename
            
            # 保存图片
            content = await image.read()
            with open(image_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Saved image {new_filename} for user {user_id}")
        
        # 使用conversation_id作为thread_id，如果没有提供则创建新的
        thread_id = conversation_id if conversation_id else new_uuid()
        thread_config = {
            "configurable": {
                "thread_id": thread_id, 
                "user_id": user_id,
                "image_path": str(image_path) if image_path else None
            }
        }
        
        # 获取当前线程状态
        state_history = None
        try:
            # 检查是否有现有的会话状态
            if thread_id:
                state_history = graph.get_state(thread_config)
                if state_history:
                    logger.info(f"Found existing conversation state for thread_id: {thread_id}")
        except Exception as e:
            logger.warning(f"Error retrieving state: {e}. Starting with fresh state.")
        
        # 准备输入状态 - 如果是现有会话，直接传入查询文本
        if state_history and len(state_history) > 0 and len(state_history[-1]) > 0:
            logger.info("Using existing conversation state")
            # 如果有现有会话，使用resume命令继续对话
            async def process_stream():
                # 这里的graph指的是langgraph的整个图结构
                # TODO 这里的用法怎么理解？
                async for c, metadata in graph.astream(
                    Command(resume=query), 
                    stream_mode="messages", 
                    config=thread_config
                ):
                    # 只处理最终展示给用户的内容，跳过中间工具调用和内部状态
                    # 只有当它在正经说话（c.content存在），且不是在调用工具时，才发给前端
                    if c.content and "research_plan" not in metadata.get("tags", []) and not c.additional_kwargs.get("tool_calls"):
                        # 关键修改：使用json.dumps处理content，确保特殊字符如换行符被正确处理
                        content_json = json.dumps(c.content, ensure_ascii=False)
                        yield f"data: {content_json}\n\n"
                        
                    # 工具调用单独处理，不发送给前端
                    # 如果它在调用工具（比如查数据库、搜图谱），只在后端打日志，不告诉用户
                    elif c.additional_kwargs.get("tool_calls"):
                        tool_data = c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                        logger.debug(f"Tool call: {tool_data}")
                        
                # 处理中断情况
                state = graph.get_state(thread_config)
                if len(state) > 0 and len(state[-1]) > 0:
                    if len(state[-1][0].interrupts) > 0:
                        interrupt_json = json.dumps({"interruption": True, "conversation_id": thread_id})
                        yield f"data: {interrupt_json}\n\n"
        else:
            # 新会话或找不到现有状态，创建新的输入状态
            logger.info("Creating new conversation state")
            input_state = InputState(messages=query)
            
            # 流式处理查询
            async def process_stream():
                async for c, metadata in graph.astream(
                    input=input_state, 
                    stream_mode="messages", 
                    config=thread_config
                ):
                    # 只处理最终展示给用户的内容，跳过中间工具调用和内部状态
                    if c.content and "research_plan" not in metadata.get("tags", []) and not c.additional_kwargs.get("tool_calls"):
                        # 关键修改：使用json.dumps处理content，确保特殊字符如换行符被正确处理
                        content_json = json.dumps(c.content, ensure_ascii=False)
                        yield f"data: {content_json}\n\n"
                        
                    # 工具调用单独处理，不发送给前端
                    elif c.additional_kwargs.get("tool_calls"):
                        tool_data = c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                        logger.debug(f"Tool call: {tool_data}")
                        
                # 处理中断情况
                state = graph.get_state(thread_config)
                if len(state) > 0 and len(state[-1]) > 0:
                    if len(state[-1][0].interrupts) > 0:
                        interrupt_json = json.dumps({"interruption": True, "conversation_id": thread_id})
                        yield f"data: {interrupt_json}\n\n"
        
        response = StreamingResponse(
            process_stream(),
            media_type="text/event-stream"
        )
        
        # 添加会话ID到响应头，方便前端获取
        response.headers["X-Conversation-ID"] = thread_id
        
        return response
        
    except Exception as e:
        logger.error(f"LangGraph query error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/langgraph/resume")
async def langgraph_resume(request: LangGraphResumeRequest):
    """继续执行LangGraph流程"""
    try:
        logger.info(f"Resuming LangGraph query for user {request.user_id} with conversation {request.conversation_id}")
        
        # 使用会话ID作为线程ID
        thread_config = {"configurable": {"thread_id": request.conversation_id}}
        
        # 流式处理恢复
        async def process_resume():
            async for c, metadata in graph.astream(Command(resume=request.query), stream_mode="messages", config=thread_config):
                # 只处理最终展示给用户的内容
                if c.content and not c.additional_kwargs.get("tool_calls"):
                    # 同样使用json.dumps处理内容
                    content_json = json.dumps(c.content, ensure_ascii=False)
                    yield f"data: {content_json}\n\n"
                
                # 工具调用单独处理，不发送给前端
                elif c.additional_kwargs.get("tool_calls"):
                    tool_data = c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments")
                    logger.debug(f"Tool call: {tool_data}")
        
        return StreamingResponse(
            process_resume(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"LangGraph resume error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# -----------图片上传接口-----------
@app.post("/api/upload/image")
async def upload_image(
    image: UploadFile = File(...),
    user_id: int = Form(...),
    conversation_id: Optional[str] = Form(None)
):
    """上传图片并返回图片存储路径"""
    try:
        # 创建图片存储目录
        image_dir = Path("uploads/images")
        if conversation_id:
            image_dir = image_dir / conversation_id
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name, ext = os.path.splitext(image.filename)
        new_filename = f"{original_name}_{timestamp}{ext}"
        image_path = image_dir / new_filename
        
        # 保存图片
        content = await image.read()
        with open(image_path, "wb") as f:
            f.write(content)
        
        # 获取图片信息
        image_info = {
            "filename": new_filename,
            "original_name": image.filename,
            "size": len(content),
            "type": image.content_type,
            "path": str(image_path).replace('\\', '/'),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "upload_time": timestamp
        }
        
        logger.info(f"Image uploaded: {image_info}")
        
        return image_info
        
    except Exception as e:
        logger.error(f"Image upload failed for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 前端页面的托管
# 它把 Vue.js 编译打包好的前端静态文件（index.html, js, css）挂载到了网站的根路径 / 上。这意味着你在生产环境部署时，不需要单独搞一个 Nginx，只要启动了 run.py，访问 http://你的IP:8000 就能直接打开漂亮的前端聊天界面了
# 最后挂载静态文件，并确保使用绝对路径
STATIC_DIR = Path(__file__).parent / "static" / "dist"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
