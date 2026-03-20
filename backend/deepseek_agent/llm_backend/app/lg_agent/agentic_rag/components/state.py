from operator import add
from typing import Annotated, Any, Dict, List, Literal, TypedDict
from pydantic import BaseModel, Field

# ==========================================
# 模型与状态定义
# ==========================================
# 🌟 1. 定义单轮历史记录结构
class HistoryRecord(TypedDict):
    question: str
    answer: str

# 🌟 2. 定义滑动窗口聚合函数
def update_history(history: List[HistoryRecord], new: List[HistoryRecord]) -> List[HistoryRecord]:
    SIZE: int = 5
    if history is None:
        history = []
    history.extend(new)
    return history[-SIZE:]

class SubTask(BaseModel):
    question: str = Field(description="拆分出的独立子问题")

class PlannerOutput(BaseModel):
    tasks: List[SubTask] = Field(default=[], description="拆分出的子任务列表")

class IntentOutput(BaseModel):
    intent: Literal["fact", "causality", "summary"] = Field(description="问题类型分类")

class TaskState(TypedDict):
    """单个子任务流转状态"""
    question: str
    # intent: str  # 🌟 新增：用于存放识别出来的意图

# ----------------- 1. InputState (输入状态) -----------------
# 职责：定义启动这个工作流【必须提供】的初始参数
class InputState(TypedDict):
    original_question: str  # 用户原始问题
    # history 不需要外部手动传，MemorySaver 会自动注水，但需要在这里声明以保证类型完整
    history: Annotated[List[HistoryRecord], update_history] 

# ----------------- 2. OutputState (输出状态) -----------------
# 职责：定义工作流跑完后，【最终暴露】给前端/外部系统的干净数据
class OutputState(TypedDict):
    final_answer: str       # 最终回答
    # 必须保留 history，这样图运行结束时才能将最新的对话写入 MemorySaver
    history: Annotated[List[HistoryRecord], update_history] 

# ----------------- 3. OverallState (总状态/内部运转状态) -----------------
# 职责：包含 Input + Output + 所有内部节点流转所需的【中间变量】
class OverallState(TypedDict):
    original_question: str
    history: Annotated[List[HistoryRecord], update_history]
    
    # --- 以下为内部中间变量，外部不可见 ---
    tasks: List[SubTask] 
    # 并发执行时，必须使用 add 聚合结果，防止覆盖
    task_results: Annotated[List[Dict[str, str]], add] 
    intent_results: Annotated[List[Dict[str, Any]], add]
    aggregated_answers: str
    final_answer: str

# 原来用主图全局状态
class MainGraphState(TypedDict):
    """主图全局状态"""
    original_question: str
    # 🌟 3. 注册历史记录状态，并绑定滑动窗口聚合函数
    history: Annotated[List[HistoryRecord], update_history]
    tasks: List[SubTask]
    # 使用 add 聚合所有并发 RAG 返回的结果
    task_results: Annotated[List[Dict[str, str]], add] 
    aggregated_answers: str
    final_answer: str