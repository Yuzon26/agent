from typing import List
from langgraph.types import Send
from ..components.state import OverallState, TaskState

def map_tasks_to_intent(state: OverallState) -> List[Send]:
    """并发分发：把每个拆分出的子问题并发推给意图识别节点"""
    return [
        Send("intent_recognition_node", {"question": task.question})
        for task in state.get("tasks", [])
    ]

# 🌟 新增这个条件边函数
def route_based_on_intent(state: OverallState) -> List[Send]:
    """条件边：根据意图状态，使用 Send 动态分发给对应的 RAG 节点"""
    # intent = state.get("intent", "")
    # question = state.get("question", "")
    
    # if intent in ["causality", "summary"]:
    #     return Send("kg_rag_node", {"question": question})
    # else:
    #     return Send("hybrid_rag_node", {"question": question})
    
    sends = []
    for item in state.get("intent_results", []):
        if item["intent"] in ["causality", "summary"]:
            sends.append(Send("kg_rag_node", {"question": item["question"]}))
        else:
            sends.append(Send("hybrid_rag_node", {"question": item["question"]}))
    return sends