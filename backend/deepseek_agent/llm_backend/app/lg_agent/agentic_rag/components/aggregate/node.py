from typing import Any, Callable, Coroutine, Dict
from app.core.logger import get_logger

from ..state import OverallState

logger = get_logger(service="aggregate_node")

def create_aggregate_node() -> Callable[[OverallState], Coroutine[Any, Any, Dict[str, Any]]]:
    """创建结果物理无损拼接节点"""
    async def aggregate_node(state: OverallState) -> Dict[str, Any]:
        logger.info("✨ [Aggregate] 正在物理拼接知识库解答...")
        results = state.get("task_results", [])
        
        raw_context = "\n\n".join([
            f"🎯【子问题】: {r['question']}\n📑【知识库详尽解答】:\n{r.get('result', '')}" 
            for r in results
        ])
        
        return {"aggregated_answers": raw_context}

    return aggregate_node