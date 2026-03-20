import sys
import os
from pathlib import Path

# 将项目根目录添加到 Python 路径
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# YUZON
# __file__ 是 Python 的一个变量，表示 当前脚本文件的路径。
# .parent 表示 上一级目录。
# Path 是 Python 的 路径对象类，用来处理文件路径，比 os.path 更方便。

from app.lg_agent.lg_states import AgentState, InputState
from app.lg_agent.utils import new_uuid
from app.lg_agent.lg_builder import graph
from langgraph.types import Command
import asyncio
import time
import builtins

thread = {"configurable": {"thread_id": new_uuid()}}


async def process_query(query):
    inputState = InputState(messages=query)

    async for c, metadata in graph.astream(input=inputState, stream_mode="messages", config=thread):
        # if c.additional_kwargs.get("tool_calls"):
        #     print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)

        if c.content and "research_plan" not in metadata.get("tags", []):
            print(c.content, end="", flush=True)
        # if c.content:
        #     print(c.content, end="", flush=True)
    # async for c in graph.astream(input=inputState, stream_mode="values", config=thread):
    #     print(c, end="", flush=True)

    if len(graph.get_state(thread)[-1]) > 0:
        if len(graph.get_state(thread)[-1][0].interrupts) > 0:
            response = input('\n响应可能包含不确定信息。重试生成？如果是，按"y"：')
            if response.lower() == 'y':
                async for c, metadata in graph.astream(Command(resume=response), stream_mode="messages", config=thread):
                    if c.additional_kwargs.get("tool_calls"):
                        print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="")
                    if c.content:
                        time.sleep(0.05)
                        print(c.content, end="", flush=True)


async def main():
    input = builtins.input
    while True:
        query = input("> ")
        if query.strip().lower() == "q":
            print("Exiting...")
            break
        await process_query(query)


if __name__ == "__main__":
    asyncio.run(main())
