"""
A tool_selection node must
* take a single task at a time
* retrieve a list of available tools
    * text2cypher
    * custom pre-written cypher executors
        * these can be numerous and may be retrieved in the same fashion as CypherQuery node contents
    * unstructured text search (sim search)
* decide the appropriate tool for the task
* generate and validate parameters for the selected tool
* send the validated parameters to the appropriate tool node
"""

from typing import Any, Callable, Coroutine, Dict, List, Literal, Set
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables.base import Runnable
from langgraph.types import Command, Send
from pydantic import BaseModel


from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.state import ToolSelectionInputState
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.tool_selection.prompts import create_tool_selection_prompt_template

# 定义工具选择提示词
tool_selection_prompt = create_tool_selection_prompt_template()


# 声明式的使用可配置模型：https://python.langchain.com/docs/how_to/chat_models_universal_init/#using-a-configurable-model-declaratively
def create_tool_selection_node(
    llm: BaseChatModel,
    tool_schemas: List[type[BaseModel]],
    default_to_text2cypher: bool = True,
) -> Callable[[ToolSelectionInputState], Coroutine[Any, Any, Command[Any]]]:
    """
    Create a tool_selection node to be used in a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM used to process data.
    tool_schemas : Sequence[Union[Dict[str, Any], type, Callable, BaseTool]
        tools schemas that inform the LLM which tools are available.
    default_to_text2cypher : bool, optional
        Whether to attempt Text2Cypher if no tool calls are returned by the LLM, by default True

    Returns
    -------
    Callable[[ToolSelectionInputState], ToolSelectionOutputState]
        The LangGraph node.
    """

    # 构建工具选择链，由大模型根据传递过来的 Task，在预定义的工具列表中选择一个工具。
    tool_selection_chain: Runnable[Dict[str, Any], Any] = (
        tool_selection_prompt
        # bind_tools 是 LangChain 核心库封装的一个标准化方法，用于将工具绑定到语言模型（如 OpenAI 的聊天模型）上，使模型能够调用这些工具来处理复杂任务。LLM 运行后会返回一个特殊的 tool_calls 结构，里面包含了工具名和提取出的参数。
        | llm.bind_tools(tools=tool_schemas) 
        # 模型返回的原始数据通常是复杂的 JSON 字符串，利用这个parser强行把模型输出的 JSON 数据重新实例化为你在 tool_schemas 中定义的 Pydantic 对象。
        # first_tool_only=True：告诉解析器，哪怕模型脑热想一次调用三个工具，我们也只取第一个最重要的。
        | PydanticToolsParser(tools=tool_schemas, first_tool_only=True) 
    )

    # 从传入的tool_schemas列表中，获取每个工具的title属性，创建出一个工具名称集合。
    predefined_cypher_tools: Set[str] = {
        t.model_json_schema().get("title", "") for t in tool_schemas
    }

    # 以下注释-> Command[Literal["cypher_query", ...]]这说明代码可能正在经历从 text2cypher 到 cypher_query 的更名过程，go_to_text2cypher 可能在重构时被不小心删除了。
    # async def tool_selection(
    #     state: ToolSelectionInputState,
    # ) -> Command[Literal["text2cypher", "predefined_cypher", "customer_tools"]]:


    async def tool_selection(
        state: ToolSelectionInputState,
    ) -> Command[Literal["cypher_query", "predefined_cypher", "customer_tools"]]:
        """
        Choose the appropriate tool for the given task.
        """
        # 调用工具选择链，生成针对每个任务要调用的工具名称和参数
        tool_selection_output: BaseModel = await tool_selection_chain.ainvoke(
            {"question": state.get("question", "")}
        )

        # 根据路由到对应的工具节点
        if tool_selection_output is not None:
            # title 字段默认就是你的 Pydantic 类的名字。它既不是在 parser 中定义的，也不是在大模型里定义的，而是在你定义 Pydantic 类（工具 Schema） 时，由 Pydantic 自动生成的。 
            tool_name: str = tool_selection_output.model_json_schema().get("title", "")
            # tool_args：这是把 AI 填好的 Pydantic 对象转化成一个普通的 Python 字典。比如 AI 填了 order_id=123，model_dump() 就会把它变成 {"order_id": 123}，方便作为参数传递。
            tool_args: Dict[str, Any] = tool_selection_output.model_dump() 
            if tool_name == "predefined_cypher":
                # 在早期的 LangGraph 中，节点只能返回一个字典来更新状态（State），至于下一步去哪，完全由外面定义的“边（Edge）”说了算。
                # Command 的用法是： 允许你在节点函数内部，根据计算的结果，动态地发出指令。它不仅可以更新数据，还可以直接控制程序的跳转方向
                # goto 参数是 Command 的核心，它代表了“下一站去哪。
                return Command(
                    goto=Send(
                        "predefined_cypher",
                        {
                            "task": state.get("question", ""),
                            "query_name": tool_name,
                            "query_parameters": tool_args,
                            "steps": ["tool_selection"],
                        },
                    )
                )
            elif tool_name == "cypher_query":
                return Command(
                    goto=Send(
                        "cypher_query",
                        {
                            "task": state.get("question", ""),
                            "query_name": tool_name,
                            "query_parameters": tool_args,
                            "steps": ["tool_selection"],
                        },
                    )
                )
            
            else:
                return Command(
                    goto=Send(
                        "customer_tools",
                        {
                            "task": state.get("question", ""),
                            "query_name": tool_name,
                            "query_parameters": tool_args,
                            "steps": ["tool_selection"],
                        },
                    )
                )


           
        # TODO 这里应该有问题，缺少了go_to_text2cypher的定义，需要补全一个command给go_to_text2cypher
        elif default_to_text2cypher:
            return go_to_text2cypher

        # handle instance where no tool is chosen
        else:
            return Command(
                goto=Send(
                    "error_tool_selection",
                    {
                        "task": state.get("question", ""),
                        "errors": [
                            f"Unable to assign tool to question: `{state.get('question', '')}`"
                        ],
                        "steps": ["tool_selection"],
                    },
                )
            )

        return go_to_text2cypher

    return tool_selection
