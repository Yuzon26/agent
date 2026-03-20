from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

##YUZON  定义了一个名为 Router 的 TypedDict 类，用于表示用户查询的分类结果。
class Router(TypedDict):
#Yuzon 
# Router
# │
# ├─ logic      分类逻辑说明
# ├─ type       查询类型
# └─ question   用户问题（默认 ""）
    """Classify user query."""
    logic: str
    type: Literal["general-query", "additional-query", "graphrag-query", "image-query", "file-query"] # type: Literal[...] 限制了 type 字段的值只能是列表中的那五个字符串之一
    question: str = field(default_factory=str) # field(default_factory=str) 确保每个实例在创建时都会调用一次 default_factory(用于生成默认值的函数) 来生成一个新的字符串，防止所有实例会共享同一个字符串对象
    #YUZON  定义一个字符串类型的字段 question，默认值为空字符串。这个字段用于存储用户的查询问题。 
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )#YUZON  定义一个字符串类型的字段 binary_score，用于表示生成的答案中是否存在幻觉（hallucination），'1' 表示存在幻觉，'0' 表示不存在幻觉。

# @dataclass(kw_only=True)： 强制要求数据类中的所有字段必须以关键字参数的形式提供。即不能以位置参数的方式传递。
@dataclass(kw_only=True)

##YUZON                  定义Agent输入状态
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. 
    """

    messages: Annotated[list[AnyMessage], add_messages]
    #YUZON  
    # 定义一个字段 messages，它是一个列表，列表中的元素可以是任何类型的消息（AnyMessage）。
    # 这个字段使用了 Annotated 来添加元数据，元数据是 add_messages，这可能是一个函数或类，用于处理或验证消息列表。
    
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    

    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.
    

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""
    

# @dataclass(kw_only=True)： 强制要求数据类中的所有字段必须以关键字参数的形式提供。即不能以位置参数的方式传递。
# 在实例化这个类时，你不能只传数值（位置参数），必须明确写出 变量名=数值（实参对应形参）。即"形参=实参"的形式
# AgentState 继承了 InputState。它除了有聊天记录（messages），还增加了 router（用来存大模型的分类结果）、steps（用来存思考步骤）等
@dataclass(kw_only=True)

##YUZON  定义一个名为 AgentState 的数据类，继承自 InputState。这个类表示检索图/代理的状态。
class AgentState(InputState):
    #Yuzon 
    # AgentState
    # │
    # ├─ messages        对话历史
    # ├─ router          Router分类结果
    # ├─ steps           执行步骤
    # ├─ question        用户问题
    # ├─ answer          最终回答
    # └─ hallucination   幻觉检测结果
    """State of the retrieval graph / agent."""
    # default_factory 要求接收一个**“可调用对象”（Callable）**，也就是一个函数
    # 如果是简单的类型：你可以写 default_factory=list 或 default_factory=dict
    # 如果是带参数的初始化：比如你想让默认的 Router 带有特定的 type 和 logic 值，你就不能直接传 Router 这个类名了，因为你需要传参数
    router: Router = field(default_factory=lambda: Router(type="general-query", logic=""))
    #YUZON  
    # 定义一个字段 router存储Router分类结果.
    # 它的类型是 Router，type 字段的默认值为 "general-query"，logic 字段的默认值为空字符串。
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    #YUZON  定义一个字段 steps，它是一个字符串列表，默认值为空列表，用于记录Agent执行过程。
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    question: str = field(default_factory=str) 
    answer: str = field(default_factory=str)  
    hallucination: GradeHallucinations = field(default_factory=lambda: GradeHallucinations(binary_score="0"))
    #YUZON  默认认为没有验证到幻觉。
