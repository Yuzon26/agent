流程总结：
1. planner_node节点根据用户query进行任务分解，得到tasks
2. 然后条件边map_tasks_to_intent遍历tasks，为每个子任务task创建一个send任务，发送到intent_recognition_node，进行map，并行进行意图识别
3. 然后需要借助意图汇聚节点intent_recognition_node，让并行进行的意图识别结果汇聚
4. 然后从intent_recognition_node经过 route_based_on_intent 边给每个意图寻找工具：
    条件边 route_based_on_intent 遍历 intent_results ，为每个意图创建一个send任务，发送到不同的工具，解决每个意图对应的子任务
5. 最后所有子任务结果汇聚到结果节点final_answer_node，再给LLM总结回答。至此完成整个流程



# Map-Reduce 示例： 基于 Map-Reduce 的笑话生成系统

流程总结：
1. generate_topics 节点根据用户输入的总主题 (topic) 进行任务分解，得到多个子主题 (subjects)。
2. 然后条件边 continue_to_jokes 遍历 subjects，为每个子主题创建一个 send 任务，发送到 generate_joke 节点，进行 map，并行进行笑话生成。
3. 然后借助全局状态中的 operator.add 机制，让并行执行的 generate_joke 节点产生的结果安全汇聚到全局的 jokes 列表中。
4. 最后所有子任务生成的笑话汇聚到结果节点 best_joke，再交给 LLM 进行评估并选出最佳笑话。至此完成整个流程。


## 一、 项目概述与系统架构

**项目需求：** 构建一个智能笑话生成系统，演示 Map-Reduce 在 LangGraph 中的完整工作流。

* **Map 阶段（映射/发散）**：根据用户输入的主题（如”动物”），将其拆分为多个子主题，并**并行**为每个子主题生成笑话。
* **Reduce 阶段（归约/收束）**：从所有并行生成的笑话中，评估并选出最佳的一个。

**系统架构流转图：**

```text
用户输入主题 "animals"
        ↓
[generate_topics] 生成子主题 (如: mammals, reptiles, birds)
        ↓
  (Send API 动态分发)
        ↓
   ┌────────┬────────┬────────┐
   ↓        ↓        ↓        ↓
[joke-1] [joke-2] [joke-3]  (Map 阶段：多节点并行生成)
   ↓        ↓        ↓
   └────────┴────────┴────────┘
             ↓
        [best_joke]  (Reduce 阶段：评估并选择最佳)
             ↓
         返回最终结果

```

---

## 二、 核心状态设计 (State)

Map-Reduce 架构成功的关键在于**全局状态**与**局部状态**的分离。

### 1. 全局状态 (OverallState)

负责贯穿整个图的生命周期，收集并行节点的结果。

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel

class OverallState(TypedDict):
    topic: str                                  # 用户输入的主题
    subjects: list                              # 生成的子主题列表
    # 🌟 关键点：使用 operator.add reducer，允许多个并行节点同时安全追加数据
    jokes: Annotated[list, operator.add]        
    best_selected_joke: str                     # 最终选出的最佳笑话

```

### 2. 局部状态 (JokeState)

供单个并行节点使用的轻量级状态。

```python
class JokeState(TypedDict):
    subject: str  # 单个笑话生成节点只需要知道自己的子主题即可

```

> 💡 **知识点拓展：Pydantic BaseModel**
> Pydantic 的 `BaseModel` 用于数据验证、类型检查和结构化输出。
> 例如定义 `class Joke(BaseModel): joke: str` 后，通过 `model.with_structured_output(Joke).invoke(prompt)`，LLM 就会强制返回符合该结构的 JSON/对象数据。

---

## 三、 代码实现拆解

### Step 1: 初始化与提示词定义

定义了三个核心 Prompt 分别对应三大任务，并初始化大模型。

```python
from langchain_openai import ChatOpenAI

# 1. 拆分子主题
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
# 2. 生成单条笑话
joke_prompt = """Generate a joke about {subject}"""
# 3. 评选最佳笑话
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

model = ChatOpenAI(model="gpt-4o", temperature=0)

```

### Step 2: [Map 前置] 生成子主题

将大主题拆解为多个子主题，存入全局状态。

```python
class Subjects(BaseModel):
    subjects: list[str]

def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}

```

### Step 3: ✨ 核心魔法 —— Send API 动态任务分发

利用 `Send` API 根据子主题的数量，动态创建对应的并行任务。

```python
from langgraph.types import Send

def continue_to_jokes(state: OverallState):
    # 为每个子主题创建一个 Send 任务，派发给 "generate_joke" 节点
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

```

* **动态并行化**：无需预先知道任务数量（无论3个还是300个），运行时自动拉起等量的并行分支。
* **状态灵活**：派发时只需传入目标节点需要的局部状态（`{"subject": s}`），无需传递整个 `OverallState`。

### Step 4: [Map 阶段] 并行生成笑话

接收局部状态，生成笑话，并以**列表形式**返回，触发 `operator.add` 写入全局黑板。

```python
class Joke(BaseModel):
    joke: str

def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    # ⚠️ 注意：必须返回列表 ["xxx"]，才能被 operator.add 正确追加到 jokes 列表中
    return {"jokes": [response.joke]} 

```

### Step 5: [Reduce 阶段] 选出最佳笑话

等待所有 Map 节点完成后，收集所有笑话并进行最终裁决。

```python
class BestJoke(BaseModel):
    id: int

def best_joke(state: OverallState):
    # 1. 拼接所有收集到的笑话
    jokes = "\n\n".join(state["jokes"])

    # 2. 让 LLM 评估并返回最佳 ID
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)

    # 3. 提取最佳笑话并写回状态
    return {"best_selected_joke": state["jokes"][response.id]}

```

---

## 四、 构建与编译状态图 (Graph)

将上述节点和边组合起来，特别注意 `add_conditional_edges` 与 `Send` API 的结合使用。

```python
from langgraph.graph import END, StateGraph, START

# 1. 声明图与全局状态
graph = StateGraph(OverallState)

# 2. 添加所有节点
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)

# 3. 编排边 (Edges)
graph.add_edge(START, "generate_topics")

# 🌟 核心：使用条件边和 Send 函数实现动态并行分发 (Map)
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])

# 所有 generate_joke 执行完毕后，自动收束到 best_joke (Reduce)
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# 4. 编译图
app = graph.compile()

```