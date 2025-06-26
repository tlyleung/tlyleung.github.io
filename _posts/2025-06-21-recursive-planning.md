---
layout: post
title: Extending Anthropic's Agent Workflows with Recursive Planning
description: Custom agent design using Google's new ADK framework.
authors: [tlyleung]
x: 8
y: 67
---

Back in April 2025 at ICLR in Singapore, I was introduced to the Google Agent Development Kit (ADK)[^google-adk] by a demo at Google's booth. ADK is a framework for building AI agents with reusable orchestration components. Up to that point, I'd always rolled my own logic using custom Python classes and API calls. I hadn't used LangGraph or other frameworks built around agents or workflows. After reading Anthropic's post on "[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)"[^anthropic-building-effective-agents], I was curious: could I replicate and extend those patterns using Google ADK?

This post walks through how I did that, by implementing recursive, planner-based agents that can dynamically spawn parallel or sequential subtasks recursively using ADK.

## Anthropic's Agent Workflows

Anthropic outlined a few useful patterns in their post:
- **Prompt chaining**: Break a task into sequential steps.
- **Routing**: Classify a task and send it to one of several specialized subagents.
- **Parallelization**: Split a task into subtasks that can run simultaneously.
- **Orchestrator-workers**: A central orchestrator creates subtasks and synthesizes the results.
- **Evaluator-optimizer**: One agent produces, another evaluates, and they iterate.

I wanted to try building something like the *orchestrator-workers* setup, but to take it further by supporting both parallel and sequential subtasks, and doing this recursively. In other words, an agent that learns when to delegate, and how, on the fly.

## A Dynamic Agent

I created a custom `DynamicAgent`. It receives a task, uses a planner to decide whether to solve it directly or break it down, and then recursively builds out its subagents as needed.

At each node:
- If the task is simple, it uses a single `LlmAgent`.
- If the task can be parallelized, it spawns a `ParallelAgent` with multiple `DynamicAgent` subagents.
- If the task is sequential, it spawns a `SequentialAgent`, again composed of `DynamicAgent` subagents.

Here's what the core structure looks like (simplified):

```python
MAX_DEPTH = 3

class DynamicAgent:

	def __init__(self, task: str, depth: int = 0):
		self.task = task
		self.depth = depth
	
	def run(self):
		planner_agent = LlmAgent(task)
		
		# Run planner_agent
		plan = planner_agent.run()

		# Initalize sub_agents from plan
		sub_agents = [DynamicAgent(sub_task, self.depth + 1)
					  for sub_task in plan['sub_tasks']]

		# Create agent from plan
		if plan['type'] == "Llm" or self.depth >= MAX_DEPTH:
			agent = LlmAgent(task)
		elif plan['type'] == "Parallel":
			agent = ParallelAgent(sub_agents)
		elif plan['type'] == "Sequential":
			agent = SequentialAgent(sub_agents)
			
		# Run agent
		output = agent.run()

		return output
```

Each subagent is also a `DynamicAgent`. This means any subtask can be further broken down, until we reach `MAX_DEPTH` or a task suitable for a single LLM call.

## Using Google ADK

Google ADK requires a few changes from the pseudocode above:

### Initial Task

The initial task is retrieved from the user message, while subsequent tasks are passed in as arguments.

```python
self.task = self.task or ctx.user_content.parts[-1].text
```

### Planner Agent

We use a custom instruction template to generate a plan from the task, using a structured output schema `Plan` via Pydantic.

```python
from typing import Literal
from pydantic import BaseModel, Field


class Plan(BaseModel):
	type: Literal["Llm", "Parallel", "Sequential"] = Field(description="The type of the agent, can be 'Llm', 'Parallel' or 'Sequential'.")
	sub_tasks: list[str] = Field(description="The list of sub tasks.")
```

The planner chooses the best workflow for the task.
- **LLM agent**: straight-forward task requiring a single LLM call using the curent task and all ancestor tasks as context.
- **Parallel agent**: dynamically breaks down a task into parallel subtasks and synthesizes the results.
- **Sequential agent**: dynamically breaks down a task into sequential subtasks and synthesizes the results.

We set the instruction that includes constraints and examples.

{% raw %}
```python
PLANNER_INSTRUCTION = """You are the Planner Agent.

You determine if this task can be completed as a single task or whether it should be broken down into parallel or sequential subtasks.

Return a JSON object with the following fields:
- type: "Llm" | "Parallel" | "Sequential"
- sub_tasks: list[str]

<Constraints>
- Do not ask the user any follow-up questions.
- The maximum number of subtasks is {MAX_SUBTASKS}.
</Constraints>

<Ancestor Tasks>
{ANCESTOR_TASKS}
</Ancestor Tasks>

<Task>
{TASK}
</Task>

<JSON Example>
{{{{
    "type": "Llm",
    "sub_tasks": []
}}}}
</JSON Example>

<JSON Example>
{{
    "type": "Parallel",
    "sub_tasks": [
        "Researches renewable energy sources.",
        "Researches electric vehicle technology.",
        "Researches carbon capture methods.",
    ]
}}
</JSON Example>

<JSON Example>
{{
    "type": "Sequential",
    "sub_tasks": [
        "Writes initial Python code based on a specification.",
        "Reviews code and provides feedback.",
        "Refactors code based on review comments.",
    ]
}}
</JSON Example>
"""
```
{% endraw %}

We build the planner as an `LlmAgent`.

```python
planner_agent = LlmAgent(
    name=f"{self.name}_PlannerAgent",
    model=MODEL,
    description="Determines if this task should be broken down into subtasks.",
    instruction=PLANNER_INSTRUCTION.format(
        TASK=self.task,
        ANCESTOR_TASKS="\n".join(self.ancestor_tasks),
        MAX_SUBTASKS=MAX_SUBTASKS,
    ),
    output_schema=Plan,
    output_key=f"{self.name}_plan",
    include_contents="none",
)
```

In Google ADK, data is passed around using the state by specifying the `output_key` of an agent and read by referencing this state key in another agent's instruction. This is a departure from the traditional approach of passing data as arguments and in the [Observations on Google ADK](#observations-on-google-adk) section, we discuss the nuances of how to handle this.

### Dynamic Agent Decision Logic

Parallel and sequential agents have subagents, which are themselves instances of `DynamicAgent`, enabling recursion.

```python
sub_agents = [
	DynamicAgent(
		name=f"{self.name}_DynamicAgent_{i}",
		task=sub_task,
		ancestor_tasks=self.ancestor_tasks + [self.task],
		depth=self.depth + 1,
	)
	for i, sub_task in enumerate(plan["sub_tasks"])
]
```

With the `sub_agents` in hand, we can route to the correct agent.

```python
if plan["type"] == "Llm" or not plan["sub_tasks"] or self.depth >= MAX_DEPTH:
	agent = build_llm_agent(self.name, self.task, self.ancestor_tasks)
elif plan["type"] == "Parallel":
	agent = build_parallel_agent(self.name, self.task, sub_agents)
elif plan["type"] == "Sequential":
	agent = build_sequential_agent(self.name, self.task, sub_agents)
else:
	raise ValueError(f"Unsupported agent type: {plan['type']}")
```

Notice that if no subtasks are generated by the planner or if the depth is already at `MAX_DEPTH`, we revert to calling a single `LlmAgent`.

#### LLM Agent

The `LlmAgent` handles single-shot subtasks.

<figure>
    ```mermaid
    flowchart LR
        In([In]):::io --> Task[LLM Call]:::block
        Task --> Out([Out]):::io

        classDef io fill:#faf0ed,stroke:#f4c7bf,color:#b9441c
        classDef block fill:#eff5ea,stroke:#c2dcab,color:#4a8a1d
        linkStyle default stroke:#9c9c9a,stroke-dasharray: 5 5;
    ```
    <figcaption>LLM agent workflow</figcaption>
</figure>

```python
def build_llm_agent(name: str, task: str, ancestor_tasks: list[str]):
    return LlmAgent(
        name=f"{name}_LlmAgent",
        model=MODEL,
        description="Runs a single task.",
        instruction=LLM_INSTRUCTION.format(
            TASK=task, ANCESTOR_TASKS="\n".join(ancestor_tasks)
        ),
        output_key=f"{name}_result",
        include_contents="none",
    )
```

#### Parallel Agent

Parallel subtasks are each run by a `DynamicAgent`, and a `SynthesizerAgent` combines the results:

<figure>
    ```mermaid
    flowchart LR
        In([In]):::io --> Orchestrator:::block
        Orchestrator --> Task1[LLM Call 1]:::block
        Orchestrator --> Task2[LLM Call 2]:::block
        Orchestrator --> Task3[LLM Call 3]:::block
        Task1 --> Synthesizer:::block
        Task2 --> Synthesizer
        Task3 --> Synthesizer
        Synthesizer --> Out([Out]):::io

        classDef io fill:#faf0ed,stroke:#f4c7bf,color:#b9441c
        classDef block fill:#eff5ea,stroke:#c2dcab,color:#4a8a1d
        linkStyle default stroke:#9c9c9a,stroke-dasharray: 5 5;
    ```
    <figcaption>Parallel agent workflow</figcaption>
</figure>

```python
def build_parallel_agent(name: str, task: str, sub_agents: list[BaseAgent]):
    parallel_agent = ParallelAgent(
        name=f"{name}_ParallelAgent",
        description="Runs multiple subtasks in parallel.",
        sub_agents=sub_agents,
    )

    subtasks = "\n\n".join(
        [
            f"<Subtask>\n{sub_agent.task}\n</Subtask>\n\n<Subtask Result>\nStored in state key '{sub_agent.name}_result'\n</Subtask Result>"
            for sub_agent in sub_agents
        ]
    )

    synthesizer_agent = LlmAgent(
        name=f"{name}_SynthesizerAgent",
        model=MODEL,
        description="Synthesizes sub task results into a single result.",
        instruction=PARALLEL_SYNTHESIZER_INSTRUCTION.format(
            TASK=task, SUBTASKS=subtasks
        ),
        output_key=f"{name}_result",
        include_contents="none",
    )

    return SequentialAgent(
        name=f"{name}_WorkflowAgent",
        sub_agents=[parallel_agent, synthesizer_agent],
    )
```


#### Sequential Agent

Sequential subtasks run one after another, and we use a final identity agent to copy the last result into the output key.

<figure>
    ```mermaid
    flowchart LR
        In([In]):::io --> Orchestrator:::block
        Orchestrator --> Task1[LLM Call 1]:::block
        Orchestrator --> Task2[LLM Call 2]:::block
        Orchestrator --> Task3[LLM Call 3]:::block
        Task1 --> Task2
        Task2 --> Task3
        Task3 --> Identity:::block
        Identity --> Out([Out]):::io

        classDef io fill:#faf0ed,stroke:#f4c7bf,color:#b9441c
        classDef block fill:#eff5ea,stroke:#c2dcab,color:#4a8a1d
        linkStyle default stroke:#9c9c9a,stroke-dasharray: 5 5;
    ```
     <figcaption>Sequential agent workflow</figcaption>
</figure>

```python
def build_sequential_agent(name: str, task: str, sub_agents: list[BaseAgent]):
    # FIXME: Inject previous sub agent results into current sub agent task
    for i, sub_agent in enumerate(sub_agents[1:], 1):
        sub_agent.task += f" Result from previous step is stored in state key '{name}_DynamicAgent_{i-1}_result'."

    sequential_agent = SequentialAgent(
        name=f"{name}_SequentialAgent",
        description="Runs multiple subtasks sequentially.",
        sub_agents=sub_agents,
    )

    input_key = f"{name}_DynamicAgent_{len(sub_agents) - 1}_result"
    identity_agent = LlmAgent(
        name=f"{name}_IdentityAgent",
        model=MODEL,
        description="Copies the exact contents of the input.",
        instruction=SEQUENTIAL_IDENTITY_INSTRUCTION.format(INPUT_KEY=input_key),
        output_key=f"{name}_result",
        include_contents="none",
    )

    return SequentialAgent(
        name=f"{name}_WorkflowAgent",
        sub_agents=[sequential_agent, identity_agent],
    )
```

### Planner Example

Here's a recursive breakdown for an example task:

<figure>
    ```mermaid
    flowchart TD
        A["Plan a weekend trip to Tokyo.<br>(SequentialAgent)"]:::sequential
        A --> A1["Research popular attractions and activities in Tokyo.<br>(LlmAgent)"]:::llm
        A --> A2["Create a day-by-day itinerary including specific locations and estimated times.<br>(LlmAgent)"]:::llm
        A --> A3["Identify transportation options and restaurant recommendations for each day.<br>(ParallelAgent)"]:::parallel
        A3 --> A3a["Identify transportation options for each day of the Tokyo itinerary.<br>(LlmAgent)"]:::llm
        A3 --> A3b["Identify restaurant recommendations for each day of the Tokyo itinerary.<br>(LlmAgent)"]:::llm

        classDef llm fill:#eff5ea,stroke:#c2dcab,color:#4a8a1d
        classDef parallel fill:#f3f2f9,stroke:#ada6ce,color:#6560a3
        classDef sequential fill:#faf0ed,stroke:#f4c7bf,color:#b9441c
        linkStyle default stroke:#9c9c9a,stroke-dasharray: 5 5;
    ```
    <figcaption>Plan a weekend trip to Tokyo.</figcaption>
</figure>

Each node is a `DynamicAgent`. The leaves are `LlmAgents`.

## Observations on Google ADK

My main goal with this project was to push Google's Agent Development Kit (ADK) beyond its typical use cases, specifically by implementing recursive agent orchestration strategies like those described in Anthropic's *Building Effective Agents*. Along the way, I uncovered both some powerful primitives and a fair number of rough edges. Here's what stood out.

### A Work in Progress

The ADK feels promising but unfinished. The codebase has numerous visible TODOs, for example, [this one in `state.py`](https://github.com/google/adk-python/blob/917a8a19f794ba33fef08898937a73f0ceb809a2/src/google/adk/sessions/state.py#L42), that suggest internal components are still under construction. Documentation is similarly uneven. In one case, `BasePlanner` is referenced in the [Planning & Code Execution](https://google.github.io/adk-docs/agents/llm-agents/#planning-code-execution) guide, but clicking through to the [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/) page for more details yields no mention of it.

That said, the pace of development is rapid. At the time of writing, there were eight releases in the past month. Under the surface, ADK includes mechanisms like [`built_in_planner`](https://github.com/google/adk-python/blob/917a8a19f794ba33fef08898937a73f0ceb809a2/src/google/adk/planners/built_in_planner.py) that aren't clearly surfaced in the docs but point to more advanced planning capabilities present in the Gemini API. I'd love to see how much of this is dogfooded internally within Google's own AI products, and how the tight integration with Gemini plays out.

### Limited Support for File-Based Workflows

The evalset tools don't currently support uploading file artifacts. Instead, I had to manually build up `user_content` object, base64-encode the file contents, and embed the text directly into JSON payloads. This makes evaluating agents with file-based workflows more awkward than they should be, especially when multiple test cases utilize the same file.

### Opaque Content Controls

ADK provides control over what an agent can see via the `include_contents` field. You can set it to `'none'`, where an agent works only off its own instructions and current user inputs, or `'default'`, which includes "relevant" history from the session. However, it's unclear how relevance is determined or whether you can scope it (e.g., restrict an agent to reading only from its parent or sibling agents). For deeper workflows involving recursive or sequential planning, more transparency and control over context propagation would be hugely helpful.

### Ambiguity Around `output_key` Referencing

{% raw %}
The `output_key` mechanism is central to passing data between agents in ADK, but it's not always clear how it should be referenced in agent instructions. With no clear documentation, I was manually injecting the state keys into the agent instructions. Later, I found that on the [Loop Agents](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/#full-example-iterative-document-improvement) page, templating with `{{key}}` is used, e.g. `"Topic: {{initial_topic}}"`. But in some other places, like on the [Multi-agent Systems](https://google.github.io/adk-docs/agents/multi-agents/#sequential-pipeline-pattern) page, the key can be referenced through natural language, e.g. `"Report the result from state key 'result'"`. This inconsistency made it difficult to know how to use the `output_key`, which is central to the design of the ADK.
{% endraw %}

### The Single-Parent Rule Feels Restrictive

One architectural constraint that created friction was ADK's [single-parent rule](https://google.github.io/adk-docs/agents/multi-agents/#11-agent-hierarchy-parent-agent-sub-agents): a given agent instance can only belong to one parent. This makes sense for clean tree-like execution, but in practice it becomes limiting, especially in recursive workflows where you want a `DynamicAgent` to own the logic and identity of its children, while also handing those same children off to a `ParallelAgent` or `SequentialAgent` for execution. In my case, I had to restructure parts of the design to work around this constraint.

## Full Code

```python
from typing import AsyncGenerator, Literal, Optional

from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import BaseModel, Field
from typing_extensions import override


MODEL = "gemini-2.0-flash"
MAX_DEPTH = 3
MAX_SUBTASKS = 3

PLANNER_INSTRUCTION = """You are the Planner Agent.

You determine if this task can be completed as a single task or whether it should be broken down into parallel or sequential subtasks.

Return a JSON object with the following fields:
- type: "Llm" | "Parallel" | "Sequential"
- sub_tasks: list[str]

<Constraints>
- Do not ask the user any follow-up questions.
- The maximum number of subtasks is {MAX_SUBTASKS}.
</Constraints>

<Ancestor Tasks>
{ANCESTOR_TASKS}
</Ancestor Tasks>

<Task>
{TASK}
</Task>

<JSON Example>
{{
    "type": "Llm",
    "sub_tasks": []
}}
</JSON Example>

<JSON Example>
{{
    "type": "Parallel",
    "sub_tasks": [
        "Researches renewable energy sources.",
        "Researches electric vehicle technology.",
        "Researches carbon capture methods.",
    ]
}}
</JSON Example>

<JSON Example>
{{
    "type": "Sequential",
    "sub_tasks": [
        "Writes initial Python code based on a specification.",
        "Reviews code and provides feedback.",
        "Refactors code based on review comments.",
    ]
}}
</JSON Example>
"""

LLM_INSTRUCTION = """You are an AI Assistant responding to a single task.

<Constraints>
- Do not ask the user any follow-up questions.
</Constraints>

<Ancestor Tasks>
{ANCESTOR_TASKS}
</Ancestor Tasks>

<Task>
{TASK}
</Task>
"""

PARALLEL_SYNTHESIZER_INSTRUCTION = """You are an AI Assistant responsible for synthesizing sub task results into a single result.

<Constraints>
- Do not ask the user any follow-up questions.
</Constraints>

<Task>
{TASK}
</Task>

{SUBTASKS}
"""

SEQUENTIAL_IDENTITY_INSTRUCTION = (
    "Copy the exact content of state key '{INPUT_KEY}' into your output. Do not modify."
)


class Plan(BaseModel):
    type: Literal["Llm", "Parallel", "Sequential"] = Field(
        description="The type of the agent, can be 'Llm', 'Parallel' or 'Sequential'."
    )
    sub_tasks: list[str] = Field(description="The list of sub tasks.")


def build_llm_agent(name: str, task: str, ancestor_tasks: list[str]):
    return LlmAgent(
        name=f"{name}_LlmAgent",
        model=MODEL,
        description="Runs a single task.",
        instruction=LLM_INSTRUCTION.format(
            TASK=task, ANCESTOR_TASKS="\n".join(ancestor_tasks)
        ),
        output_key=f"{name}_result",
        include_contents="none",
    )


def build_parallel_agent(name: str, task: str, sub_agents: list[BaseAgent]):
    parallel_agent = ParallelAgent(
        name=f"{name}_ParallelAgent",
        description="Runs multiple subtasks in parallel.",
        sub_agents=sub_agents,
    )

    subtasks = "\n\n".join(
        [
            f"<Subtask>\n{sub_agent.task}\n</Subtask>\n\n<Subtask Result>\nStored in state key '{sub_agent.name}_result'\n</Subtask Result>"
            for sub_agent in sub_agents
        ]
    )

    synthesizer_agent = LlmAgent(
        name=f"{name}_SynthesizerAgent",
        model=MODEL,
        description="Synthesizes sub task results into a single result.",
        instruction=PARALLEL_SYNTHESIZER_INSTRUCTION.format(
            TASK=task, SUBTASKS=subtasks
        ),
        output_key=f"{name}_result",
        include_contents="none",
    )

    return SequentialAgent(
        name=f"{name}_WorkflowAgent",
        sub_agents=[parallel_agent, synthesizer_agent],
    )


def build_sequential_agent(name: str, task: str, sub_agents: list[BaseAgent]):
    # FIXME: Inject previous sub agent results into current sub agent task
    for i, sub_agent in enumerate(sub_agents[1:], 1):
        sub_agent.task += f" Result from previous step is stored in state key '{name}_DynamicAgent_{i-1}_result'."

    sequential_agent = SequentialAgent(
        name=f"{name}_SequentialAgent",
        description="Runs multiple subtasks sequentially.",
        sub_agents=sub_agents,
    )

    input_key = f"{name}_DynamicAgent_{len(sub_agents) - 1}_result"
    identity_agent = LlmAgent(
        name=f"{name}_IdentityAgent",
        model=MODEL,
        description="Copies the exact contents of the input.",
        instruction=SEQUENTIAL_IDENTITY_INSTRUCTION.format(INPUT_KEY=input_key),
        output_key=f"{name}_result",
        include_contents="none",
    )

    return SequentialAgent(
        name=f"{name}_WorkflowAgent",
        sub_agents=[sequential_agent, identity_agent],
    )


class DynamicAgent(BaseAgent):
    task: Optional[str] = None
    ancestor_tasks: list[str] = Field(default_factory=list)
    depth: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str = "DynamicAgent_0",
        task: Optional[str] = None,
        ancestor_tasks: list[str] = [],
        depth: int = 0,
    ):
        super().__init__(
            name=name,
            task=task,
            ancestor_tasks=ancestor_tasks,
            depth=depth,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Get initial task from user message
        self.task = self.task or ctx.user_content.parts[-1].text

        planner_agent = LlmAgent(
            name=f"{self.name}_PlannerAgent",
            model=MODEL,
            description="Determines if this task should be broken down into subtasks.",
            instruction=PLANNER_INSTRUCTION.format(
                TASK=self.task,
                ANCESTOR_TASKS="\n".join(self.ancestor_tasks),
                MAX_SUBTASKS=MAX_SUBTASKS,
            ),
            output_schema=Plan,
            output_key=f"{self.name}_plan",
            include_contents="none",
        )

        async for event in planner_agent.run_async(ctx):
            yield event

        plan: Plan = ctx.session.state[f"{self.name}_plan"]

        sub_agents = [
            DynamicAgent(
                name=f"{self.name}_DynamicAgent_{i}",
                task=sub_task,
                ancestor_tasks=self.ancestor_tasks + [self.task],
                depth=self.depth + 1,
            )
            for i, sub_task in enumerate(plan["sub_tasks"])
        ]

        if plan["type"] == "Llm" or not plan["sub_tasks"] or self.depth >= MAX_DEPTH:
            agent = build_llm_agent(self.name, self.task, self.ancestor_tasks)
        elif plan["type"] == "Parallel":
            agent = build_parallel_agent(self.name, self.task, sub_agents)
        elif plan["type"] == "Sequential":
            agent = build_sequential_agent(self.name, self.task, sub_agents)
        else:
            raise ValueError(f"Unsupported agent type: {plan['type']}")

        async for event in agent.run_async(ctx):
            yield event


root_agent = DynamicAgent(name="RootAgent")
```

## References

[^google-adk]: Google. (n.d.). [Agent Development Kit (ADK).](https://google.github.io/adk-docs/)
[^anthropic-building-effective-agents]: Anthropic. (19 December 2024).  [Building effective agents.](https://www.anthropic.com/engineering/building-effective-agents)