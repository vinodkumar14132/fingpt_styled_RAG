from typing import TypedDict

class FinanceState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_node(state: FinanceState):

    print("Running Retrieve Node")

    state["context"] = (
        "TCS AI revenue crossed $2.3 billion"
    )

    return state


def answer_node(state: FinanceState):

    print("Running Answer Node")

    state["answer"] = (
        f"Answer based on: "
        f"{state['context']}"
    )

    return state

from langgraph.graph import StateGraph
from langgraph.graph import START, END

graph = StateGraph(FinanceState)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "answer",
    answer_node
)

graph.add_edge(
    START,
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "answer"
)

graph.add_edge(
    "answer",
    END
)

app = graph.compile()
result = app.invoke(
    {
        "question": "What is TCS AI strategy?",
        "context": "",
        "answer": ""
    }
)

print(result)