from typing import TypedDict
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph
from langgraph.graph import START, END

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import yfinance as yf


# ----------------------------
# Load Environment
# ----------------------------

load_dotenv()


# ----------------------------
# Groq LLM
# ----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# ----------------------------
# Vector DB
# ----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)


# ----------------------------
# State
# ----------------------------

from typing import TypedDict, List
conversation_memory = []


class FinanceState(TypedDict):
    question: str
    route: str
    context: str
    answer: str
    review: str
    approval: str
    memory: List[str]

# ----------------------------
# Coordinator Agent
# ----------------------------

def router_node(state: FinanceState):

    print("\nRunning Coordinator Agent...\n")

    prompt = f"""
You are a routing agent.

Available routes:

rag
stock

Rules:

- Questions about AI strategy,
  revenue, earnings, transcripts,
  management comments -> rag

- Questions about stock prices,
  tickers, market data -> stock

Return ONLY:

rag

or

stock

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()

    print("Selected Route:", route)

    state["route"] = route

    return state


# ----------------------------
# RAG Agent
# ----------------------------

def rag_agent(state: FinanceState):

    print("\nRunning RAG Agent...\n")

    results = vector_db.similarity_search(
        state["question"],
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    state["context"] = context

    prompt = f"""
Answer only using the context.

Context:
{context}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    state["memory"].append(
    f"Q: {state['question']}"
)

    state["memory"].append(
    f"A: {state['answer']}"
)

    return state


# ----------------------------
# Stock Agent
# ----------------------------

def stock_agent(state: FinanceState):

    print("\nRunning Stock Agent...\n")

    words = state["question"].split()

    ticker = words[0].upper()

    stock = yf.Ticker(ticker)

    try:

        price = stock.info.get(
            "currentPrice",
            "Not Available"
        )

        state["answer"] = (
            f"{ticker} Current Price = {price}"
        )

    except Exception as e:

        state["answer"] = str(e)
    state["memory"].append(
    f"Q: {state['question']}"
)

    state["memory"].append(
    f"A: {state['answer']}"
)

    return state


# ----------------------------
# Reviewer Agent
# ----------------------------

def reviewer_agent(state):

    print("\nRunning Reviewer Agent...\n")

    prompt = f"""
Question:
{state['question']}

Answer:
{state['answer']}

Return ONLY:

GOOD

or

BAD
"""

    response = llm.invoke(prompt)

    state["review"] = (
        response.content.strip().upper()
    )

    return state
#improve agent

def improve_agent(state):

    print("\nRunning Improve Agent...\n")

    prompt = f"""
Question:
{state['question']}

Current Answer:
{state['answer']}

Review:
{state['review']}

Improve the answer.
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    state["retry_count"] += 1

    return state


def human_node(state):

    print("\nHuman Approval Required\n")

    print("\nGenerated Answer:\n")
    print(state["answer"])

    approval = input(
        "\nApprove? (yes/no): "
    )

    state["approval"] = approval.lower()

    return state


def memory_node(state):

    print("\nRunning Memory Node...\n")

    print("Memory Contents:")

    for item in state["memory"]:
        print(item)

    return state

# ----------------------------
# Route Function
# ----------------------------

def route_function(state: FinanceState):

    return state["route"]

def review_route(state):

    if state["review"] == "GOOD":
        return "end"

    if state["retry_count"] >= 2:
        return "end"

    return "improve"

def approval_route(state):

    if state["approval"] == "yes":
        return "approved"

    return "rejected"

def approved_node(state):

    print("\nApproved By Human\n")

    return state

def rejected_node(state):

    print("\nRejected By Human\n")

    state["answer"] = (
        "Request rejected by human."
    )

    return state
# ----------------------------
# Graph
# ----------------------------

graph = StateGraph(FinanceState)


# Nodes

graph.add_node(
    "router",
    router_node
)

graph.add_node(
    "rag_agent",
    rag_agent
)

graph.add_node(
    "stock_agent",
    stock_agent
)

graph.add_node(
    "reviewer",
    reviewer_agent
)
graph.add_node(
    "improve_agent",
    improve_agent
)
graph.add_node(
    "human",
    human_node
)

graph.add_node(
    "approved",
    approved_node
)

graph.add_node(
    "rejected",
    rejected_node
)

# Start


graph.add_node(
    "memory",
    memory_node
)

graph.add_edge(
    START,
    "memory"
)

graph.add_edge(
    "memory",
    "router"
)

# Conditional Routing

graph.add_conditional_edges(
    "router",
    route_function,
    {
        "rag": "rag_agent",
        "stock": "stock_agent"
    }
)


# Agents -> Reviewer

graph.add_edge(
    "rag_agent",
    "reviewer"
)

graph.add_conditional_edges(
    "reviewer",
    review_route,
    {
        "improve": "improve_agent",
        "end": END
    }
)
graph.add_edge(
    "improve_agent",
    "reviewer"
)

graph.add_edge(
    "stock_agent",
    "reviewer"
)
# Reviewer -> End

'''graph.add_edge(
    "reviewer",
    END
)'''
graph.add_edge(
    "reviewer",
    "human"
)
graph.add_conditional_edges(
    "human",
    approval_route,
    {
        "approved": "approved",
        "rejected": "rejected"
    }
)
graph.add_edge(
    "approved",
    END
)

graph.add_edge(
    "rejected",
    END
)
# Compile

app = graph.compile()


# ----------------------------
# Test
# ----------------------------

'''result = app.invoke(
{
    "question":"AAPL stock price",
    "route":"",
    "context":"",
    "answer":"",
    "review":""
}
)

print("\nFINAL RESULT\n")

print("Question:")
print(result["question"])

print("\nAnswer:")
print(result["answer"])

print("\nReview:")
print(result["review"])'''


result = app.invoke(
{
    "question":"What is TCS AI strategy?",
    "route":"",
    "context":"",
    "answer":"",
    "review":"",
    "approval":"",
    "memory":conversation_memory

}
)
conversation_memory = result["memory"]
print("\nFINAL RESULT\n")

print("Question:")
print(result["question"])

print("\nAnswer:")
print(result["answer"])

print("\nReview:")
print(result["review"])
