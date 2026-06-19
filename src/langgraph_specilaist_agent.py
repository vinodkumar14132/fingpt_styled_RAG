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

class FinanceState(TypedDict):
    question: str
    route: str
    context: str
    answer: str
    review: str


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

    return state


# ----------------------------
# Reviewer Agent
# ----------------------------

def reviewer_agent(state: FinanceState):

    print("\nRunning Reviewer Agent...\n")

    prompt = f"""
Question:
{state['question']}

Answer:
{state['answer']}

Review whether the answer
addresses the question.

Return ONLY:

GOOD

or

BAD
"""

    response = llm.invoke(prompt)

    state["review"] = (
        response.content
        .strip()
        .upper()
    )

    return state


# ----------------------------
# Route Function
# ----------------------------

def route_function(state: FinanceState):

    return state["route"]


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


# Start

graph.add_edge(
    START,
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

graph.add_edge(
    "stock_agent",
    "reviewer"
)


# Reviewer -> End

graph.add_edge(
    "reviewer",
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
print(result["review"])