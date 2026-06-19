from typing import TypedDict

from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph
from langgraph.graph import START, END

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------------
# Load Environment
# ---------------------------------

load_dotenv()


# ---------------------------------
# LLM
# ---------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------------
# Vector DB
# ---------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)


# ---------------------------------
# State
# ---------------------------------

from typing import TypedDict

class FinanceState(TypedDict):
    question: str
    route: str
    context: str
    answer: str

def router_node(state):

    print("\nRunning Router Node...\n")

    question = state["question"].lower()

    if "price" in question:
        state["route"] = "stock"

    elif "*" in question:
        state["route"] = "calculator"

    else:
        state["route"] = "rag"

    return state

import yfinance as yf

def stock_node(state):

    print("\nRunning Stock Node...\n")

    question = state["question"]

    ticker = question.split()[0].upper()

    stock = yf.Ticker(ticker)

    price = stock.info.get(
        "currentPrice",
        "Not Available"
    )

    state["answer"] = (
        f"{ticker} Current Price = {price}"
    )

    return state
def calculator_node(state):

    print("\nRunning Calculator Node...\n")

    expression = state["question"]

    state["answer"] = str(eval(expression))

    return state
def retrieve_node(state: FinanceState):

    print("\nRunning Retrieve Node...\n")

    results = vector_db.similarity_search(
        state["question"],
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    state["context"] = context

    return state
def answer_node(state: FinanceState):

    print("\nRunning Answer Node...\n")

    prompt = f"""
You are a financial analyst.

Answer ONLY using the context below.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state

def route_function(state):

    return state["route"]
graph = StateGraph(FinanceState)

graph.add_node(
    "router",
    router_node
)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "answer",
    answer_node
)

graph.add_node(
    "stock",
    stock_node
)

graph.add_node(
    "calculator",
    calculator_node
)
graph.add_edge(
    START,
    "router"
)
graph.add_conditional_edges(
    "router",
    route_function,
    {
        "rag": "retrieve",
        "stock": "stock",
        "calculator": "calculator"
    }
)
graph.add_edge(
    "retrieve",
    "answer"
)

graph.add_edge(
    "answer",
    END
)
graph.add_edge(
    "stock",
    END
)

graph.add_edge(
    "calculator",
    END
)
app = graph.compile()

result = app.invoke(
{
    "question":"AAPL stock price",
    "route":"",
    "context":"",
    "answer":""
}
)

print(result["answer"])

