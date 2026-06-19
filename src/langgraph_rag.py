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

class FinanceState(TypedDict):
    question: str
    context: str
    answer: str


# ---------------------------------
# Node 1 : Retrieve
# ---------------------------------

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


# ---------------------------------
# Node 2 : Answer
# ---------------------------------

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


# ---------------------------------
# Graph
# ---------------------------------

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


# ---------------------------------
# Run
# ---------------------------------

result = app.invoke(
    {
        "question": "What is TCS AI strategy?",
        "context": "",
        "answer": ""
    }
)

print("\nFINAL STATE\n")

print(result)