from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

load_dotenv()

# -------------------
# LLM
# -------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------
# Tools
# -------------------

@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations"""
    return str(eval(expression))

@tool
def transcript_search(query: str):
    """
    Searches TCS earnings call transcripts.
    Use for revenue, margins, AI strategy,
    management commentary, guidance,
    growth and business questions.
    """

    results = vector_db.similarity_search(
        query,
        k=3
    )

    return "\n\n".join(
        [doc.page_content for doc in results]
    )




tools = [
    calculator,
    transcript_search
]

# -------------------
# Agent
# -------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a finance assistant."
)

# -------------------
# Test
# -------------------

response = agent.invoke(
{
    "messages": [
        {
            "role": "user",
            "content":
            "Use transcript_search tool. What is TCS AI strategy?"
        }
    ]
}
)

from pprint import pprint

pprint(response)
