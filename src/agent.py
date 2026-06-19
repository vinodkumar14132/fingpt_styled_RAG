from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

def transcript_tool(question):

    results = vector_db.similarity_search(
        question,
        k=1
    )

    return results[0].page_content

def calculator_tool(expression):

    return eval(expression)
import yfinance as yf

def stock_tool(ticker):

    stock = yf.Ticker(ticker)

    return stock.info.get(
        "currentPrice",
        "Not Available"
    )

def finance_agent(question):

    question = question.lower()

    if "price" in question:

        return stock_tool("AAPL")

    elif "revenue" in question:

        return transcript_tool(question)

    elif any(
        char.isdigit()
        for char in question
    ):

        return calculator_tool(
            "100*12"
        )

    else:

        return (
            "No tool selected"
        )
    
    print(
    finance_agent(
        "What was TCS revenue?"
    )
)
print(finance_agent(
    "What is TCS revenue?"
))

print(finance_agent(
    "What is TCS AI strategy?"
))

print(finance_agent(
    "What did management say about AI?"
))