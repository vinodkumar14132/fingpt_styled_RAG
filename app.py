from fastapi import FastAPI

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
app = FastAPI(
    title="FinGPT RAG Assistant"
)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)
@app.get("/")
def home():

    return {
        "message": "FinGPT RAG Running"
    }
@app.get("/ask")
def ask(question: str):

    results = vector_db.similarity_search(
        question,
        k=3
    )

    response = []

    for doc in results:

        response.append(
            doc.page_content
        )

    return {
        "question": question,
        "retrieved_chunks": response
    }