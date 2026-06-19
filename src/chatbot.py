from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

query = "What is TCS's AI strategy?"

results = vector_db.similarity_search(
    query,
    k=3
)

context = "\n\n".join(
    [doc.page_content for doc in results]
)

prompt = f"""
You are a financial analyst.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

print(prompt)