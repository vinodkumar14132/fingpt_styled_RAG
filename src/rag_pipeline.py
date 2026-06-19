from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
loader = PyPDFLoader(
    "data/reports/tcs_transcripts.pdf"
)

documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vector_db"
)
print(
    f"Chunks Stored: {len(chunks)}"
)