from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = "data/reports/tcs_transcripts.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"Pages Loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")

print(chunks[0].page_content[:500])