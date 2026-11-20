
from app.rag.loader import load_document
from app.rag.chunker import chunk_documents


documents = load_document("data/documents/sample.txt")

chunks = chunk_documents(
    documents,
    chunk_size=100,
    overlap=20,
)

print(f"Number of chunks: {len(chunks)}")

for chunk in chunks:
    print("\n--------------------")
    print("Metadata:", chunk["metadata"])
    print("Text:", chunk["text"])
