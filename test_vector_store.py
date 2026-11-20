
from app.rag.loader import load_document
from app.rag.chunker import chunk_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store


# Step 1: Load document
documents = load_document("data/documents/sample.txt")

# Step 2: Split into chunks
chunks = chunk_documents(
    documents,
    chunk_size=100,
    overlap=20,
)

# Step 3: Generate embeddings
embedding_model = get_embedding_model()

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode_texts(texts)

# Step 4: Store in ChromaDB
vector_store = get_vector_store()

vector_store.add_documents(
    chunks=chunks,
    embeddings=embeddings,
)

# Step 5: Check database
print("Stored chunks:", vector_store.get_collection_count())
