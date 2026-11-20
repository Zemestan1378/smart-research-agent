
from app.rag.embeddings import get_embedding_model


embedding_model = get_embedding_model()

texts = [
    "Artificial intelligence is changing the world.",
    "Machine learning is a branch of artificial intelligence.",
    "I like eating pizza.",
]

vectors = embedding_model.encode_texts(texts)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))

for index, vector in enumerate(vectors):
    print(f"\nText {index + 1}")
    print("First 5 values:", vector[:5])
