
from app.rag.retriever import get_retriever


retriever = get_retriever()

query = "What is retrieval augmented generation?"

results = retriever.search(query, top_k=3)

print(f"Number of results: {len(results)}")

for index, result in enumerate(results, start=1):
    print(f"\n========== Result {index} ==========")

    print("Text:")
    print(result["text"])

    print("\nMetadata:")
    print(result["metadata"])

    print("\nDistance:")
    print(result["distance"])
