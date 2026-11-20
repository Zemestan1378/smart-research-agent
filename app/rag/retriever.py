
from app.core.config import get_settings
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store


class Retriever:

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.top_k = get_settings().top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = self.top_k

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_model.encode_text(query)

        collection = self.vector_store.collection

        total_chunks = collection.count()

        if total_chunks == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append({
                "text": document,
                "metadata": metadata,
                "distance": distance,
            })

        return retrieved_chunks


_retriever = None


def get_retriever() -> Retriever:
    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever
