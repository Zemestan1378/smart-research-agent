
import chromadb

from app.core.config import get_settings


class VectorStore:

    def __init__(self):
        settings = get_settings()

        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store chunks, embeddings, and metadata in ChromaDB.
        """

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be equal."
            )

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            metadata = chunk["metadata"]

            document_id = (
                f'{metadata["source"]}_'
                f'{metadata["page"]}_'
                f'{metadata["chunk_index"]}'
            )

            ids.append(document_id)
            documents.append(chunk["text"])
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def get_collection_count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()


_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store
