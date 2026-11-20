
from pathlib import Path

from app.rag.loader import load_document
from app.rag.chunker import chunk_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store


class DocumentPipeline:

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()

    def ingest_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> dict:
        """
        Load, chunk, embed, and store a document.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        # Step 1: Load document
        documents = load_document(str(path))

        if not documents:
            return {
                "filename": path.name,
                "pages": 0,
                "chunks": 0,
                "status": "empty",
            }

        # Step 2: Split into chunks
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            return {
                "filename": path.name,
                "pages": len(documents),
                "chunks": 0,
                "status": "empty",
            }

        # Step 3: Generate embeddings
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode_texts(texts)

        # Step 4: Save in ChromaDB
        self.vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "filename": path.name,
            "pages": len(documents),
            "chunks": len(chunks),
            "status": "success",
        }


_pipeline = None


def get_document_pipeline() -> DocumentPipeline:
    global _pipeline

    if _pipeline is None:
        _pipeline = DocumentPipeline()

    return _pipeline
