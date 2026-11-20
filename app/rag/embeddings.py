
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingModel:
    def __init__(self):
        settings = get_settings()

        self.model = SentenceTransformer(
            settings.embedding_model
        )

    def encode_text(self, text: str) -> list[float]:
        """
        Convert a single text into an embedding vector.
        """

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def encode_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert multiple texts into embedding vectors.
        """

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return vectors.tolist()


_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model
