"""Embedding Module for RAG Pipeline.

Generates dense vector embeddings for text chunks and queries using
SentenceTransformers with the lightweight, high-performing 'all-MiniLM-L6-v2' model.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Reusable class for generating normalized sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize and load the SentenceTransformer embedding model.

        Args:
            model_name: HuggingFace model identifier. Defaults to 'all-MiniLM-L6-v2'
                        which produces 384-dimensional dense vectors.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        """Return vector dimension produced by this model (384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def embed_query(self, text: str) -> List[float]:
        """Generate a normalized embedding vector for a single query string.

        Args:
            text: Query string.

        Returns:
            A list of floats representing the 384-dimensional embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text query.")

        # normalize_embeddings=True ensures cosine similarity can be computed via dot product
        # convert_to_tensor=True bypasses numpy conversion errors on older pytorch
        embedding = self.model.encode(
            text.strip(),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate normalized embedding vectors for multiple text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            A list of embedding vectors (list of float lists).
        """
        if not texts:
            return []

        # Filter and clean whitespace while preserving list length
        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=len(cleaned_texts) > 20,
        )
        return [vec.tolist() for vec in embeddings]


# Global cached singleton instance for quick access across the application
_DEFAULT_EMBEDDER: Union[EmbeddingGenerator, None] = None


def get_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """Get or instantiate a reusable singleton instance of EmbeddingGenerator.

    Args:
        model_name: Name of the sentence transformer model.

    Returns:
        Shared EmbeddingGenerator instance.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model_name != model_name:
        _DEFAULT_EMBEDDER = EmbeddingGenerator(model_name=model_name)
    return _DEFAULT_EMBEDDER
