"""Vector Store Module for RAG Pipeline.

Manages persistent on-disk vector storage using ChromaDB (v1.5.9) PersistentClient.
Stores document chunks, metadata, and normalized dense embeddings under `data/vector_db`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_core.documents import Document

from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator


class ChromaVectorStore:
    """Persistent ChromaDB vector store for debate documents and evidence."""

    DEFAULT_COLLECTION_NAME = "debate_knowledge_base"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: Optional[EmbeddingGenerator] = None,
    ):
        """Initialize ChromaDB PersistentClient and load or create the collection.

        Args:
            persist_directory: Absolute or relative directory path for ChromaDB storage.
                               Defaults to `<project_root>/data/vector_db`.
            collection_name: Name of the vector collection.
            embedder: EmbeddingGenerator instance to generate vectors for chunks.
        """
        if persist_directory is None:
            # Default to project_root / data / vector_db
            project_root = Path(__file__).resolve().parent.parent.parent
            persist_directory = str(project_root / "data" / "vector_db")

        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.collection_name = collection_name
        self.embedder = embedder or get_embedding_generator()

        # Initialize ChromaDB persistent storage client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Add or upsert chunked Document objects into the collection.

        Generates embeddings if not pre-computed. Uses upsert to prevent duplicate errors.

        Args:
            documents: List of chunked Document objects.
            embeddings: Optional precomputed list of embedding vectors.

        Returns:
            List of IDs for the documents stored.
        """
        if not documents:
            return []

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for index, doc in enumerate(documents):
            # Prefer existing chunk_id from chunker, otherwise generate deterministic ID
            doc_id = doc.metadata.get(
                "chunk_id",
                f"{doc.metadata.get('source_file', 'doc')}#chunk_{index}",
            )
            ids.append(str(doc_id))
            texts.append(doc.page_content)

            # ChromaDB metadata values must be int, float, str, or bool
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self.embedder.embed_documents(texts)

        # Upsert documents and embeddings into ChromaDB (overwrites on duplicate ID)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return ids

    def count(self) -> int:
        """Return the number of document chunks stored in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """Reset and empty the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        # Re-create fresh collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def peek(self, limit: int = 5) -> Dict[str, Any]:
        """Peek at the first few items in the vector collection for debugging."""
        return self.collection.peek(limit=limit)
