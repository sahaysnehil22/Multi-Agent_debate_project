"""Retriever Module for RAG Pipeline.

Accepts queries, converts them to normalized vector embeddings, and performs
nearest-neighbor similarity search against the persistent ChromaDB collection.
"""

from typing import Any, Dict, List, Optional

from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator
from src.rag.vector_store import ChromaVectorStore


class RAGRetriever:
    """Retrieves the top-k most relevant evidence chunks from ChromaDB for a given query."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        default_top_k: int = 5,
    ):
        """Initialize retriever with vector store and embedding generator.

        Args:
            vector_store: ChromaVectorStore instance. Defaults to standard persistent store.
            embedder: EmbeddingGenerator instance. Defaults to all-MiniLM-L6-v2 embedder.
            default_top_k: Default number of relevant chunks to retrieve (default: 5).
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or get_embedding_generator()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant document chunks for a text query.

        Args:
            query: The user premise, question, or agent argument text.
            top_k: Number of chunks to retrieve (defaults to 5).
            where_filter: Optional ChromaDB metadata filter (e.g. {"source_file": "doc.pdf"}).

        Returns:
            A list of retrieved result dictionaries, ordered by relevance:
            [
                {
                    "chunk_id": str,
                    "content": str,
                    "metadata": dict,
                    "distance": float,
                    "similarity_score": float,
                },
                ...
            ]
        """
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None and top_k > 0 else self.default_top_k

        # Total available items in collection
        total_items = self.vector_store.count()
        if total_items == 0:
            return []

        # ChromaDB requires n_results <= total_items
        query_k = min(k, total_items)

        # 1. Convert user query to a normalized embedding vector
        query_embedding = self.embedder.embed_query(query)

        # 2. Search ChromaDB using the query vector
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": query_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.vector_store.collection.query(**query_kwargs)

        # 3. Format raw ChromaDB results into structured, developer-friendly dictionaries
        retrieved_chunks: List[Dict[str, Any]] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            chunk_id = ids_list[i]
            content = docs_list[i] if i < len(docs_list) else ""
            metadata = metas_list[i] if i < len(metas_list) and metas_list[i] else {}
            distance = float(dists_list[i]) if i < len(dists_list) else 0.0

            # Cosine distance range is typically [0, 2]; for normalized vectors cosine sim = 1 - (dist / 2)
            # Higher similarity score means more relevant
            similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "distance": round(distance, 4),
                    "similarity_score": round(similarity_score, 4),
                    "source_file": metadata.get("source_file", "unknown"),
                    "page_number": metadata.get("page_number", 1),
                }
            )

        return retrieved_chunks


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    vector_store: Optional[ChromaVectorStore] = None,
) -> List[Dict[str, Any]]:
    """Convenience helper function to retrieve chunks for a query in one line.

    Args:
        query: Query string.
        top_k: Number of chunks to retrieve.
        vector_store: Optional ChromaVectorStore.

    Returns:
        List of retrieved chunk dictionaries.
    """
    retriever = RAGRetriever(vector_store=vector_store, default_top_k=top_k)
    return retriever.retrieve(query=query, top_k=top_k)
