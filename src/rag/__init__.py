"""RAG (Retrieval-Augmented Generation) & Grounding Subsystem.

Provides end-to-end PDF loading, recursive character chunking, dense vector embeddings,
persistent ChromaDB storage, semantic retrieval, and factual grounding for multi-agent debates.
"""

from src.rag.document_loader import DocumentLoader, load_pdf
from src.rag.chunker import DocumentChunker, chunk_documents
from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator
from src.rag.vector_store import ChromaVectorStore
from src.rag.retriever import RAGRetriever, retrieve_relevant_chunks
from src.rag.grounding import (
    GroundingManager,
    format_evidence_context,
    create_grounded_prompt,
)

__all__ = [
    # Document loading
    "DocumentLoader",
    "load_pdf",
    # Chunking
    "DocumentChunker",
    "chunk_documents",
    # Embeddings
    "EmbeddingGenerator",
    "get_embedding_generator",
    # Vector store
    "ChromaVectorStore",
    # Retriever
    "RAGRetriever",
    "retrieve_relevant_chunks",
    # Grounding
    "GroundingManager",
    "format_evidence_context",
    "create_grounded_prompt",
]
