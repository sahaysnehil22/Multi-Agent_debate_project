"""Unit tests for RAG pipeline components: loader, chunker, embeddings, vector store, and retriever."""

import os
import pytest
from langchain_core.documents import Document

from src.rag.document_loader import DocumentLoader, load_pdf
from src.rag.chunker import DocumentChunker, chunk_documents
from src.rag.embeddings import EmbeddingGenerator, get_embedding_generator
from src.rag.vector_store import ChromaVectorStore
from src.rag.retriever import RAGRetriever


@pytest.fixture
def sample_pdf_path():
    """Return path to existing sample debate PDF."""
    path = os.path.abspath("data/documents/sample_debate_brief.pdf")
    assert os.path.exists(path), f"Sample PDF missing at {path}"
    return path


def test_document_loader_success(sample_pdf_path):
    """Test loading valid PDF document."""
    docs = load_pdf(sample_pdf_path)
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "source_file" in docs[0].metadata
    assert "page_number" in docs[0].metadata
    assert docs[0].metadata["page_number"] == 1
    assert len(docs[0].page_content) > 0


def test_document_loader_file_not_found():
    """Test error handling when PDF does not exist."""
    with pytest.raises(FileNotFoundError):
        load_pdf("data/documents/non_existent_file.pdf")


def test_document_loader_invalid_extension():
    """Test error handling for non-pdf file."""
    with pytest.raises(ValueError):
        load_pdf("requirements.txt")


def test_chunker_metadata_enrichment():
    """Test chunking and metadata enrichment."""
    sample_docs = [
        Document(
            page_content="A" * 1500,
            metadata={"source_file": "policy.pdf", "page_number": 1},
        )
    ]
    chunks = chunk_documents(sample_docs, chunk_size=800, chunk_overlap=150)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata["source_file"] == "policy.pdf"
        assert c.metadata["page_number"] == 1
        assert "chunk_id" in c.metadata
        assert c.metadata["chunk_index"] == i
        assert c.metadata["character_count"] == len(c.page_content)


def test_embedding_generator():
    """Test sentence transformer embedding dimensions and normalization."""
    embedder = get_embedding_generator("all-MiniLM-L6-v2")
    assert embedder.dimension == 384

    query = "Multi-agent consensus reasoning"
    vec = embedder.embed_query(query)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

    # Test batch embedding
    batch_vecs = embedder.embed_documents(["Argument Alpha", "Argument Beta"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384


def test_vector_store_and_retriever(tmp_path):
    """Test full vector store storage, indexing, and retriever query."""
    test_db_dir = str(tmp_path / "test_chroma")
    vector_store = ChromaVectorStore(
        persist_directory=test_db_dir,
        collection_name="test_collection",
    )

    docs = [
        Document(
            page_content="Quantum computing drastically reduces decryption time for legacy RSA.",
            metadata={"source_file": "quantum.pdf", "page_number": 1, "chunk_id": "q1"},
        ),
        Document(
            page_content="Solar and wind energy costs dropped by seventy percent in the last decade.",
            metadata={"source_file": "energy.pdf", "page_number": 2, "chunk_id": "e1"},
        ),
    ]

    ids = vector_store.add_documents(docs)
    assert len(ids) == 2
    assert vector_store.count() == 2

    # Query retriever
    retriever = RAGRetriever(vector_store=vector_store, default_top_k=1)
    results = retriever.retrieve("cryptography and RSA encryption keys", top_k=1)

    assert len(results) == 1
    assert "quantum.pdf" in results[0]["source_file"]
    assert "decryption" in results[0]["content"].lower()
    assert results[0]["similarity_score"] > 0.0
