"""Complete End-to-End RAG Pipeline Demonstration Script.

Demonstrates the 6-stage RAG flow for the Multi-Agent Debate Engine:
1. Load PDF document (document_loader.py)
2. Chunk document into semantic segments (chunker.py)
3. Generate normalized dense embeddings (embeddings.py)
4. Store chunks & embeddings persistently in ChromaDB (vector_store.py)
5. Retrieve top-k relevant evidence for a debate query (retriever.py)
6. Synthesize grounded context & strict debate agent prompt (grounding.py)
"""

import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (
    load_pdf,
    chunk_documents,
    get_embedding_generator,
    ChromaVectorStore,
    RAGRetriever,
    GroundingManager,
)


def run_rag_demo():
    print("\n" + "=" * 80)
    print("      MULTI-AGENT DEBATE ENGINE - RAG & GROUNDING PIPELINE DEMO")
    print("=" * 80 + "\n")

    sample_pdf_path = os.path.join(PROJECT_ROOT, "data", "documents", "sample_debate_brief.pdf")

    # -------------------------------------------------------------------------
    # STAGE 1: Load Document
    # -------------------------------------------------------------------------
    print("[STAGE 1] Loading Document with PyPDFLoader...")
    print(f"Target PDF: {sample_pdf_path}")
    raw_docs = load_pdf(sample_pdf_path)
    print(f"✓ Loaded {len(raw_docs)} document pages successfully.")
    for idx, doc in enumerate(raw_docs, start=1):
        print(f"  • Page {idx}: {len(doc.page_content)} characters | Source: {doc.metadata.get('source_file')}")

    # -------------------------------------------------------------------------
    # STAGE 2: Chunk Document
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Chunking Document into Semantic Segments...")
    print("Parameters: chunk_size=800, chunk_overlap=150")
    chunks = chunk_documents(raw_docs, chunk_size=800, chunk_overlap=150)
    print(f"✓ Created {len(chunks)} text chunks with enriched metadata.")
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        print(f"  • Chunk #{idx} ID: {meta.get('chunk_id')} | Length: {meta.get('character_count')} chars")

    # -------------------------------------------------------------------------
    # STAGE 3: Embeddings
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Initializing Sentence-Transformers Embeddings...")
    embedder = get_embedding_generator(model_name="all-MiniLM-L6-v2")
    print(f"✓ Model: {embedder.model_name} (Output Dimension: {embedder.dimension})")
    sample_text = chunks[0].page_content[:60] + "..."
    sample_vec = embedder.embed_query(chunks[0].page_content)
    print(f"  • Sample Embedding generated for: '{sample_text}'")
    print(f"  • Vector length: {len(sample_vec)} floats | First 5 values: {[round(x, 4) for x in sample_vec[:5]]}")

    # -------------------------------------------------------------------------
    # STAGE 4: Store in Persistent ChromaDB
    # -------------------------------------------------------------------------
    print("\n[STAGE 4] Storing in ChromaDB Persistent Vector Database...")
    db_path = os.path.join(PROJECT_ROOT, "data", "vector_db")
    print(f"Storage Directory: {db_path}")
    vector_store = ChromaVectorStore(persist_directory=db_path, embedder=embedder)

    # Clear previous run data to ensure clean state
    vector_store.clear_collection()

    # Index chunks
    inserted_ids = vector_store.add_documents(chunks)
    print(f"✓ Stored {len(inserted_ids)} chunks into collection '{vector_store.collection_name}'.")
    print(f"✓ Total items now in ChromaDB: {vector_store.count()}")

    # -------------------------------------------------------------------------
    # STAGE 5: Retrieve Relevant Chunks for a Debate Query
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Retrieving Relevant Chunks for Debate Query...")
    debate_query = "What evidence demonstrates that multi-agent debate reduces hallucinations in AI systems?"
    print(f"Query: \"{debate_query}\"")
    retriever = RAGRetriever(vector_store=vector_store, embedder=embedder, default_top_k=3)
    results = retriever.retrieve(query=debate_query, top_k=3)
    print(f"✓ Retrieved top-{len(results)} most relevant chunks:")

    for idx, res in enumerate(results, start=1):
        print(f"\n  [Result #{idx}]")
        print(f"  • Chunk ID        : {res.get('chunk_id')}")
        print(f"  • Source          : {res.get('source_file')}, Page {res.get('page_number')}")
        print(f"  • Similarity Score: {res.get('similarity_score')} (Distance: {res.get('distance')})")
        print(f"  • Content Snippet : \"{res.get('content')[:120]}...\"")

    # -------------------------------------------------------------------------
    # STAGE 6: Grounding & Evidence Formatting
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Synthesizing Grounded Context & Debate Agent Prompt...")
    evidence_context = GroundingManager.format_evidence_context(results)
    proponent_prompt = GroundingManager.create_grounded_prompt(
        query=debate_query,
        evidence_context=evidence_context,
        role="proponent",
    )

    print("\n--- FORMATTED EVIDENCE CONTEXT PASSED TO DEBATE AGENTS ---")
    print(evidence_context)

    print("\n--- PROMPT INSTRUCTION FOR PROPONENT AGENT ---")
    print(proponent_prompt[:650] + "\n... [Prompt continues with strict citation instructions] ...")

    # Test citation extraction helper
    mock_agent_response = (
        "Adversarial multi-agent debate reduces factual errors by over 65 percent "
        "[Source: sample_debate_brief.pdf, Page: 1]. Furthermore, in diagnostic trials, "
        "accuracy climbed from 71.4% to 89.2% [Source: sample_debate_brief.pdf, Page: 2]."
    )
    citations = GroundingManager.extract_citations(mock_agent_response)
    print(f"\n✓ Citation Extraction Audit Test: Found {len(citations)} citations:")
    for cit in citations:
        print(f"  • Cited File: {cit['source']} | Page: {cit['page']}")

    print("\n" + "=" * 80)
    print("      ✓ COMPLETE RAG & GROUNDING PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_rag_demo()
