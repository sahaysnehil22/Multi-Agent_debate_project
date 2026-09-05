"""Unit tests for GroundingManager and citation extraction."""

from src.rag.grounding import GroundingManager, format_evidence_context, create_grounded_prompt


def test_format_evidence_context_empty():
    """Test formatting when no chunks were retrieved."""
    context = format_evidence_context([])
    assert "No relevant document evidence found" in context


def test_format_evidence_context_structure():
    """Test formatting of retrieved evidence blocks."""
    mock_chunks = [
        {
            "chunk_id": "test.pdf#page_1#chunk_0",
            "source_file": "test.pdf",
            "page_number": 1,
            "similarity_score": 0.85,
            "content": "Adversarial debate improves consensus accuracy.",
        }
    ]
    formatted = format_evidence_context(mock_chunks)
    assert "VERIFIED DOCUMENT EVIDENCE" in formatted
    assert "test.pdf" in formatted
    assert "Page Number   : 1" in formatted
    assert "Adversarial debate improves consensus accuracy." in formatted


def test_create_grounded_prompt():
    """Test building grounded instructions for debate agents."""
    prompt = create_grounded_prompt(
        query="Should AI models be allowed to self-replicate?",
        evidence_context="Sample verified context.",
        role="proponent",
    )
    assert "Proponent (Agent Alpha)" in prompt
    assert "STRICT FACTUALITY" in prompt
    assert "MANDATORY CITATIONS" in prompt
    assert "Should AI models be allowed to self-replicate?" in prompt
    assert "Sample verified context." in prompt


def test_extract_citations():
    """Test extraction of source citations from agent arguments."""
    agent_text = (
        "According to research, safety filters fail in 15% of edge cases "
        "[Source: ai_safety_review.pdf, Page: 4]. Meanwhile, human oversight "
        "(Source: governance_2025.pdf, Page: 12) restores alignment."
    )
    citations = GroundingManager.extract_citations(agent_text)
    assert len(citations) == 2
    assert citations[0]["source"] == "ai_safety_review.pdf"
    assert citations[0]["page"] == "4"
    assert citations[1]["source"] == "governance_2025.pdf"
    assert citations[1]["page"] == "12"
