"""Grounding Module for RAG Pipeline.

Prepares retrieved evidence chunks into structured context blocks and builds
grounding prompt instructions to ensure debate agents (Proponent, Opponent, Moderator)
reason strictly on document-verified claims and provide citations.
"""

import re
from typing import Any, Dict, List, Optional


class GroundingManager:
    """Prepares retrieved document evidence and builds grounded agent prompts."""

    @staticmethod
    def format_evidence_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clear, cited evidence context string.

        Args:
            retrieved_chunks: List of chunk dictionaries returned by RAGRetriever.

        Returns:
            A formatted multi-line string containing structured evidence with metadata.
        """
        if not retrieved_chunks:
            return "No relevant document evidence found for this query."

        lines = [
            "======================= VERIFIED DOCUMENT EVIDENCE =======================",
            "The following evidence excerpts were retrieved from local reference documents.",
            "Agents must base their claims and rebuttals STRICTLY on these excerpts.",
            "",
        ]

        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source_file") or chunk.get("metadata", {}).get("source_file", "unknown")
            page = chunk.get("page_number") or chunk.get("metadata", {}).get("page_number", "N/A")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            similarity = chunk.get("similarity_score", 0.0)
            content = chunk.get("content", "").strip()

            lines.append(f"--- [Evidence #{i}] ---")
            lines.append(f"Source File   : {source}")
            lines.append(f"Page Number   : {page}")
            lines.append(f"Chunk ID      : {chunk_id}")
            lines.append(f"Relevance     : {similarity:.2%}")
            lines.append("Content:")
            lines.append(f'"{content}"')
            lines.append("")

        lines.append("==========================================================================")
        return "\n".join(lines)

    @staticmethod
    def create_grounded_prompt(
        query: str,
        evidence_context: str,
        role: str = "proponent",
        counter_argument: Optional[str] = None,
    ) -> str:
        """Build an instruction prompt that forces an LLM agent to ground its answer in evidence.

        Args:
            query: The debate premise or topic.
            evidence_context: Formatted evidence string produced by format_evidence_context.
            role: Agent role ('proponent', 'opponent', or 'moderator').
            counter_argument: Optional counter-argument to rebut.

        Returns:
            A complete, structured prompt for a debate agent.
        """
        role_titles = {
            "proponent": "Proponent (Agent Alpha) - Defend the Thesis",
            "opponent": "Opponent (Agent Beta) - Challenge and Audit the Thesis",
            "moderator": "Moderator (Orchestrator) - Impartial Consensus Judge",
        }
        role_title = role_titles.get(role.lower(), "Debate Agent")

        prompt_lines = [
            f"### ROLE: You are the {role_title}.",
            "",
            "### GROUNDING & FACTUALITY CONSTRAINTS:",
            "1. STRICT FACTUALITY: You MUST base all factual statements ONLY on the verified evidence below.",
            "2. NO HALLUCINATIONS: Do not assume, fabricate, or extrapolate facts not present in the evidence.",
            "3. MANDATORY CITATIONS: Whenever you state a fact or point of evidence, you MUST cite it explicitly",
            "   using the format: [Source: <filename>, Page: <page_number>].",
            "4. MISSING EVIDENCE: If the evidence does not contain sufficient data to support or refute a point,",
            "   explicitly declare: 'Insufficient evidence in provided documents.'",
            "",
            "### TOPIC / PREMISE:",
            f"{query}",
            "",
        ]

        if counter_argument:
            prompt_lines.extend([
                "### OPPOSING ARGUMENT TO REBUT:",
                f"{counter_argument}",
                "",
            ])

        prompt_lines.extend([
            "### RETRIEVED DOCUMENT EVIDENCE:",
            f"{evidence_context}",
            "",
            "### YOUR ARGUMENTATION TASK:",
            f"Formulate your {role.lower()} response following the constraints above. Always include evidence citations.",
        ])

        return "\n".join(prompt_lines)

    @staticmethod
    def extract_citations(agent_response: str) -> List[Dict[str, str]]:
        """Extract citations from an agent response for auditing by the Moderator.

        Searches for citations matching patterns like:
        `[Source: filename.pdf, Page: 2]` or `(Source: filename.pdf, Page: 2)`

        Args:
            agent_response: Text generated by a debate agent.

        Returns:
            List of parsed citation dictionaries: [{'source': 'doc.pdf', 'page': '2'}].
        """
        pattern = r"[\[\(]Source:\s*([^,\]\)]+),\s*Page:\s*([^\]\)]+)[\]\)]"
        matches = re.findall(pattern, agent_response, re.IGNORECASE)

        citations: List[Dict[str, str]] = []
        for source, page in matches:
            citations.append({
                "source": source.strip(),
                "page": page.strip(),
            })

        return citations


# Convenience helper functions
def format_evidence_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into evidence context string."""
    return GroundingManager.format_evidence_context(retrieved_chunks)


def create_grounded_prompt(
    query: str,
    evidence_context: str,
    role: str = "proponent",
    counter_argument: Optional[str] = None,
) -> str:
    """Build grounded debate agent prompt."""
    return GroundingManager.create_grounded_prompt(
        query=query,
        evidence_context=evidence_context,
        role=role,
        counter_argument=counter_argument,
    )
