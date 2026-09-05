"""Document Chunker Module for RAG Pipeline.

Splits loaded LangChain Document objects into smaller overlapping chunks
using RecursiveCharacterTextSplitter while maintaining trace metadata.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Splits full document pages into smaller, semantically coherent text chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum character count per chunk (default: 800).
            chunk_overlap: Character overlap between consecutive chunks (default: 150).
            separators: Hierarchy of characters used to split text (paragraphs, newlines, spaces).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Standard splitting hierarchy: double newline -> single newline -> space -> character
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into smaller chunked Document objects.

        Preserves existing metadata and adds unique identifiers and chunk statistics.

        Args:
            documents: List of input LangChain Document objects.

        Returns:
            List of chunked Document objects with enriched metadata.
        """
        if not documents:
            return []

        # Split documents using LangChain text splitter
        split_chunks = self.text_splitter.split_documents(documents)

        # Enhance chunk metadata for citation and debate grounding
        enriched_chunks: List[Document] = []
        for index, chunk in enumerate(split_chunks):
            source_file = chunk.metadata.get("source_file", "unknown_source")
            page_number = chunk.metadata.get("page_number", 1)

            # Construct clean deterministic ID: e.g. "report.pdf#page_2#chunk_5"
            chunk_id = f"{source_file}#page_{page_number}#chunk_{index}"

            updated_metadata = dict(chunk.metadata)
            updated_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "character_count": len(chunk.page_content),
                }
            )

            enriched_chunks.append(
                Document(
                    page_content=chunk.page_content.strip(),
                    metadata=updated_metadata,
                )
            )

        return enriched_chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Convenience helper function to chunk a list of documents in one call.

    Args:
        documents: List of Document objects to split.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlapping character count.

    Returns:
        List of chunked Document objects.
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_documents(documents)
