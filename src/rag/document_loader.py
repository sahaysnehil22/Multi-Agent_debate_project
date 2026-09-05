"""Document Loader Module for RAG Pipeline.

Loads PDF documents using LangChain's PyPDFLoader, performs path validation,
and returns standardized LangChain Document objects with source and page metadata.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:
    """Handles loading and validating PDF documents for the debate RAG system."""

    def __init__(self, file_path: str):
        """Initialize loader with target file path.

        Args:
            file_path: Path to the PDF file to load.
        """
        self.file_path = str(file_path)
        self._validate_path()

    def _validate_path(self) -> None:
        """Validate that the file exists and has a .pdf extension."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Document file not found at: '{self.file_path}'. "
                f"Please verify that the file exists in the documents directory."
            )

        if not os.path.isfile(self.file_path):
            raise ValueError(
                f"Path '{self.file_path}' is a directory, not a valid document file."
            )

        if not self.file_path.lower().endswith(".pdf"):
            raise ValueError(
                f"Unsupported file format for '{self.file_path}'. "
                f"Only PDF files (.pdf) are supported by DocumentLoader."
            )

    def load(self) -> List[Document]:
        """Load and parse the PDF document into LangChain Document objects.

        Returns:
            A list of Document objects representing individual pages.

        Raises:
            RuntimeError: If document parsing fails.
        """
        try:
            loader = PyPDFLoader(self.file_path)
            raw_docs = loader.load()
        except Exception as err:
            raise RuntimeError(
                f"Failed to read and parse PDF file '{self.file_path}': {err}"
            ) from err

        if not raw_docs:
            raise ValueError(
                f"The PDF file '{self.file_path}' appears to be empty or contains no readable text."
            )

        # Standardize metadata across documents
        cleaned_docs: List[Document] = []
        file_name = Path(self.file_path).name

        for doc in raw_docs:
            # PyPDFLoader usually provides 0-based 'page', convert to 1-based for humans
            raw_page = doc.metadata.get("page", 0)
            page_number = int(raw_page) + 1 if isinstance(raw_page, int) else 1

            metadata = {
                "source_file": file_name,
                "file_path": os.path.abspath(self.file_path),
                "page_number": page_number,
                "total_pages": len(raw_docs),
            }

            cleaned_docs.append(
                Document(
                    page_content=doc.page_content.strip(),
                    metadata=metadata,
                )
            )

        return cleaned_docs


def load_pdf(file_path: str) -> List[Document]:
    """Convenience helper function to load a PDF file in a single call.

    Args:
        file_path: Path to the target PDF.

    Returns:
        List of loaded Document objects.
    """
    loader = DocumentLoader(file_path=file_path)
    return loader.load()
