"""
ingest.py
---------
Responsibility: PDF → raw text → chunks.
No knowledge of embeddings, vector stores, or the LLM.
"""

import fitz  # PyMuPDF
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 2000      # characters per chunk (not tokens, but close enough)
CHUNK_OVERLAP = 300    # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Full text of the document as a single string.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF has no extractable text (e.g. scanned image PDF).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text.strip()}")

    doc.close()

    if not pages:
        raise ValueError(
            f"No extractable text found in {pdf_path.name}. "
            "The PDF may be a scanned image. Try an OCR tool first."
        )

    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long string into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on paragraph
    boundaries first, then sentences, then words, before hard-splitting.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Combined helper
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: str | Path) -> list[str]:
    """
    Extract text from a PDF and return it as a list of chunks.
    Convenience wrapper around extract_text + chunk_text.
    """
    text = extract_text(pdf_path)
    return chunk_text(text)
