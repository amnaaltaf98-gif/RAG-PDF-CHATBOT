"""
test_ingest.py
--------------
Unit tests for ingest.py (extraction + chunking).
Run with: pytest tests/test_ingest.py -v
"""

import pytest
from app.ingest import chunk_text, extract_text, CHUNK_SIZE


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    text = "Hello world. " * 200  # long enough to produce multiple chunks
    chunks = chunk_text(text)
    assert len(chunks) > 1, "Expected multiple chunks for long text"


def test_chunk_size_respected():
    text = "A" * 5000
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    for chunk in chunks:
        # Allow a small buffer for splitter edge cases
        assert len(chunk) <= 600, f"Chunk too large: {len(chunk)} chars"


def test_chunk_text_empty_string():
    chunks = chunk_text("")
    assert chunks == [], "Empty string should return no chunks"


def test_chunk_text_short_string():
    text = "Short text."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_overlap_creates_shared_content():
    # With overlap, adjacent chunks should share some text
    text = ("The quick brown fox jumps over the lazy dog. " * 50)
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    if len(chunks) > 1:
        # Last 50 chars of chunk[0] should appear somewhere in chunk[1]
        overlap_text = chunks[0][-30:]
        assert overlap_text in chunks[1], "Expected overlap between adjacent chunks"


# ---------------------------------------------------------------------------
# extract_text (requires a real PDF — skip if not available)
# ---------------------------------------------------------------------------

def test_extract_text_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text("nonexistent_file.pdf")


def test_chunk_text_strips_whitespace():
    text = "  Hello world.  " * 100
    chunks = chunk_text(text)
    for chunk in chunks:
        assert chunk == chunk.strip(), "Chunks should be stripped of leading/trailing whitespace"
