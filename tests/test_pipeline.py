"""
test_pipeline.py
----------------
Integration tests for the embed → store → retrieve cycle.
Does NOT call the Groq API (avoids network/cost in CI).
Run with: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest
from app.embedder import embed, embed_query
from app.vector_store import add_documents, retrieve, clear, count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_store():
    """Wipe the vector store before each test to avoid cross-test pollution."""
    clear()
    yield
    clear()


SAMPLE_CHUNKS = [
    "The mitochondria is the powerhouse of the cell.",
    "Photosynthesis converts sunlight into chemical energy.",
    "DNA carries genetic information in living organisms.",
    "Neurons transmit electrical signals in the brain.",
    "The speed of light in a vacuum is approximately 299,792 km/s.",
]


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

def test_embed_returns_correct_shape():
    embeddings = embed(SAMPLE_CHUNKS)
    assert embeddings.shape == (len(SAMPLE_CHUNKS), 384)


def test_embed_query_returns_1d():
    q = embed_query("What is DNA?")
    assert q.ndim == 1
    assert q.shape[0] == 384


def test_embed_empty_raises():
    with pytest.raises(ValueError):
        embed([])


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

def test_add_and_count():
    embeddings = embed(SAMPLE_CHUNKS)
    add_documents(SAMPLE_CHUNKS, embeddings, source="test.pdf")
    assert count() == len(SAMPLE_CHUNKS)


def test_retrieve_returns_top_k():
    embeddings = embed(SAMPLE_CHUNKS)
    add_documents(SAMPLE_CHUNKS, embeddings, source="test.pdf")

    q_emb = embed_query("What is DNA?")
    results = retrieve(q_emb, top_k=2)

    assert len(results) == 2
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "score" in r


def test_retrieve_most_relevant_chunk():
    embeddings = embed(SAMPLE_CHUNKS)
    add_documents(SAMPLE_CHUNKS, embeddings, source="test.pdf")

    q_emb = embed_query("What carries genetic information?")
    results = retrieve(q_emb, top_k=1)

    # The DNA chunk should be the top result
    assert "DNA" in results[0]["text"]


def test_retrieve_empty_store_returns_empty():
    q_emb = embed_query("anything")
    results = retrieve(q_emb)
    assert results == []


def test_scores_between_0_and_1():
    embeddings = embed(SAMPLE_CHUNKS)
    add_documents(SAMPLE_CHUNKS, embeddings)

    q_emb = embed_query("biology")
    results = retrieve(q_emb, top_k=3)

    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"
