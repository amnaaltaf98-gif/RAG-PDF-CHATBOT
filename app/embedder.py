"""
embedder.py
-----------
Responsibility: text → dense vector embeddings.
Runs entirely locally — no API key, no cost.

Model: all-MiniLM-L6-v2
  - ~80 MB download (cached after first run)
  - 384-dimensional embeddings
  - Fast on CPU, great quality for RAG retrieval
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Model (loaded once, reused across calls)
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model so it's only downloaded once."""
    global _model
    if _model is None:
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """
    Convert a list of strings into a 2-D numpy array of embeddings.

    Args:
        texts: List of strings to embed.
        show_progress: Show a tqdm progress bar (useful for large batches).

    Returns:
        numpy array of shape (len(texts), 384).
    """
    if not texts:
        raise ValueError("Cannot embed an empty list.")

    model = _get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string. Returns shape (384,).
    Thin wrapper around embed() for clarity at call sites.
    """
    return embed([query])[0]
