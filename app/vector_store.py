"""
vector_store.py
---------------
Responsibility: store embeddings in ChromaDB, retrieve top-k similar chunks.
Persists to disk so the index survives restarts.
"""

import numpy as np
import chromadb
from chromadb.config import Settings
from pathlib import Path


DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "pdf_chunks"

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    """Return (creating if needed) the persistent ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(chunks: list[str], embeddings: np.ndarray, source: str = "upload") -> int:
    """Store text chunks + their embeddings in ChromaDB."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length.")

    collection = _get_collection()
    existing = collection.count()
    ids = [f"{source}_{existing + i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": existing + i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return len(chunks)


def retrieve(query_embedding: np.ndarray, top_k: int = 4, source_filter: str | None = None) -> list[dict]:
    """
    Find the top-k most similar chunks for a query embedding.

    Args:
        query_embedding: 1-D numpy array from embedder.embed_query().
        top_k: Number of results to return.
        source_filter: If given, only search within this source filename.

    Returns:
        List of dicts with keys: 'text', 'source', 'score'.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    where_clause = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    # query() can return zero matches for a filter with no rows
    if not results["documents"] or not results["documents"][0]:
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": round(1 - dist, 4),
        })
    return output


def get_all_sources() -> list[str]:
    """Return a list of unique source filenames currently indexed."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    return list({m["source"] for m in all_meta if m and "source" in m})


def clear() -> None:
    """Delete all documents from the collection (useful for testing)."""
    global _collection
    if _client is not None and _collection is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None


def count() -> int:
    """Return total number of chunks stored."""
    return _get_collection().count()