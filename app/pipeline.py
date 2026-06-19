"""
pipeline.py
-----------
Responsibility: orchestrate the full RAG flow.

Ingest flow:  PDF path → extract → chunk → embed → store
Query flow:   question → embed → retrieve → answer
"""

import re
from pathlib import Path
from app.ingest import load_pdf
from app.embedder import embed, embed_query
from app.vector_store import add_documents, retrieve, count, get_all_sources
from app.llm import answer


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str | Path) -> dict:
    """Full ingest pipeline: PDF → chunks → embeddings → vector store."""
    pdf_path = Path(pdf_path)
    print(f"\n[Ingest] Processing '{pdf_path.name}'...")

    print("  → Extracting and chunking text...")
    chunks = load_pdf(pdf_path)
    print(f"  → {len(chunks)} chunks created.")

    print("  → Generating embeddings (this may take a moment)...")
    embeddings = embed(chunks, show_progress=True)
    print(f"  → Embeddings shape: {embeddings.shape}")

    print("  → Storing in ChromaDB...")
    added = add_documents(chunks, embeddings, source=pdf_path.name)
    total = count()
    print(f"  → Done. {added} chunks added. {total} total chunks in store.\n")

    return {
        "filename": pdf_path.name,
        "chunks_added": added,
        "total_in_store": total,
    }


def ingest_multiple(pdf_paths: list[str | Path]) -> list[dict]:
    """Ingest a list of PDF files. Returns a list of ingest result dicts."""
    return [ingest_pdf(p) for p in pdf_paths]


def ingest_folder(folder_path: str | Path) -> list[dict]:
    """Ingest all PDFs found in a folder (non-recursive)."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return []

    print(f"Found {len(pdfs)} PDF(s) in {folder}")
    return ingest_multiple(pdfs)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def _extract_years(text: str) -> list[str]:
    """Find 4-digit years (19xx or 20xx) mentioned in a string."""
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def query(question: str, top_k: int = 4) -> dict:
    """
    Full query pipeline: question → retrieve relevant chunks → LLM answer.

    If the question mentions specific years (e.g. "2025"), retrieval is
    restricted to source files whose filename contains that year, and chunks
    are pulled evenly across each matching file. Otherwise, falls back to a
    normal similarity search across all indexed chunks.
    """
    if count() == 0:
        return {
            "answer": "No documents have been indexed yet. Please upload and index a PDF first.",
            "sources": [],
            "chunks_used": 0,
        }

    q_embedding = embed_query(question)

    # 1. Find years mentioned in the question
    question_years = _extract_years(question)

    # 2. Match against filenames of indexed sources
    targeted_sources = []
    if question_years:
        unique_sources = get_all_sources()
        for source in unique_sources:
            if any(year in source for year in question_years):
                targeted_sources.append(source)

    # 3. Retrieve chunks — targeted by year if matched, else open search
    if targeted_sources:
        chunks_per_file = max(2, top_k // len(targeted_sources))
        all_chunks = []
        for source in targeted_sources:
            all_chunks.extend(
                retrieve(q_embedding, top_k=chunks_per_file, source_filter=source)
            )
    else:
        all_chunks = retrieve(q_embedding, top_k=top_k)

    # 4. Generate the final answer using the retrieved chunks
    response = answer(question, all_chunks)
    sources = list(dict.fromkeys(c["source"] for c in all_chunks))

    return {
        "answer": response,
        "sources": sources,
        "chunks_used": len(all_chunks),
    }