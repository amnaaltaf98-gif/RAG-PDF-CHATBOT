"""
ui.py - Gradio 6 compatible UI with per-file progress, file status table,
duplicate prevention, and improved chat display.
"""

import shutil
import time
from pathlib import Path
import gradio as gr
from app.pipeline import ingest_pdf, query
from app.vector_store import count, _get_collection

UPLOAD_DIR = Path(r"C:\Users\amnaa\Desktop\RAG Chatbot\Uploads")
if not UPLOAD_DIR.exists():
    UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Track which files have already been indexed this session
_indexed_files: set[str] = set()


def get_indexed_filenames() -> set[str]:
    """Read already-indexed source names from ChromaDB metadata."""
    try:
        col = _get_collection()
        if col.count() == 0:
            return set()
        results = col.get(include=["metadatas"])
        return {m.get("source", "") for m in results["metadatas"] if m}
    except Exception:
        return set()


def build_file_table(statuses: list[dict]) -> str:
    """Render a markdown table of file statuses."""
    if not statuses:
        return "*No files processed yet.*"
    rows = ["| File | Status | Chunks |", "|---|---|---|"]
    for s in statuses:
        icon = {"done": "✅", "error": "❌", "skipped": "⏭️", "indexing": "⏳"}.get(s["status"], "❓")
        rows.append(f"| {s['name']} | {icon} {s['status'].capitalize()} | {s.get('chunks', '—')} |")
    return "\n".join(rows)


def handle_index_folder(progress=gr.Progress()):
    """Scan UPLOAD_DIR, skip already-indexed files, show per-file progress."""
    pdfs = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdfs:
        yield (
            f"⚠️ No PDFs found in `{UPLOAD_DIR}`",
            "*No files found.*",
            f"**Total chunks in store: {count()}**"
        )
        return

    already_indexed = get_indexed_filenames()
    statuses = []
    total = len(pdfs)

    for i, pdf in enumerate(pdfs):
        progress((i / total), desc=f"Processing {pdf.name}...")

        if pdf.name in already_indexed:
            statuses.append({"name": pdf.name, "status": "skipped", "chunks": "already indexed"})
            yield (
                f"⏭️ Skipping `{pdf.name}` (already indexed)...",
                build_file_table(statuses),
                f"**Total chunks in store: {count()}**"
            )
            continue

        statuses.append({"name": pdf.name, "status": "indexing", "chunks": "..."})
        yield (
            f"⏳ Indexing `{pdf.name}`... ({i+1}/{total})",
            build_file_table(statuses),
            f"**Total chunks in store: {count()}**"
        )

        try:
            result = ingest_pdf(pdf)
            statuses[-1] = {"name": pdf.name, "status": "done", "chunks": result["chunks_added"]}
            already_indexed.add(pdf.name)
        except Exception as e:
            statuses[-1] = {"name": pdf.name, "status": "error", "chunks": str(e)[:40]}

        progress(((i + 1) / total), desc=f"Done {pdf.name}")
        yield (
            f"✅ Indexed `{pdf.name}` ({i+1}/{total})",
            build_file_table(statuses),
            f"**Total chunks in store: {count()}**"
        )

    done = sum(1 for s in statuses if s["status"] == "done")
    skipped = sum(1 for s in statuses if s["status"] == "skipped")
    errors = sum(1 for s in statuses if s["status"] == "error")
    summary = f"✅ Done — {done} indexed, {skipped} skipped, {errors} errors"
    yield (summary, build_file_table(statuses), f"**Total chunks in store: {count()}**")


def handle_index_uploads(files, progress=gr.Progress()):
    """Index manually uploaded files with per-file progress."""
    if not files:
        yield ("⚠️ No files selected.", "*No files uploaded.*", f"**Total: {count()}**")
        return

    already_indexed = get_indexed_filenames()
    statuses = []
    total = len(files)

    for i, file in enumerate(files):
        src = Path(file.name)
        progress((i / total), desc=f"Processing {src.name}...")

        if src.name in already_indexed:
            statuses.append({"name": src.name, "status": "skipped", "chunks": "already indexed"})
            yield (
                f"⏭️ Skipping `{src.name}` (already indexed)",
                build_file_table(statuses),
                f"**Total chunks in store: {count()}**"
            )
            continue

        dest = UPLOAD_DIR / src.name
        shutil.copy(src, dest)
        statuses.append({"name": src.name, "status": "indexing", "chunks": "..."})
        yield (
            f"⏳ Indexing `{src.name}`... ({i+1}/{total})",
            build_file_table(statuses),
            f"**Total chunks in store: {count()}**"
        )

        try:
            result = ingest_pdf(dest)
            statuses[-1] = {"name": src.name, "status": "done", "chunks": result["chunks_added"]}
            already_indexed.add(src.name)
        except Exception as e:
            statuses[-1] = {"name": src.name, "status": "error", "chunks": str(e)[:40]}

        progress(((i + 1) / total), desc=f"Done {src.name}")
        yield (
            f"✅ `{src.name}` done ({i+1}/{total})",
            build_file_table(statuses),
            f"**Total chunks in store: {count()}**"
        )

    done = sum(1 for s in statuses if s["status"] == "done")
    skipped = sum(1 for s in statuses if s["status"] == "skipped")
    summary = f"✅ Done — {done} indexed, {skipped} skipped"
    yield (summary, build_file_table(statuses), f"**Total chunks in store: {count()}**")


def handle_chat(message: str, history):
    if history is None:
        history = []

    if not message.strip():
        return "", history

    history.append({"role": "user", "content": message})

    if count() == 0:
        history.append({"role": "assistant", "content": "⚠️ No documents indexed yet. Please index your PDFs first."})
        return "", history

    try:
        result = query(message)

        answer_text = result["answer"]
        sources = result["sources"]
        chunks_used = result["chunks_used"]

        footer = (
            f"\n\n---\n"
            f"📄 Sources: {', '.join(sources)}\n"
            f"🔍 Chunks retrieved: {chunks_used}"
        )

        answer_text += footer

    except Exception as e:
        answer_text = f"❌ Error: {str(e)}"

    history.append({"role": "assistant", "content": answer_text})

    return "", history

def handle_clear():
    return []

def get_indexed_files_display() -> str:
    """Show all currently indexed files as a table."""
    try:
        col = _get_collection()
        if col.count() == 0:
            return "*No files indexed yet.*"
        results = col.get(include=["metadatas"])
        sources = {}
        for m in results["metadatas"]:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        rows = ["| File | Chunks |", "|---|---|"]
        for name, cnt in sorted(sources.items()):
            rows.append(f"| {name} | {cnt} |")
        return "\n".join(rows)
    except Exception:
        return "*Could not load indexed files.*"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG PDF Chatbot") as demo:

        gr.Markdown("""# 📄 RAG PDF Chatbot
Powered by **LLaMA 3.3 70B** via Groq + **ChromaDB** vector search""")

        with gr.Tabs():

            # ================================================================
            # TAB 1: Index
            # ================================================================
            with gr.Tab("📂 Index PDFs"):

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(f"### Uploads Folder\n`{UPLOAD_DIR}`")
                        index_folder_btn = gr.Button(
                            "⚡ Index All PDFs in Folder", variant="primary", size="lg"
                        )

                        gr.Markdown("---\n### Or Upload Additional PDFs")
                        file_input = gr.File(
                            label="Drag & drop PDFs here",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )
                        index_upload_btn = gr.Button("⚡ Index Uploaded Files", variant="secondary")

                    with gr.Column(scale=3):
                        gr.Markdown("### Progress")
                        status_msg = gr.Markdown("*Click a button to start indexing.*")
                        file_table = gr.Markdown("*File statuses will appear here.*")
                        chunk_count = gr.Markdown(f"**Total chunks in store: {count()}**")

                gr.Markdown("---\n### Currently Indexed Files")
                indexed_display = gr.Markdown(get_indexed_files_display())
                refresh_btn = gr.Button("🔄 Refresh file list", size="sm", variant="secondary")

            # ================================================================
            # TAB 2: Chat
            # ================================================================
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("### Index Summary")
                        chat_chunk_count = gr.Markdown(f"**{count()} chunks** in store")
                        chat_files = gr.Markdown(get_indexed_files_display())
                        refresh_chat_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")

                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(label="Chat", height=500)
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a question about your PDFs...",
                                label="Your question",
                                scale=5,
                                lines=1,
                            )
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑️ Clear chat", size="sm", variant="secondary")

        # ----------------------------------------------------------------
        # Wire events
        # ----------------------------------------------------------------
        index_folder_btn.click(
            fn=handle_index_folder,
            inputs=[],
            outputs=[status_msg, file_table, chunk_count],
        )

        index_upload_btn.click(
            fn=handle_index_uploads,
            inputs=[file_input],
            outputs=[status_msg, file_table, chunk_count],
        )

        refresh_btn.click(fn=get_indexed_files_display, outputs=[indexed_display])

        refresh_chat_btn.click(
            fn=lambda: (f"**{count()} chunks** in store", get_indexed_files_display()),
            outputs=[chat_chunk_count, chat_files],
        )

        send_btn.click(
            fn=handle_chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        msg_input.submit(
            fn=handle_chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        clear_btn.click(fn=handle_clear, outputs=[chatbot])

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    demo = build_ui()
    demo.launch(
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
    )