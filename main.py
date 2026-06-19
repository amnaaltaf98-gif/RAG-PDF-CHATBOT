"""
main.py
-------
Entry point. Run with:

    python main.py               # Launches Gradio UI at http://localhost:7860
    python main.py --cli         # Interactive CLI mode (no UI)
    python main.py --share       # Gradio public share link
"""

import utils.session
import argparse
from dotenv import load_dotenv

load_dotenv()  # Load .env before any other imports that read env vars


def run_ui(share: bool = False) -> None:
    from app.ui import launch
    print("Starting Gradio UI at http://localhost:7860 ...")
    launch(share=share)


def run_cli() -> None:
    """Simple terminal-based RAG loop (no UI dependencies)."""
    from app.pipeline import ingest_pdf, ingest_folder, query
    from app.vector_store import count

    print("\n=== RAG PDF Chatbot (CLI mode) ===")
    print("Commands:")
    print("  index <path/to/file.pdf>        — index a single PDF")
    print("  index-folder <path/to/folder>   — index all PDFs in a folder")
    print("  ask <question>                  — ask a question")
    print("  quit\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        elif user_input.lower().startswith("index-folder "):
            folder_path = user_input[13:].strip()
            try:
                results = ingest_folder(folder_path)
                for r in results:
                    print(f"✅ {r['filename']} — {r['chunks_added']} chunks")
                print(f"Total in store: {count()}\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

        elif user_input.lower().startswith("index "):
            pdf_path = user_input[6:].strip()
            try:
                result = ingest_pdf(pdf_path)
                print(f"✅ Indexed {result['chunks_added']} chunks from '{result['filename']}'.")
                print(f"   Total in store: {result['total_in_store']}\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

        elif user_input.lower().startswith("ask "):
            question = user_input[4:].strip()
            if count() == 0:
                print("⚠️  No documents indexed yet. Use: index <path>\n")
                continue
            result = query(question)
            print(f"\nAnswer:\n{result['answer']}")
            if result["sources"]:
                print(f"\nSources: {', '.join(result['sources'])}")
            print()

        else:
            print("Unknown command. Use 'index <pdf>' or 'ask <question>'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG PDF Chatbot")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Gradio UI")
    parser.add_argument("--share", action="store_true", help="Generate a public Gradio share link")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        run_ui(share=args.share)
