# RAG PDF Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and ask questions about them. Uses free tools only.

## Stack

| Role | Tool |
|---|---|
| PDF parsing | PyMuPDF |
| Chunking | LangChain |
| Embeddings | `all-MiniLM-L6-v2` (runs locally) |
| Vector DB | ChromaDB (local, persistent) |
| LLM | Groq API + LLaMA 3 (free tier) |
| UI | Gradio |

## Setup

### 1. Clone & create virtual environment
```bash
git clone <your-repo>
cd rag-pdf-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Get a free key at https://console.groq.com

```bash
cp .env.example .env
# Edit .env and paste your key
```

### 4. Run the app
```bash
python main.py
```

Then open http://localhost:7860 in your browser.

## Project Structure

```
rag-pdf-chatbot/
├── .env                  # Your API keys (never commit)
├── requirements.txt
├── main.py               # Entry point
├── app/
│   ├── ingest.py         # PDF extraction + chunking
│   ├── embedder.py       # Sentence embeddings
│   ├── vector_store.py   # ChromaDB wrapper
│   ├── llm.py            # Groq + LLaMA answer generation
│   ├── pipeline.py       # Orchestrates the full flow
│   └── ui.py             # Gradio interface
├── uploads/              # Temporary PDF storage
├── chroma_db/            # Persistent vector store
└── tests/
    ├── test_ingest.py
    └── test_pipeline.py
```

## Usage

1. Upload one or more PDFs using the file picker
2. Click **Index PDFs** and wait for confirmation
3. Type your question in the chat box
4. Get answers grounded in your documents

## Example Chat

Indexed: Apple 10-K filings (2013–2025)

> **Q: What was Apple's total net sales for the fiscal year ended September 27, 2025?**
> Apple's total net sales for the fiscal year ended September 27, 2025, was $416,161 million. (Source: aapl-20250927-2025.pdf)

> **Q: Compare iPhone net sales between 2024 and 2025. Did they increase or decrease?**
> iPhone net sales increased by 4% in 2025 compared to 2024, from $201,183 million to $209,586 million. (Source: aapl-20250927-2025.pdf, aapl-20240928-2024.pdf)

> **Q: Provide Apple's exact total net sales for the fiscal years 2013, 2018, and 2025 as reported in their respective 10-K filings.**
> - 2018: $265,595 million (source: 2018.pdf)
> - 2025: $416,161 million (source: aapl-20250927-2025.pdf)
> - 2013: not explicitly stated in the retrieved context — only the year-over-year change ($14.4B increase) was found.

The chatbot automatically detects years mentioned in a question (e.g. "2025", "2018") and restricts retrieval to the matching 10-K filings, pulling chunks evenly across each one — which is why multi-year comparison questions return sources from all relevant files.
