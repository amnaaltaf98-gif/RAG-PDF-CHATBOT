"""
llm.py
------
Responsibility: take a question + retrieved context chunks → generate an answer.
Uses Groq API with LLaMA 3.3 (free tier, very fast).
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config — updated to current supported model
# ---------------------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"   # replaces decommissioned llama3-8b-8192
MAX_TOKENS = 1024
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a helpful financial analyst assistant that answers questions strictly based on the provided document context (Apple Inc. annual reports).

Rules:
- Only use information from the context below to answer.
- If the answer is not in the context, say "I couldn't find that in the uploaded documents."
- Be concise and accurate. Cite years/figures when available.
- If quoting directly, mention which source file it came from."""

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        parts.append(f"[Chunk {i} — source: {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def answer(question: str, context_chunks: list[dict]) -> str:
    context = build_context(context_chunks)
    user_message = f"""Context from uploaded documents:

{context}

---

Question: {question}"""

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()