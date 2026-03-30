import argparse
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

STORE_DIR = Path("rag_store")
INDEX_PATH = STORE_DIR / "index.faiss"
CHUNKS_PATH = STORE_DIR / "chunks.json"

EMBED_MODEL = "gemini-embedding-001"
#LLM_MODEL = "gemini-2.0-flash"
LLM_MODEL = "gemini-3-flash-preview"


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")

    client = _client()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    print(f"Generated embeddings for {len(texts)} texts.")
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    print(f"Embedding shape: {vectors.shape}")
    faiss.normalize_L2(vectors)
    print("Normalized embeddings.")
    print(vectors[:2])  # Print first 2 vectors for inspection
    return vectors


def _load_store() -> tuple[faiss.Index, list[str]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("RAG store not found. Run ingest first.")

    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


def ingest(pdf_path: str) -> None:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    reader = PdfReader(str(pdf))
    full_text = " ".join(_clean_text(page.extract_text()) for page in reader.pages)
    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    vectors = _embed(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ingested {len(chunks)} chunks from {pdf}.")


def answer_question(question: str, k: int = 5) -> None:
    index, chunks = _load_store()
    print(f"Loaded {len(chunks)} chunks from store.")

    query_vector = _embed([question])
    top_k = max(1, min(k, len(chunks)))
    _, indices = index.search(query_vector, top_k)

    selected = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(selected))

    prompt = (
        "Use the context to answer the question. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    client = _client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You answer based on provided context only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    print(response.choices[0].message.content or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple PDF RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF")
    p_ingest.add_argument("--pdf", required=True, help="Path to PDF")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", help="User question")
    p_ask.add_argument("--k", type=int, default=5, help="Top-k chunks")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.pdf)
    elif args.cmd == "ask":
        answer_question(args.question, k=args.k)


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    main()
