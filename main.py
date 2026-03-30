import argparse
import json
import os
import re
from pathlib import Path

import chromadb
import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

STORE_DIR = Path("rag_store")
INDEX_PATH = STORE_DIR / "index.faiss"
CHUNKS_PATH = STORE_DIR / "chunks.json"
CHROMA_DIR = STORE_DIR / "chroma"
CHROMA_COLLECTION = "pdf_chunks"
ENV_PATH = Path(".env")

EMBED_MODEL = "gemini-embedding-001"
#LLM_MODEL = "gemini-2.0-flash"
LLM_MODEL = "gemini-3-flash-preview"


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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


def _chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _save_faiss_store(chunks: list[str], vectors: np.ndarray) -> None:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved FAISS store.")


def _save_chroma_store(pdf: Path, chunks: list[str], vectors: np.ndarray) -> None:
    collection = _chroma_collection()
    try:
        collection.delete(where={"source": str(pdf)})
    except Exception:
        pass

    ids = [f"{pdf.stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": str(pdf), "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=vectors.tolist(),
        metadatas=metadatas,
    )
    print("Saved ChromaDB store.")


def _search_faiss(query_vector: np.ndarray, k: int) -> list[str]:
    index, chunks = _load_store()
    top_k = max(1, min(k, len(chunks)))
    _, indices = index.search(query_vector.reshape(1, -1), top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def _search_chroma(query_vector: np.ndarray, k: int) -> list[str]:
    collection = _chroma_collection()
    total = collection.count()
    if total == 0:
        return []
    top_k = max(1, min(k, total))
    result = collection.query(query_embeddings=[query_vector.tolist()], n_results=top_k)
    docs = result.get("documents", [])
    if not docs:
        return []
    return [doc for doc in docs[0] if doc]


def ingest(pdf_path: str, vector_db: str = "both") -> None:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    reader = PdfReader(str(pdf))
    full_text = " ".join(_clean_text(page.extract_text()) for page in reader.pages)
    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    vectors = _embed(chunks)
    if vector_db in ("faiss", "both"):
        _save_faiss_store(chunks, vectors)
    if vector_db in ("chroma", "both"):
        _save_chroma_store(pdf, chunks, vectors)
    print(f"Ingested {len(chunks)} chunks from {pdf}.")


def answer_question(question: str, k: int = 5, vector_db: str = "both") -> None:
    query_vector = _embed([question])[0]
    selected: list[str] = []

    if vector_db in ("faiss", "both"):
        try:
            selected.extend(_search_faiss(query_vector, k))
        except FileNotFoundError:
            if vector_db == "faiss":
                raise
            print("FAISS store not found. Continuing with available stores.")

    if vector_db in ("chroma", "both"):
        selected.extend(_search_chroma(query_vector, k))

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise FileNotFoundError("No vector data found for the selected store(s). Run ingest first.")
    selected = selected[: max(1, k)]
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
    p_ingest.add_argument(
        "--vector-db",
        choices=["faiss", "chroma", "both"],
        default="both",
        help="Choose which vector DB to write to",
    )

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", help="User question")
    p_ask.add_argument("--k", type=int, default=5, help="Top-k chunks")
    p_ask.add_argument(
        "--vector-db",
        choices=["faiss", "chroma", "both"],
        default="both",
        help="Choose which vector DB to read from",
    )

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.pdf, vector_db=args.vector_db)
    elif args.cmd == "ask":
        answer_question(args.question, k=args.k, vector_db=args.vector_db)


if __name__ == "__main__":
    _load_env_file()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    main()
