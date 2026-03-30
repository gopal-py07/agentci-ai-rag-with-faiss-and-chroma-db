# agenic-ai-rag

Simple PDF RAG (Retrieval-Augmented Generation) CLI app with FAISS + ChromaDB.

## What this project does

This project lets you:
1. Ingest a PDF file into vector storage.
2. Ask questions about that PDF.
3. Compare retrieval behavior using `faiss`, `chroma`, or `both`.

It uses:
- `pypdf` for reading PDF text
- `openai` client with Google Gemini OpenAI-compatible endpoint
- `faiss` for local vector search
- `chromadb` for persistent vector search

## Prerequisites

1. Python `3.14+` (as defined in `pyproject.toml`)
2. A valid Gemini API key

## Setup

### 1. Clone repo

```bash
git clone https://github.com/gopal-py07/agenic-ai-rag.git
cd agenic-ai-rag
```

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Set environment variable

PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

Command Prompt:

```cmd
set GEMINI_API_KEY=your_api_key_here
```

## How to run

### Step 1: Ingest a PDF

Recommended:

```bash
python main.py ingest --pdf "path/to/your-file.pdf" --vector-db both
```

Vector DB options:

```bash
python main.py ingest --pdf "path/to/your-file.pdf" --vector-db faiss
python main.py ingest --pdf "path/to/your-file.pdf" --vector-db chroma
python main.py ingest --pdf "path/to/your-file.pdf" --vector-db both
```

This creates/updates:
- `rag_store/index.faiss` (FAISS index)
- `rag_store/chunks.json` (chunk text for FAISS)
- `rag_store/chroma/` (ChromaDB persistent store)

### Step 2: Ask a question

Recommended:

```bash
python main.py ask "What is this PDF about?" --vector-db both
```

Examples:

```bash
python main.py ask "What is this PDF about?" --k 5 --vector-db faiss
python main.py ask "What is this PDF about?" --k 5 --vector-db chroma
python main.py ask "What is this PDF about?" --k 5 --vector-db both
```

- `--k` controls top matching chunks.
- `--vector-db` selects retrieval source (`faiss`, `chroma`, `both`).

## How flow works internally

1. `ingest` command
- Reads all PDF pages.
- Cleans text (`_clean_text`).
- Splits text into overlapping chunks (`_chunk_text`, size 1200, overlap 200).
- Generates embeddings for each chunk (`_embed`).
- Stores vectors in selected DB(s): FAISS, ChromaDB, or both.

2. `ask` command
- Embeds your question once.
- Retrieves top-k chunks from selected DB(s).
- If `both`, merges and de-duplicates results.
- Builds context prompt from retrieved chunks.
- Sends prompt to Gemini chat model.
- Prints final answer in terminal.

## Project structure

```text
.
|-- main.py
|-- pyproject.toml
|-- uv.lock
`-- rag_store/
    |-- chunks.json
    |-- index.faiss
    `-- chroma/
```

## Common errors

1. `Please set the GEMINI_API_KEY environment variable.`
- Set env var before running commands.

2. `RAG store not found. Run ingest first.`
- Run `ingest` before `ask`.

3. `No text could be extracted from this PDF.`
- PDF may be scanned/image-only or empty text layer.
