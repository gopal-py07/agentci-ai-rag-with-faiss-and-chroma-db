# agenic-ai

Simple PDF RAG (Retrieval-Augmented Generation) CLI app.

## What this project does

This project lets you:
1. Ingest a PDF file into a local vector store.
2. Ask questions about that PDF.
3. Get answers generated from retrieved PDF chunks.

It uses:
- `pypdf` for reading PDF text
- `openai` client with Google Gemini OpenAI-compatible endpoint
- `faiss` for local vector search

## Prerequisites

1. Python `3.14+` (as defined in `pyproject.toml`)
2. A valid Gemini API key

## Setup

### 1. Clone repo

```bash
git clone https://github.com/gopal-py07/agenic-ai.git
cd agenic-ai
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

```bash
python main.py ingest --pdf "path/to/your-file.pdf"
```

This creates/updates:
- `rag_store/index.faiss` (vector index)
- `rag_store/chunks.json` (text chunks)

### Step 2: Ask a question

```bash
python main.py ask "What is this PDF about?"
```

Optional:

```bash
python main.py ask "What is this PDF about?" --k 5
```

`--k` controls how many top matching chunks are used as context.

## How flow works internally

1. `ingest` command
- Reads all PDF pages.
- Cleans text (`_clean_text`).
- Splits text into overlapping chunks (`_chunk_text`, size 1200, overlap 200).
- Generates embeddings for each chunk (`_embed`).
- Normalizes vectors and stores them in FAISS index.
- Saves chunks + index to `rag_store/`.

2. `ask` command
- Loads FAISS index and chunk list from `rag_store/`.
- Embeds your question.
- Finds top-k similar chunks using vector search.
- Builds a context prompt from those chunks.
- Sends prompt to Gemini chat model.
- Prints final answer in terminal.

## Project structure

```text
.
├── main.py
├── pyproject.toml
├── uv.lock
└── rag_store/
    ├── chunks.json
    └── index.faiss
```

## Common errors

1. `Please set the GEMINI_API_KEY environment variable.`
- Set env var before running commands.

2. `RAG store not found. Run ingest first.`
- Run `ingest` before `ask`.

3. `No text could be extracted from this PDF.`
- PDF may be scanned/image-only or empty text layer.
