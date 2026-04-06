# Speak Like Me

A CLI tool that rewrites AI-generated text to match your personal writing style. Feed it your blog posts, notes, or any writing samples, and it learns how you write. Then give it bland AI output and it rewrites it in your voice.

## How it works

The pipeline has three stages: **ingest**, **retrieve**, **rewrite**.

### 1. Ingest

```
slm ingest ~/writing/blog-posts/
```

Your writing samples go through a processing pipeline:

- **Sentence splitting** (`ingest.py`) — Regex-based splitter that handles abbreviations (Dr., p.m., etc.) without false splits. Supports `.txt` and `.md` files (markdown formatting is stripped).
- **Windowed chunking** — Sentences are grouped into overlapping windows (default: 4 sentences, stride 2). This preserves local context while keeping chunks small enough for meaningful similarity search.
- **Embedding** (`embedder.py`) — Each chunk is embedded using `all-MiniLM-L6-v2` via ONNX Runtime. No PyTorch, no GPU required. The model downloads automatically on first run (~80MB) and caches to `~/.local/share/slm/models/`. Embeddings are L2-normalized 384-dimensional vectors.
- **Storage** (`store.py`) — Embeddings and text are stored in ChromaDB (embedded mode, SQLite-backed) with cosine similarity indexing. Chunks are deduplicated by content hash. Persists to `~/.local/share/slm/chromadb/` by default.

### 2. Retrieve

```
slm search "debugging old code"
```

When you search or rewrite, the input text is embedded with the same model, then ChromaDB returns the most similar chunks from your corpus. The retriever (`retriever.py`) formats these as a context block with source attribution.

### 3. Rewrite

```
slm rewrite "Distributed systems require careful consideration of network partitions and consensus algorithms."
```

The retrieved style excerpts and the input text are assembled into a prompt for Claude Sonnet 4 via the Anthropic API. The system prompt instructs the model to match your voice, rhythm, vocabulary, and sentence patterns while preserving the original meaning exactly. Output streams to stdout in real time.

## Architecture

```
src/slm/
  cli.py          Typer CLI — ingest, search, rewrite, stats commands
  config.py       TOML config (~/.config/slm/config.toml) + env var overrides
  exceptions.py   Typed exception hierarchy (SLMError base)
  ingest.py       Sentence splitting, windowed chunking, file reading
  embedder.py     ONNX Runtime wrapper for all-MiniLM-L6-v2
  store.py        ChromaDB interface — add, query, stats, deduplication
  retriever.py    Embedding + vector search + context formatting
  rewriter.py     Claude API prompt construction + streaming response
```

All modules propagate errors through typed exceptions that the CLI catches and renders as user-facing messages.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Requires Python 3.12+. The ONNX embedding model downloads automatically on first use.

For the rewrite command, set your API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

## Configuration

Optional config at `~/.config/slm/config.toml`:

```toml
[slm]
chunk_sentences = 4       # sentences per chunk
chunk_stride = 2          # overlap stride
retrieval_count = 8       # style excerpts for rewriting
chroma_dir = "~/.local/share/slm/chromadb"
rewrite_model = "claude-sonnet-4-20250514"
```

## Testing

```bash
# Run all tests (excludes live API tests)
.venv/bin/pytest -m "not live"

# Run live API tests (requires ANTHROPIC_API_KEY)
.venv/bin/pytest -m live
```

59 tests covering: unit tests, property-based tests (Hypothesis), ChromaDB integration tests, golden-file retrieval regression tests, mock-based API tests, and CLI end-to-end tests.

## Design constraints

Built to run on a NUC (i7-7500U, 8GB RAM, no GPU). This drives two key decisions:

- **Local embeddings via ONNX Runtime** instead of PyTorch — ~200MB peak RAM, no GPU needed.
- **Claude API for rewriting** instead of a local LLM — the NUC can't run inference on anything that produces quality rewrites. API cost is ~$0.004 per rewrite.

Everything except the rewrite step works fully offline.
