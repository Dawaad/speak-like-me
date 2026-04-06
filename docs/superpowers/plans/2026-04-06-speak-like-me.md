# Speak Like Me — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool (`slm`) that ingests your writing samples, stores them as vector embeddings in ChromaDB, and uses retrieved style excerpts to rewrite AI-generated text in your personal voice via the Claude API.

**Architecture:** Single Python package with 7 focused modules. Local ONNX-based embeddings (`all-MiniLM-L6-v2`) for ingestion and retrieval. ChromaDB (embedded, SQLite-backed) for vector storage. Claude Sonnet 4 API for the rewriting step with streaming output. CLI built with Typer.

**Tech Stack:** Python 3.14, Typer (CLI), ChromaDB (vector DB), ONNX Runtime + sentence-transformers (embeddings), Anthropic SDK (rewriting), Hypothesis (property-based testing), pytest, tqdm (progress bars)

---

## File Structure

```
speak-like-me/
├── src/slm/
│   ├── __init__.py          # Package version
│   ├── cli.py               # Typer CLI: ingest, rewrite, search, stats commands
│   ├── config.py            # Load TOML config + env var overrides
│   ├── exceptions.py        # Typed exception hierarchy
│   ├── ingest.py            # File reading, sentence splitting, windowed chunking
│   ├── embedder.py          # ONNX Runtime embedding model wrapper
│   ├── store.py             # ChromaDB interface: add, query, stats
│   ├── retriever.py         # Query embedding + ChromaDB search + ranking
│   └── rewriter.py          # Claude API prompt construction + streaming call
├── tests/
│   ├── conftest.py          # Shared fixtures (tmp ChromaDB, sample texts, embedder)
│   ├── test_config.py       # Config loading tests
│   ├── test_exceptions.py   # Exception hierarchy tests
│   ├── test_ingest.py       # Chunking: property-based + unit tests
│   ├── test_embedder.py     # Embedding model tests
│   ├── test_store.py        # ChromaDB integration tests
│   ├── test_retriever.py    # Retrieval pipeline integration tests
│   ├── test_rewriter.py     # VCR-based rewriter tests
│   ├── test_cli.py          # E2E CLI tests
│   ├── golden/              # Golden file fixtures for retrieval regression
│   │   └── retrieval_golden.json
│   └── live/                # Manual-only live API tests
│       └── test_rewrite_live.py
├── pyproject.toml           # Project metadata, dependencies, scripts
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/slm/__init__.py`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "speak-like-me"
version = "0.1.0"
description = "Rewrite AI-generated text to match your personal writing style"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.15.0",
    "chromadb>=1.0.0",
    "onnxruntime>=1.21.0",
    "tokenizers>=0.21.0",
    "anthropic>=0.52.0",
    "tomli>=2.2.0; python_version < '3.11'",
    "tqdm>=4.67.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-recording>=0.13.0",
    "vcrpy>=7.0.0",
    "hypothesis>=6.122.0",
]

[project.scripts]
slm = "slm.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "live: tests that call real external APIs (deselect with '-m not live')",
]
```

- [ ] **Step 2: Create src/slm/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
*.onnx
chromadb_data/
.env
tests/cassettes/
```

- [ ] **Step 4: Create tests/__init__.py and tests/conftest.py**

`tests/__init__.py` — empty file.

`tests/conftest.py`:

```python
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_text():
    return (
        "I've always thought the best code reads like a conversation. "
        "You don't need comments when the names tell the story. "
        "That said, sometimes you gotta leave a note for future-you. "
        "Because future-you is basically a stranger who happens to have your SSH keys. "
        "And that stranger will mass-delete your clever abstractions without remorse."
    )


@pytest.fixture
def sample_paragraphs():
    return [
        (
            "The thing about distributed systems is they distribute your problems too. "
            "You trade one big failure for a thousand small ones. "
            "And each small failure has its own personality."
        ),
        (
            "I rewrote the parser three times before I realized the grammar was ambiguous. "
            "Not my parser. The actual grammar. "
            "Sometimes the spec is the bug."
        ),
        (
            "Debugging is just reverse engineering your own past decisions. "
            "Except past-you left no documentation. "
            "And the git blame points at a squash commit with the message 'stuff'."
        ),
    ]
```

- [ ] **Step 5: Create venv and install in dev mode**

```bash
cd /home/jared/dev/speak-like-me
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 6: Verify pytest runs (no tests yet, should pass with 0 collected)**

```bash
.venv/bin/pytest -v
```

Expected: `no tests ran` with exit code 5 (no tests collected) or 0.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/slm/__init__.py .gitignore tests/__init__.py tests/conftest.py
git commit -m "feat: scaffold project with pyproject.toml, package structure, and test fixtures"
```

---

## Task 2: Exceptions Module

**Files:**
- Create: `src/slm/exceptions.py`
- Create: `tests/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

`tests/test_exceptions.py`:

```python
from slm.exceptions import (
    SLMError,
    ConfigError,
    IngestError,
    EmbeddingError,
    StoreError,
    RetrievalError,
    RewriteError,
)


def test_all_exceptions_inherit_from_slm_error():
    for exc_class in [ConfigError, IngestError, EmbeddingError, StoreError, RetrievalError, RewriteError]:
        err = exc_class("test message")
        assert isinstance(err, SLMError)
        assert isinstance(err, Exception)
        assert str(err) == "test message"


def test_slm_error_is_base():
    err = SLMError("base error")
    assert str(err) == "base error"


def test_exceptions_are_distinct():
    classes = [ConfigError, IngestError, EmbeddingError, StoreError, RetrievalError, RewriteError]
    for i, cls_a in enumerate(classes):
        for cls_b in classes[i + 1:]:
            assert not issubclass(cls_a, cls_b)
            assert not issubclass(cls_b, cls_a)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_exceptions.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.exceptions'`

- [ ] **Step 3: Write the implementation**

`src/slm/exceptions.py`:

```python
class SLMError(Exception):
    """Base exception for all speak-like-me errors."""


class ConfigError(SLMError):
    """Configuration loading or validation failed."""


class IngestError(SLMError):
    """File reading, parsing, or chunking failed."""


class EmbeddingError(SLMError):
    """Embedding model loading or inference failed."""


class StoreError(SLMError):
    """ChromaDB connection, insert, or query failed."""


class RetrievalError(SLMError):
    """Retrieval pipeline failed (embedding + store query)."""


class RewriteError(SLMError):
    """LLM API call for rewriting failed."""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_exceptions.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/slm/exceptions.py tests/test_exceptions.py
git commit -m "feat: add typed exception hierarchy"
```

---

## Task 3: Config Module

**Files:**
- Create: `src/slm/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
import os
import pytest
from pathlib import Path
from slm.config import SLMConfig, load_config
from slm.exceptions import ConfigError


def test_default_config():
    config = SLMConfig()
    assert config.chroma_dir == Path.home() / ".local" / "share" / "slm" / "chromadb"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.chunk_sentences == 4
    assert config.chunk_stride == 2
    assert config.retrieval_count == 8
    assert config.rewrite_model == "claude-sonnet-4-20250514"
    assert config.api_key is None


def test_config_from_toml(tmp_dir):
    config_path = tmp_dir / "config.toml"
    config_path.write_text(
        '[slm]\n'
        'chunk_sentences = 6\n'
        'chunk_stride = 3\n'
        'retrieval_count = 12\n'
        f'chroma_dir = "{tmp_dir / "mydb"}"\n'
    )
    config = load_config(config_path)
    assert config.chunk_sentences == 6
    assert config.chunk_stride == 3
    assert config.retrieval_count == 12
    assert config.chroma_dir == tmp_dir / "mydb"
    # Defaults preserved for unset values
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_api_key_from_env(monkeypatch, tmp_dir):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-123")
    config_path = tmp_dir / "config.toml"
    config_path.write_text("[slm]\n")
    config = load_config(config_path)
    assert config.api_key == "sk-test-key-123"


def test_api_key_env_overrides_nothing_in_file(monkeypatch, tmp_dir):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    config = SLMConfig()
    config.api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert config.api_key == "sk-from-env"


def test_load_config_missing_file_uses_defaults():
    config = load_config(Path("/nonexistent/path/config.toml"))
    assert config.chunk_sentences == 4


def test_load_config_invalid_toml(tmp_dir):
    bad_file = tmp_dir / "config.toml"
    bad_file.write_text("this is not valid toml [[[")
    with pytest.raises(ConfigError, match="Failed to parse"):
        load_config(bad_file)


def test_chunk_stride_must_be_less_than_chunk_sentences():
    with pytest.raises(ConfigError, match="stride"):
        SLMConfig(chunk_sentences=3, chunk_stride=5).validate()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.config'`

- [ ] **Step 3: Write the implementation**

`src/slm/config.py`:

```python
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from slm.exceptions import ConfigError


@dataclass
class SLMConfig:
    chroma_dir: Path = field(
        default_factory=lambda: Path.home() / ".local" / "share" / "slm" / "chromadb"
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_sentences: int = 4
    chunk_stride: int = 2
    retrieval_count: int = 8
    rewrite_model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None

    def validate(self) -> "SLMConfig":
        if self.chunk_stride >= self.chunk_sentences:
            raise ConfigError(
                f"chunk_stride ({self.chunk_stride}) must be less than "
                f"chunk_sentences ({self.chunk_sentences})"
            )
        return self


def load_config(config_path: Path | None = None) -> SLMConfig:
    if config_path is None:
        config_path = Path.home() / ".config" / "slm" / "config.toml"

    raw: dict = {}
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                raw = tomllib.load(f).get("slm", {})
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"Failed to parse config at {config_path}: {e}") from e

    # Convert chroma_dir string to Path if present
    if "chroma_dir" in raw:
        raw["chroma_dir"] = Path(raw["chroma_dir"])

    config = SLMConfig(**raw)
    config.api_key = os.environ.get("ANTHROPIC_API_KEY")
    config.validate()
    return config
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_config.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/slm/config.py tests/test_config.py
git commit -m "feat: add config module with TOML loading and env var overrides"
```

---

## Task 4: Ingestion / Chunking Module

**Files:**
- Create: `src/slm/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests (property-based + unit)**

`tests/test_ingest.py`:

```python
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pathlib import Path

from slm.ingest import split_sentences, chunk_sentences, read_and_chunk
from slm.exceptions import IngestError


# --- Sentence splitting ---

def test_split_simple_sentences():
    text = "First sentence. Second sentence. Third sentence."
    result = split_sentences(text)
    assert result == ["First sentence.", "Second sentence.", "Third sentence."]


def test_split_handles_abbreviations():
    text = "Dr. Smith went to Washington. He arrived at 3 p.m. today."
    result = split_sentences(text)
    # Should not split on Dr. or p.m.
    assert len(result) == 2


def test_split_handles_question_marks_and_exclamations():
    text = "What happened? I don't know! But it was wild."
    result = split_sentences(text)
    assert len(result) == 3


def test_split_empty_string():
    assert split_sentences("") == []


def test_split_whitespace_only():
    assert split_sentences("   \n\t  ") == []


def test_split_single_sentence_no_period():
    result = split_sentences("No period here")
    assert result == ["No period here"]


# --- Windowed chunking ---

def test_chunk_basic_window():
    sentences = ["A.", "B.", "C.", "D.", "E."]
    chunks = chunk_sentences(sentences, window=3, stride=2)
    assert chunks == [
        "A. B. C.",
        "C. D. E.",
    ]


def test_chunk_window_larger_than_input():
    sentences = ["A.", "B."]
    chunks = chunk_sentences(sentences, window=5, stride=2)
    assert chunks == ["A. B."]


def test_chunk_stride_one():
    sentences = ["A.", "B.", "C.", "D."]
    chunks = chunk_sentences(sentences, window=2, stride=1)
    assert chunks == [
        "A. B.",
        "B. C.",
        "C. D.",
    ]


def test_chunk_single_sentence():
    sentences = ["Only one."]
    chunks = chunk_sentences(sentences, window=3, stride=2)
    assert chunks == ["Only one."]


def test_chunk_empty_list():
    assert chunk_sentences([], window=3, stride=2) == []


# --- Property-based tests ---

@given(
    sentences=st.lists(
        st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        min_size=1,
        max_size=100,
    ),
    window=st.integers(min_value=1, max_value=10),
    stride=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=200)
def test_chunking_covers_all_sentences(sentences, window, stride):
    assume(stride <= window)
    chunks = chunk_sentences(sentences, window=window, stride=stride)
    # Every sentence must appear in at least one chunk
    all_chunk_text = " ".join(chunks)
    for sentence in sentences:
        assert sentence in all_chunk_text


@given(
    sentences=st.lists(
        st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
        min_size=0,
        max_size=50,
    ),
)
@settings(max_examples=100)
def test_chunking_never_returns_empty_chunks(sentences):
    chunks = chunk_sentences(sentences, window=4, stride=2)
    for chunk in chunks:
        assert chunk.strip() != ""


# --- File reading ---

def test_read_and_chunk_txt(tmp_dir):
    f = tmp_dir / "sample.txt"
    f.write_text("First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.")
    chunks = read_and_chunk(f, window=3, stride=2)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)


def test_read_and_chunk_missing_file(tmp_dir):
    with pytest.raises(IngestError, match="not found"):
        read_and_chunk(tmp_dir / "nope.txt", window=3, stride=2)


def test_read_and_chunk_empty_file(tmp_dir):
    f = tmp_dir / "empty.txt"
    f.write_text("")
    chunks = read_and_chunk(f, window=3, stride=2)
    assert chunks == []


def test_read_and_chunk_markdown(tmp_dir):
    f = tmp_dir / "doc.md"
    f.write_text("# Header\n\nThis is a paragraph. It has two sentences.\n\n## Another\n\nMore text here.")
    chunks = read_and_chunk(f, window=2, stride=1)
    assert len(chunks) >= 1
    # Headers should be stripped
    for chunk in chunks:
        assert "#" not in chunk
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_ingest.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.ingest'`

- [ ] **Step 3: Write the implementation**

`src/slm/ingest.py`:

```python
import re
from pathlib import Path

from slm.exceptions import IngestError

# Abbreviations that shouldn't trigger sentence splits
_ABBREVIATIONS = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Jr|Sr|Inc|Ltd|Corp|vs|etc|approx|dept|est|govt|i\.e|e\.g|a\.m|p\.m)\.",
    re.IGNORECASE,
)

# Sentence-ending punctuation followed by space or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])$')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    text = text.strip()
    if not text:
        return []

    # Replace abbreviation periods with a placeholder
    placeholder = "\x00"
    protected = _ABBREVIATIONS.sub(
        lambda m: m.group(0).replace(".", placeholder), text
    )

    # Split on sentence boundaries
    parts = _SENTENCE_END.split(protected)

    # Restore periods and clean up
    sentences = []
    for part in parts:
        restored = part.replace(placeholder, ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def chunk_sentences(
    sentences: list[str], *, window: int, stride: int
) -> list[str]:
    """Create overlapping chunks from a list of sentences.

    Args:
        sentences: List of sentence strings.
        window: Number of sentences per chunk.
        stride: Number of sentences to advance between chunks.

    Returns:
        List of chunk strings, each containing `window` sentences
        (or fewer for the final chunk).
    """
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i : i + window]
        chunks.append(" ".join(chunk))
        i += stride
        # Stop if this chunk already reached the end
        if i + window > len(sentences) and i < len(sentences):
            # One more chunk to capture the tail
            tail = sentences[i:]
            if tail:
                tail_text = " ".join(tail)
                # Avoid duplicate if tail equals last chunk
                if tail_text != chunks[-1]:
                    chunks.append(tail_text)
            break

    return chunks


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting, keeping plain text."""
    # Remove headers
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}|_{1,3}", "", text)
    # Remove links, keep text: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def read_and_chunk(
    path: Path, *, window: int, stride: int
) -> list[str]:
    """Read a file and return windowed sentence chunks.

    Supports .txt and .md files.

    Raises:
        IngestError: If the file doesn't exist or can't be read.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise IngestError(f"Failed to read {path}: {e}") from e

    if path.suffix.lower() in (".md", ".markdown"):
        text = _strip_markdown(text)

    sentences = split_sentences(text)
    return chunk_sentences(sentences, window=window, stride=stride)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_ingest.py -v
```

Expected: All tests PASSED (including property-based tests).

- [ ] **Step 5: Commit**

```bash
git add src/slm/ingest.py tests/test_ingest.py
git commit -m "feat: add ingestion module with sentence splitting and windowed chunking"
```

---

## Task 5: Embedder Module

**Files:**
- Create: `src/slm/embedder.py`
- Create: `tests/test_embedder.py`

**Note:** This task requires downloading the ONNX model. The embedder uses `tokenizers` (Hugging Face) for tokenization and ONNX Runtime for inference. The model file will be downloaded on first use and cached locally.

- [ ] **Step 1: Write the failing tests**

`tests/test_embedder.py`:

```python
import pytest
import numpy as np
from pathlib import Path

from slm.embedder import Embedder
from slm.exceptions import EmbeddingError


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance — model loading is slow, share across tests."""
    return Embedder()


def test_embed_single_text(embedder):
    vec = embedder.embed("Hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)  # MiniLM-L6-v2 output dimension
    assert vec.dtype == np.float32


def test_embed_batch(embedder):
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    vecs = embedder.embed_batch(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, 384)


def test_embed_empty_string(embedder):
    vec = embedder.embed("")
    assert vec.shape == (384,)


def test_embed_batch_empty_list(embedder):
    vecs = embedder.embed_batch([])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (0, 384)


def test_similar_texts_have_higher_cosine_similarity(embedder):
    v1 = embedder.embed("The cat sat on the mat.")
    v2 = embedder.embed("A cat was sitting on a rug.")
    v3 = embedder.embed("Quantum mechanics explains particle behavior.")

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_related = cosine_sim(v1, v2)
    sim_unrelated = cosine_sim(v1, v3)
    assert sim_related > sim_unrelated


def test_embeddings_are_normalized(embedder):
    vec = embedder.embed("Test normalization.")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_embedder.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.embedder'`

- [ ] **Step 3: Write the implementation**

`src/slm/embedder.py`:

```python
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer

from slm.exceptions import EmbeddingError

_MODEL_DIR = Path.home() / ".local" / "share" / "slm" / "models"
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


def _ensure_model() -> tuple[Path, Path]:
    """Download model files if not cached. Returns (onnx_path, tokenizer_path)."""
    model_dir = _MODEL_DIR / _MODEL_NAME
    onnx_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"

    if onnx_path.exists() and tokenizer_path.exists():
        return onnx_path, tokenizer_path

    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from urllib.request import urlretrieve

        base_url = f"https://huggingface.co/sentence-transformers/{_MODEL_NAME}/resolve/main"

        if not onnx_path.exists():
            urlretrieve(f"{base_url}/onnx/model.onnx", onnx_path)

        if not tokenizer_path.exists():
            urlretrieve(f"{base_url}/tokenizer.json", tokenizer_path)

    except Exception as e:
        raise EmbeddingError(
            f"Failed to download model '{_MODEL_NAME}': {e}. "
            f"Ensure you have internet access for the first run."
        ) from e

    return onnx_path, tokenizer_path


def _mean_pool(
    token_embeddings: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Mean pooling with attention mask."""
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each vector."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return vectors / norms


class Embedder:
    """ONNX-based sentence embedding model.

    Loads the model on initialization and provides embed/embed_batch methods.
    """

    def __init__(self, model_name: str = _MODEL_NAME):
        try:
            onnx_path, tokenizer_path = _ensure_model()
            self._session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_padding()
            self._tokenizer.enable_truncation(max_length=512)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a 1-D float32 array of shape (384,)."""
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns float32 array of shape (n, 384)."""
        if not texts:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        try:
            encodings = self._tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
            token_type_ids = np.zeros_like(input_ids)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
            pooled = _mean_pool(token_embeddings, attention_mask)
            return _normalize(pooled).astype(np.float32)

        except Exception as e:
            raise EmbeddingError(f"Embedding inference failed: {e}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_embedder.py -v
```

Expected: All 6 tests PASSED. First run will download the model (~80MB).

- [ ] **Step 5: Commit**

```bash
git add src/slm/embedder.py tests/test_embedder.py
git commit -m "feat: add ONNX-based sentence embedder with auto-download"
```

---

## Task 6: ChromaDB Store Module

**Files:**
- Create: `src/slm/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:

```python
import pytest
import numpy as np
from pathlib import Path

from slm.store import VectorStore
from slm.exceptions import StoreError


@pytest.fixture
def store(tmp_dir):
    return VectorStore(persist_dir=tmp_dir / "chromadb")


@pytest.fixture
def sample_vectors():
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 384)).astype(np.float32)


def test_add_and_count(store, sample_vectors):
    texts = [f"Sentence {i}" for i in range(5)]
    sources = ["test.txt"] * 5
    store.add(texts=texts, embeddings=sample_vectors, sources=sources)
    assert store.count() == 5


def test_add_duplicate_texts_are_idempotent(store, sample_vectors):
    texts = ["Same sentence."] * 3
    sources = ["a.txt"] * 3
    # ChromaDB should deduplicate by ID (which we derive from text hash)
    store.add(texts=texts, embeddings=sample_vectors[:3], sources=sources)
    assert store.count() == 1


def test_query_returns_results(store, sample_vectors):
    texts = ["Alpha sentence.", "Beta sentence.", "Gamma sentence."]
    sources = ["test.txt"] * 3
    store.add(texts=texts, embeddings=sample_vectors[:3], sources=sources)

    results = store.query(query_embedding=sample_vectors[0], n_results=2)
    assert len(results) == 2
    assert results[0]["text"] == "Alpha sentence."
    assert "distance" in results[0]
    assert "source" in results[0]


def test_query_empty_store(store, sample_vectors):
    results = store.query(query_embedding=sample_vectors[0], n_results=5)
    assert results == []


def test_query_n_results_exceeds_store_size(store, sample_vectors):
    texts = ["Only one."]
    sources = ["test.txt"]
    store.add(texts=texts, embeddings=sample_vectors[:1], sources=sources)
    results = store.query(query_embedding=sample_vectors[0], n_results=10)
    assert len(results) == 1


def test_stats(store, sample_vectors):
    texts = ["A.", "B.", "C."]
    sources = ["file1.txt", "file1.txt", "file2.txt"]
    store.add(texts=texts, embeddings=sample_vectors[:3], sources=sources)
    stats = store.stats()
    assert stats["total_chunks"] == 3
    assert stats["unique_sources"] == 2
    assert set(stats["sources"]) == {"file1.txt", "file2.txt"}


def test_persistence(tmp_dir, sample_vectors):
    db_path = tmp_dir / "persist_test"
    texts = ["Persistent data."]
    sources = ["test.txt"]

    store1 = VectorStore(persist_dir=db_path)
    store1.add(texts=texts, embeddings=sample_vectors[:1], sources=sources)
    assert store1.count() == 1
    del store1

    store2 = VectorStore(persist_dir=db_path)
    assert store2.count() == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_store.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.store'`

- [ ] **Step 3: Write the implementation**

`src/slm/store.py`:

```python
import hashlib
import numpy as np
import chromadb
from pathlib import Path

from slm.exceptions import StoreError

_COLLECTION_NAME = "style_chunks"


class VectorStore:
    """ChromaDB-backed vector store for style chunks."""

    def __init__(self, persist_dir: Path):
        try:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise StoreError(f"Failed to initialize ChromaDB at {persist_dir}: {e}") from e

    def _text_id(self, text: str) -> str:
        """Generate a deterministic ID from text content for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        sources: list[str],
    ) -> int:
        """Add text chunks with their embeddings to the store.

        Deduplicates by text content hash. Returns number of new chunks added.
        """
        if not texts:
            return 0

        try:
            ids = [self._text_id(t) for t in texts]
            # Deduplicate within the batch
            seen: dict[str, int] = {}
            unique_indices = []
            for i, id_ in enumerate(ids):
                if id_ not in seen:
                    seen[id_] = i
                    unique_indices.append(i)

            unique_ids = [ids[i] for i in unique_indices]
            unique_texts = [texts[i] for i in unique_indices]
            unique_embeddings = embeddings[unique_indices].tolist()
            unique_sources = [sources[i] for i in unique_indices]

            self._collection.upsert(
                ids=unique_ids,
                documents=unique_texts,
                embeddings=unique_embeddings,
                metadatas=[{"source": s} for s in unique_sources],
            )
            return len(unique_ids)

        except Exception as e:
            raise StoreError(f"Failed to add chunks: {e}") from e

    def query(
        self, query_embedding: np.ndarray, n_results: int = 8
    ) -> list[dict]:
        """Query for similar chunks. Returns list of dicts with text, distance, source."""
        try:
            count = self._collection.count()
            if count == 0:
                return []

            actual_n = min(n_results, count)
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=actual_n,
            )

            items = []
            for i in range(len(results["ids"][0])):
                items.append(
                    {
                        "text": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "source": results["metadatas"][0][i]["source"],
                    }
                )
            return items

        except Exception as e:
            raise StoreError(f"Query failed: {e}") from e

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self._collection.count()

    def stats(self) -> dict:
        """Return store statistics."""
        try:
            all_metadata = self._collection.get(include=["metadatas"])
            sources = [m["source"] for m in all_metadata["metadatas"]]
            return {
                "total_chunks": self._collection.count(),
                "unique_sources": len(set(sources)),
                "sources": sorted(set(sources)),
            }
        except Exception as e:
            raise StoreError(f"Failed to get stats: {e}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_store.py -v
```

Expected: All 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/slm/store.py tests/test_store.py
git commit -m "feat: add ChromaDB vector store with deduplication and stats"
```

---

## Task 7: Retriever Module

**Files:**
- Create: `src/slm/retriever.py`
- Create: `tests/test_retriever.py`
- Create: `tests/golden/retrieval_golden.json`

- [ ] **Step 1: Write the failing tests**

`tests/test_retriever.py`:

```python
import json
import pytest
import numpy as np
from pathlib import Path

from slm.retriever import Retriever
from slm.embedder import Embedder
from slm.store import VectorStore
from slm.exceptions import RetrievalError


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def populated_store(tmp_dir, embedder, sample_paragraphs):
    store = VectorStore(persist_dir=tmp_dir / "chromadb")
    for i, para in enumerate(sample_paragraphs):
        vec = embedder.embed(para)
        store.add(
            texts=[para],
            embeddings=vec.reshape(1, -1),
            sources=[f"sample_{i}.txt"],
        )
    return store


@pytest.fixture
def retriever(embedder, populated_store):
    return Retriever(embedder=embedder, store=populated_store)


def test_retrieve_returns_results(retriever):
    results = retriever.retrieve("debugging is painful", n_results=2)
    assert len(results) == 2
    assert all("text" in r for r in results)
    assert all("distance" in r for r in results)
    assert all("source" in r for r in results)


def test_retrieve_most_relevant_first(retriever):
    results = retriever.retrieve("debugging and git blame", n_results=3)
    # The debugging paragraph should be most relevant
    assert "Debugging" in results[0]["text"] or "git blame" in results[0]["text"]


def test_retrieve_empty_query(retriever):
    results = retriever.retrieve("", n_results=3)
    # Should still return results (empty string gets embedded)
    assert len(results) == 3


def test_retrieve_n_exceeds_store(retriever):
    results = retriever.retrieve("test", n_results=100)
    assert len(results) == 3  # Only 3 paragraphs in store


def test_retrieve_formats_context(retriever):
    context = retriever.retrieve_as_context("debugging problems", n_results=2)
    assert isinstance(context, str)
    assert "---" in context  # Separator between excerpts
    assert len(context) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_retriever.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.retriever'`

- [ ] **Step 3: Write the implementation**

`src/slm/retriever.py`:

```python
from slm.embedder import Embedder
from slm.store import VectorStore
from slm.exceptions import RetrievalError


class Retriever:
    """Retrieves style-similar chunks for a given query text."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, n_results: int = 8) -> list[dict]:
        """Embed query and retrieve the most similar chunks.

        Returns list of dicts with keys: text, distance, source.
        Sorted by relevance (closest first).
        """
        try:
            query_vec = self._embedder.embed(query)
            return self._store.query(query_embedding=query_vec, n_results=n_results)
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}") from e

    def retrieve_as_context(self, query: str, n_results: int = 8) -> str:
        """Retrieve chunks and format them as a context string for the rewriter.

        Each excerpt is separated by '---' and prefixed with its source.
        """
        results = self.retrieve(query, n_results=n_results)
        if not results:
            return ""

        parts = []
        for r in results:
            parts.append(f"[Source: {r['source']}]\n{r['text']}")

        return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_retriever.py -v
```

Expected: All 5 tests PASSED.

- [ ] **Step 5: Create golden file for retrieval regression testing**

`tests/golden/retrieval_golden.json`:

```json
{
  "description": "Golden file for retrieval regression testing. Update when embedding model changes.",
  "model": "all-MiniLM-L6-v2",
  "queries": [
    {
      "query": "debugging and git blame",
      "expected_top_contains": "git blame"
    },
    {
      "query": "distributed systems failures",
      "expected_top_contains": "distributed systems"
    },
    {
      "query": "parser and grammar",
      "expected_top_contains": "parser"
    }
  ]
}
```

- [ ] **Step 6: Add golden file regression test to test_retriever.py**

Append to `tests/test_retriever.py`:

```python
def test_retrieval_golden_file(retriever):
    golden_path = Path(__file__).parent / "golden" / "retrieval_golden.json"
    with open(golden_path) as f:
        golden = json.load(f)

    for case in golden["queries"]:
        results = retriever.retrieve(case["query"], n_results=1)
        assert len(results) >= 1
        assert case["expected_top_contains"].lower() in results[0]["text"].lower(), (
            f"Query '{case['query']}' expected top result to contain "
            f"'{case['expected_top_contains']}', got: {results[0]['text'][:100]}"
        )
```

- [ ] **Step 7: Run all retriever tests**

```bash
.venv/bin/pytest tests/test_retriever.py -v
```

Expected: All 6 tests PASSED.

- [ ] **Step 8: Commit**

```bash
git add src/slm/retriever.py tests/test_retriever.py tests/golden/retrieval_golden.json
git commit -m "feat: add retriever with context formatting and golden-file regression tests"
```

---

## Task 8: Rewriter Module

**Files:**
- Create: `src/slm/rewriter.py`
- Create: `tests/test_rewriter.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_rewriter.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from slm.rewriter import Rewriter, build_rewrite_prompt
from slm.exceptions import RewriteError


def test_build_rewrite_prompt():
    style_context = "[Source: blog.txt]\nI've always thought...\n\n---\n\n[Source: notes.txt]\nThe thing about..."
    input_text = "Distributed systems are complex architectures."

    system, user = build_rewrite_prompt(style_context, input_text)

    assert "writing style" in system.lower() or "voice" in system.lower()
    assert style_context in user
    assert input_text in user


def test_build_rewrite_prompt_empty_context():
    system, user = build_rewrite_prompt("", "Some text to rewrite.")
    assert "Some text to rewrite." in user
    # Should still work, just without style examples


def test_rewriter_init_without_api_key():
    with pytest.raises(RewriteError, match="API key"):
        Rewriter(api_key=None, model="claude-sonnet-4-20250514")


def test_rewriter_rewrite_calls_api():
    with patch("slm.rewriter.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock streaming response
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Rewritten ", "text ", "here."])
        mock_client.messages.stream.return_value = mock_stream

        rewriter = Rewriter(api_key="sk-test", model="claude-sonnet-4-20250514")
        chunks = list(rewriter.rewrite(
            style_context="Some style examples.",
            input_text="Text to rewrite.",
        ))

        assert chunks == ["Rewritten ", "text ", "here."]
        mock_client.messages.stream.assert_called_once()
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 4096


def test_rewriter_rewrite_full_text():
    with patch("slm.rewriter.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Full ", "response."])
        mock_client.messages.stream.return_value = mock_stream

        rewriter = Rewriter(api_key="sk-test", model="claude-sonnet-4-20250514")
        result = rewriter.rewrite_full(
            style_context="Style.",
            input_text="Input.",
        )

        assert result == "Full response."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_rewriter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.rewriter'`

- [ ] **Step 3: Write the implementation**

`src/slm/rewriter.py`:

```python
from collections.abc import Iterator

import anthropic

from slm.exceptions import RewriteError

_SYSTEM_PROMPT = """\
You are a writing style adapter. Your job is to rewrite text so it matches \
the voice, rhythm, vocabulary, and sentence patterns shown in the style \
examples below. Preserve the original meaning and information content \
exactly — change only how it's expressed.

Rules:
- Match sentence length patterns from the examples
- Use the same level of formality, contractions, and colloquialisms
- Mirror transition patterns between ideas
- Keep the same tone (casual, technical, sardonic, etc.)
- Do NOT add information not in the original text
- Do NOT remove information from the original text
- Output ONLY the rewritten text — no commentary, no preamble"""

_USER_TEMPLATE = """\
## Style Examples

These are excerpts of the target writing style. Match this voice:

{style_context}

## Text to Rewrite

Rewrite the following text to match the style above:

{input_text}"""


def build_rewrite_prompt(
    style_context: str, input_text: str
) -> tuple[str, str]:
    """Build the system and user prompts for the rewrite call.

    Returns:
        (system_prompt, user_message)
    """
    user_msg = _USER_TEMPLATE.format(
        style_context=style_context or "(No style examples available.)",
        input_text=input_text,
    )
    return _SYSTEM_PROMPT, user_msg


class Rewriter:
    """Rewrites text using the Claude API to match a target writing style."""

    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise RewriteError(
                "API key required for rewriting. "
                "Set ANTHROPIC_API_KEY environment variable."
            )
        self._model = model
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise RewriteError(f"Failed to initialize Anthropic client: {e}") from e

    def rewrite(
        self, style_context: str, input_text: str
    ) -> Iterator[str]:
        """Rewrite text with streaming output. Yields text chunks."""
        system, user = build_rewrite_prompt(style_context, input_text)

        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                yield from stream.text_stream
        except anthropic.APIError as e:
            raise RewriteError(f"Claude API error: {e}") from e
        except Exception as e:
            raise RewriteError(f"Rewrite failed: {e}") from e

    def rewrite_full(
        self, style_context: str, input_text: str
    ) -> str:
        """Rewrite text and return the complete result as a string."""
        return "".join(self.rewrite(style_context, input_text))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_rewriter.py -v
```

Expected: All 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/slm/rewriter.py tests/test_rewriter.py
git commit -m "feat: add Claude API rewriter with streaming and prompt construction"
```

---

## Task 9: CLI Module

**Files:**
- Create: `src/slm/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
import pytest
import os
from pathlib import Path
from typer.testing import CliRunner
from slm.cli import app


runner = CliRunner()


@pytest.fixture
def config_dir(tmp_dir, monkeypatch):
    """Set up a temp config and data dir."""
    config_path = tmp_dir / "config.toml"
    data_dir = tmp_dir / "data"
    config_path.write_text(
        f'[slm]\nchroma_dir = "{data_dir}"\n'
    )
    monkeypatch.setenv("SLM_CONFIG", str(config_path))
    return tmp_dir


@pytest.fixture
def sample_file(tmp_dir):
    f = tmp_dir / "sample.txt"
    f.write_text(
        "I've always thought the best code reads like a conversation. "
        "You don't need comments when the names tell the story. "
        "That said, sometimes you gotta leave a note for future-you. "
        "Because future-you is basically a stranger who happens to have your SSH keys."
    )
    return f


def test_ingest_command(config_dir, sample_file):
    result = runner.invoke(app, ["ingest", str(sample_file), "--config", str(config_dir / "config.toml")])
    assert result.exit_code == 0
    assert "Ingested" in result.stdout or "chunk" in result.stdout.lower()


def test_ingest_missing_file(config_dir):
    result = runner.invoke(app, ["ingest", "/nonexistent/file.txt", "--config", str(config_dir / "config.toml")])
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_stats_command_empty(config_dir):
    result = runner.invoke(app, ["stats", "--config", str(config_dir / "config.toml")])
    assert result.exit_code == 0
    assert "0" in result.stdout


def test_stats_after_ingest(config_dir, sample_file):
    runner.invoke(app, ["ingest", str(sample_file), "--config", str(config_dir / "config.toml")])
    result = runner.invoke(app, ["stats", "--config", str(config_dir / "config.toml")])
    assert result.exit_code == 0
    assert "chunk" in result.stdout.lower() or "source" in result.stdout.lower()


def test_search_command(config_dir, sample_file):
    runner.invoke(app, ["ingest", str(sample_file), "--config", str(config_dir / "config.toml")])
    result = runner.invoke(app, ["search", "code and comments", "--config", str(config_dir / "config.toml")])
    assert result.exit_code == 0
    assert len(result.stdout.strip()) > 0


def test_rewrite_without_api_key(config_dir, sample_file, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner.invoke(app, ["ingest", str(sample_file), "--config", str(config_dir / "config.toml")])
    result = runner.invoke(app, ["rewrite", "Test text.", "--config", str(config_dir / "config.toml")])
    assert result.exit_code != 0
    assert "API key" in result.stdout or "api key" in result.stdout.lower()


def test_ingest_directory(config_dir, tmp_dir):
    subdir = tmp_dir / "docs"
    subdir.mkdir()
    (subdir / "a.txt").write_text("First document sentence one. Sentence two here.")
    (subdir / "b.txt").write_text("Second document sentence one. And another sentence.")
    result = runner.invoke(app, ["ingest", str(subdir), "--config", str(config_dir / "config.toml")])
    assert result.exit_code == 0


def test_version_command():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'slm.cli'`

- [ ] **Step 3: Write the implementation**

`src/slm/cli.py`:

```python
import sys
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from slm import __version__
from slm.config import load_config, SLMConfig
from slm.exceptions import SLMError
from slm.ingest import read_and_chunk
from slm.embedder import Embedder
from slm.store import VectorStore
from slm.retriever import Retriever
from slm.rewriter import Rewriter

app = typer.Typer(
    name="slm",
    help="Speak Like Me — rewrite AI text in your personal voice.",
    no_args_is_help=True,
)


def _version_callback(value: bool):
    if value:
        typer.echo(f"slm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=_version_callback, is_eager=True,
        help="Show version and exit."
    ),
):
    pass


def _load_config(config_path: Optional[Path]) -> SLMConfig:
    try:
        return load_config(config_path)
    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Ingest your writing samples into the style corpus."""
    cfg = _load_config(config)

    # Collect files
    if path.is_dir():
        files = sorted(
            f for f in path.rglob("*")
            if f.suffix.lower() in (".txt", ".md", ".markdown")
        )
        if not files:
            typer.echo(f"No .txt or .md files found in {path}")
            raise typer.Exit(code=1)
    elif path.is_file():
        files = [path]
    else:
        typer.echo(f"Error: not found: {path}")
        raise typer.Exit(code=1)

    try:
        embedder = Embedder()
        store = VectorStore(persist_dir=cfg.chroma_dir)

        total_chunks = 0
        for f in tqdm(files, desc="Ingesting", disable=len(files) == 1):
            chunks = read_and_chunk(f, window=cfg.chunk_sentences, stride=cfg.chunk_stride)
            if not chunks:
                continue
            embeddings = embedder.embed_batch(chunks)
            sources = [str(f.name)] * len(chunks)
            store.add(texts=chunks, embeddings=embeddings, sources=sources)
            total_chunks += len(chunks)

        typer.echo(f"Ingested {total_chunks} chunks from {len(files)} file(s).")

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Text to search for similar style excerpts."),
    n: int = typer.Option(5, "--n", "-n", help="Number of results."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Search your corpus for style-similar excerpts."""
    cfg = _load_config(config)

    try:
        embedder = Embedder()
        store = VectorStore(persist_dir=cfg.chroma_dir)
        retriever = Retriever(embedder=embedder, store=store)

        results = retriever.retrieve(query, n_results=n)
        if not results:
            typer.echo("No results. Ingest some writing samples first.")
            raise typer.Exit(code=0)

        for i, r in enumerate(results, 1):
            typer.echo(f"\n--- Result {i} (distance: {r['distance']:.4f}, source: {r['source']}) ---")
            typer.echo(r["text"])

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def rewrite(
    text: str = typer.Argument(..., help="Text to rewrite in your style."),
    n: int = typer.Option(8, "--n", "-n", help="Number of style excerpts to retrieve."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Rewrite AI-generated text to match your writing style."""
    cfg = _load_config(config)

    try:
        embedder = Embedder()
        store = VectorStore(persist_dir=cfg.chroma_dir)
        retriever = Retriever(embedder=embedder, store=store)

        style_context = retriever.retrieve_as_context(text, n_results=n)
        if not style_context:
            typer.echo("Warning: no style excerpts found. Ingest writing samples first.", err=True)

        rewriter = Rewriter(api_key=cfg.api_key, model=cfg.rewrite_model)

        for chunk in rewriter.rewrite(style_context=style_context, input_text=text):
            sys.stdout.write(chunk)
            sys.stdout.flush()

        typer.echo()  # Final newline

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def stats(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Show corpus statistics."""
    cfg = _load_config(config)

    try:
        store = VectorStore(persist_dir=cfg.chroma_dir)
        s = store.stats()

        typer.echo(f"Total chunks: {s['total_chunks']}")
        typer.echo(f"Unique sources: {s['unique_sources']}")
        if s["sources"]:
            typer.echo("Sources:")
            for src in s["sources"]:
                typer.echo(f"  - {src}")

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: All 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/slm/cli.py tests/test_cli.py
git commit -m "feat: add Typer CLI with ingest, search, rewrite, and stats commands"
```

---

## Task 10: Live API Tests (Manual Only)

**Files:**
- Create: `tests/live/__init__.py`
- Create: `tests/live/test_rewrite_live.py`

- [ ] **Step 1: Create the live test file**

`tests/live/__init__.py` — empty file.

`tests/live/test_rewrite_live.py`:

```python
import os
import pytest
from slm.rewriter import Rewriter

pytestmark = pytest.mark.live


@pytest.fixture
def rewriter():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return Rewriter(api_key=api_key, model="claude-sonnet-4-20250514")


STYLE_CONTEXT = """\
[Source: blog.txt]
I've always thought the best code reads like a conversation. You don't need \
comments when the names tell the story. That said, sometimes you gotta leave \
a note for future-you. Because future-you is basically a stranger who happens \
to have your SSH keys.

---

[Source: notes.txt]
The thing about distributed systems is they distribute your problems too. \
You trade one big failure for a thousand small ones. And each small failure \
has its own personality."""


def test_rewrite_streaming(rewriter):
    chunks = list(rewriter.rewrite(
        style_context=STYLE_CONTEXT,
        input_text="Machine learning models require careful hyperparameter tuning to achieve optimal performance on validation datasets.",
    ))
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 20
    # Should be rewritten, not identical to input
    assert full_text != "Machine learning models require careful hyperparameter tuning to achieve optimal performance on validation datasets."


def test_rewrite_full(rewriter):
    result = rewriter.rewrite_full(
        style_context=STYLE_CONTEXT,
        input_text="Containerization provides isolation between application dependencies and enables reproducible deployments across environments.",
    )
    assert len(result) > 20
    assert isinstance(result, str)
```

- [ ] **Step 2: Verify live tests are skipped in normal runs**

```bash
.venv/bin/pytest tests/ -v -m "not live"
```

Expected: Live tests are deselected.

- [ ] **Step 3: (Manual) Run live tests when you want to validate**

```bash
ANTHROPIC_API_KEY=your-key .venv/bin/pytest tests/live/ -v -m live
```

- [ ] **Step 4: Commit**

```bash
git add tests/live/__init__.py tests/live/test_rewrite_live.py
git commit -m "feat: add manual-only live API tests for rewriter"
```

---

## Task 11: Full Test Suite Run and Final Verification

- [ ] **Step 1: Run the complete test suite (excluding live tests)**

```bash
.venv/bin/pytest tests/ -v -m "not live" --tb=short
```

Expected: All tests PASSED (approximately 35-40 tests).

- [ ] **Step 2: Verify the CLI is installed and works**

```bash
.venv/bin/slm --version
.venv/bin/slm --help
.venv/bin/slm stats
```

Expected:
- Version prints `slm 0.1.0`
- Help shows all 4 commands
- Stats shows `Total chunks: 0`

- [ ] **Step 3: Run a quick smoke test**

```bash
echo "I always thought debugging was like archaeology. You dig through layers of decisions made by people who are no longer around. And the artifacts rarely make sense out of context." > /tmp/slm_test.txt
.venv/bin/slm ingest /tmp/slm_test.txt
.venv/bin/slm stats
.venv/bin/slm search "debugging old code"
```

Expected: Ingest succeeds, stats shows 1+ chunks, search returns relevant results.

- [ ] **Step 4: Commit any final fixes**

If any tests needed fixes, commit them:

```bash
git add -A
git commit -m "fix: resolve issues found during final verification"
```
