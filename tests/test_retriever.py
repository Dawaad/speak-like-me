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
