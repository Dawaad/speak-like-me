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
