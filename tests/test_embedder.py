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
