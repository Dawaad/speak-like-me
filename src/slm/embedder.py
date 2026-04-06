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
