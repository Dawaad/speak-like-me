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
