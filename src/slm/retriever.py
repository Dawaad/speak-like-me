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
