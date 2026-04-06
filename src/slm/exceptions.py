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
