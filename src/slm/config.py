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
