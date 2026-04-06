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
