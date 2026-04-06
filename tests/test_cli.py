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
    output = result.stdout + (result.stderr or "")
    assert "API key" in output or "api key" in output.lower()


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
