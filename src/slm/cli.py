import sys
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from slm import __version__
from slm.config import load_config, SLMConfig
from slm.exceptions import SLMError
from slm.ingest import read_and_chunk
from slm.embedder import Embedder
from slm.store import VectorStore
from slm.retriever import Retriever
from slm.rewriter import Rewriter

app = typer.Typer(
    name="slm",
    help="Speak Like Me — rewrite AI text in your personal voice.",
    no_args_is_help=True,
)


def _version_callback(value: bool):
    if value:
        typer.echo(f"slm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=_version_callback, is_eager=True,
        help="Show version and exit."
    ),
):
    pass


def _load_config(config_path: Optional[Path]) -> SLMConfig:
    try:
        return load_config(config_path)
    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Ingest your writing samples into the style corpus."""
    cfg = _load_config(config)

    # Collect files
    if path.is_dir():
        files = sorted(
            f for f in path.rglob("*")
            if f.suffix.lower() in (".txt", ".md", ".markdown")
        )
        if not files:
            typer.echo(f"No .txt or .md files found in {path}")
            raise typer.Exit(code=1)
    elif path.is_file():
        files = [path]
    else:
        typer.echo(f"Error: not found: {path}")
        raise typer.Exit(code=1)

    try:
        embedder = Embedder()
        store = VectorStore(persist_dir=cfg.chroma_dir)

        total_chunks = 0
        for f in tqdm(files, desc="Ingesting", disable=len(files) == 1):
            chunks = read_and_chunk(f, window=cfg.chunk_sentences, stride=cfg.chunk_stride)
            if not chunks:
                continue
            embeddings = embedder.embed_batch(chunks)
            sources = [str(f.name)] * len(chunks)
            store.add(texts=chunks, embeddings=embeddings, sources=sources)
            total_chunks += len(chunks)

        typer.echo(f"Ingested {total_chunks} chunks from {len(files)} file(s).")

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Text to search for similar style excerpts."),
    n: int = typer.Option(5, "--n", "-n", help="Number of results."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Search your corpus for style-similar excerpts."""
    cfg = _load_config(config)

    try:
        embedder = Embedder()
        store = VectorStore(persist_dir=cfg.chroma_dir)
        retriever = Retriever(embedder=embedder, store=store)

        results = retriever.retrieve(query, n_results=n)
        if not results:
            typer.echo("No results. Ingest some writing samples first.")
            raise typer.Exit(code=0)

        for i, r in enumerate(results, 1):
            typer.echo(f"\n--- Result {i} (distance: {r['distance']:.4f}, source: {r['source']}) ---")
            typer.echo(r["text"])

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def rewrite(
    text: str = typer.Argument(..., help="Text to rewrite in your style."),
    n: int = typer.Option(8, "--n", "-n", help="Number of style excerpts to retrieve."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Rewrite AI-generated text to match your writing style."""
    cfg = _load_config(config)

    try:
        embedder = Embedder()
        store = VectorStore(persist_dir=cfg.chroma_dir)
        retriever = Retriever(embedder=embedder, store=store)

        style_context = retriever.retrieve_as_context(text, n_results=n)
        if not style_context:
            typer.echo("Warning: no style excerpts found. Ingest writing samples first.", err=True)

        rewriter = Rewriter(api_key=cfg.api_key, model=cfg.rewrite_model)

        for chunk in rewriter.rewrite(style_context=style_context, input_text=text):
            sys.stdout.write(chunk)
            sys.stdout.flush()

        typer.echo()  # Final newline

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def stats(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path."),
):
    """Show corpus statistics."""
    cfg = _load_config(config)

    try:
        store = VectorStore(persist_dir=cfg.chroma_dir)
        s = store.stats()

        typer.echo(f"Total chunks: {s['total_chunks']}")
        typer.echo(f"Unique sources: {s['unique_sources']}")
        if s["sources"]:
            typer.echo("Sources:")
            for src in s["sources"]:
                typer.echo(f"  - {src}")

    except SLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
