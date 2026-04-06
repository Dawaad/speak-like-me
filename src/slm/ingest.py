import re
from pathlib import Path

from slm.exceptions import IngestError

# Abbreviations that shouldn't trigger sentence splits
_ABBREVIATIONS = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Jr|Sr|Inc|Ltd|Corp|vs|etc|approx|dept|est|govt|i\.e|e\.g|a\.m|p\.m)\.",
    re.IGNORECASE,
)

# Sentence-ending punctuation followed by space or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])$')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    text = text.strip()
    if not text:
        return []

    # Replace abbreviation periods with a placeholder
    placeholder = "\x00"
    protected = _ABBREVIATIONS.sub(
        lambda m: m.group(0).replace(".", placeholder), text
    )

    # Split on sentence boundaries
    parts = _SENTENCE_END.split(protected)

    # Restore periods and clean up
    sentences = []
    for part in parts:
        restored = part.replace(placeholder, ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def chunk_sentences(
    sentences: list[str], *, window: int, stride: int
) -> list[str]:
    """Create overlapping chunks from a list of sentences.

    Args:
        sentences: List of sentence strings.
        window: Number of sentences per chunk.
        stride: Number of sentences to advance between chunks.

    Returns:
        List of chunk strings, each containing `window` sentences
        (or fewer for the final chunk).
    """
    if not sentences:
        return []

    chunks = []
    i = 0
    seen_end = False
    while i < len(sentences):
        chunk = sentences[i : i + window]
        chunk_text = " ".join(chunk)
        chunks.append(chunk_text)
        if i + window >= len(sentences):
            seen_end = True
            break
        i += stride

    # If the last window didn't reach the end, add a final tail chunk
    if not seen_end and i < len(sentences):
        tail = sentences[i:]
        if tail:
            tail_text = " ".join(tail)
            if tail_text != chunks[-1]:
                chunks.append(tail_text)

    return chunks


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting, keeping plain text."""
    # Remove headers
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}|_{1,3}", "", text)
    # Remove links, keep text: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def read_and_chunk(
    path: Path, *, window: int, stride: int
) -> list[str]:
    """Read a file and return windowed sentence chunks.

    Supports .txt and .md files.

    Raises:
        IngestError: If the file doesn't exist or can't be read.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise IngestError(f"Failed to read {path}: {e}") from e

    if path.suffix.lower() in (".md", ".markdown"):
        text = _strip_markdown(text)

    sentences = split_sentences(text)
    return chunk_sentences(sentences, window=window, stride=stride)
