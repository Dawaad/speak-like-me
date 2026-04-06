import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pathlib import Path

from slm.ingest import split_sentences, chunk_sentences, read_and_chunk
from slm.exceptions import IngestError


# --- Sentence splitting ---

def test_split_simple_sentences():
    text = "First sentence. Second sentence. Third sentence."
    result = split_sentences(text)
    assert result == ["First sentence.", "Second sentence.", "Third sentence."]


def test_split_handles_abbreviations():
    text = "Dr. Smith went to Washington. He arrived at 3 p.m. today."
    result = split_sentences(text)
    # Should not split on Dr. or p.m.
    assert len(result) == 2


def test_split_handles_question_marks_and_exclamations():
    text = "What happened? I don't know! But it was wild."
    result = split_sentences(text)
    assert len(result) == 3


def test_split_empty_string():
    assert split_sentences("") == []


def test_split_whitespace_only():
    assert split_sentences("   \n\t  ") == []


def test_split_single_sentence_no_period():
    result = split_sentences("No period here")
    assert result == ["No period here"]


# --- Windowed chunking ---

def test_chunk_basic_window():
    sentences = ["A.", "B.", "C.", "D.", "E."]
    chunks = chunk_sentences(sentences, window=3, stride=2)
    assert chunks == [
        "A. B. C.",
        "C. D. E.",
    ]


def test_chunk_window_larger_than_input():
    sentences = ["A.", "B."]
    chunks = chunk_sentences(sentences, window=5, stride=2)
    assert chunks == ["A. B."]


def test_chunk_stride_one():
    sentences = ["A.", "B.", "C.", "D."]
    chunks = chunk_sentences(sentences, window=2, stride=1)
    assert chunks == [
        "A. B.",
        "B. C.",
        "C. D.",
    ]


def test_chunk_single_sentence():
    sentences = ["Only one."]
    chunks = chunk_sentences(sentences, window=3, stride=2)
    assert chunks == ["Only one."]


def test_chunk_empty_list():
    assert chunk_sentences([], window=3, stride=2) == []


# --- Property-based tests ---

@given(
    sentences=st.lists(
        st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        min_size=1,
        max_size=100,
    ),
    window=st.integers(min_value=1, max_value=10),
    stride=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=200)
def test_chunking_covers_all_sentences(sentences, window, stride):
    assume(stride <= window)
    chunks = chunk_sentences(sentences, window=window, stride=stride)
    # Every sentence must appear in at least one chunk
    all_chunk_text = " ".join(chunks)
    for sentence in sentences:
        assert sentence in all_chunk_text


@given(
    sentences=st.lists(
        st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
        min_size=0,
        max_size=50,
    ),
)
@settings(max_examples=100)
def test_chunking_never_returns_empty_chunks(sentences):
    chunks = chunk_sentences(sentences, window=4, stride=2)
    for chunk in chunks:
        assert chunk.strip() != ""


# --- File reading ---

def test_read_and_chunk_txt(tmp_dir):
    f = tmp_dir / "sample.txt"
    f.write_text("First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.")
    chunks = read_and_chunk(f, window=3, stride=2)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)


def test_read_and_chunk_missing_file(tmp_dir):
    with pytest.raises(IngestError, match="not found"):
        read_and_chunk(tmp_dir / "nope.txt", window=3, stride=2)


def test_read_and_chunk_empty_file(tmp_dir):
    f = tmp_dir / "empty.txt"
    f.write_text("")
    chunks = read_and_chunk(f, window=3, stride=2)
    assert chunks == []


def test_read_and_chunk_markdown(tmp_dir):
    f = tmp_dir / "doc.md"
    f.write_text("# Header\n\nThis is a paragraph. It has two sentences.\n\n## Another\n\nMore text here.")
    chunks = read_and_chunk(f, window=2, stride=1)
    assert len(chunks) >= 1
    # Headers should be stripped
    for chunk in chunks:
        assert "#" not in chunk
