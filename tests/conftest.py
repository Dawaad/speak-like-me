import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_text():
    return (
        "I've always thought the best code reads like a conversation. "
        "You don't need comments when the names tell the story. "
        "That said, sometimes you gotta leave a note for future-you. "
        "Because future-you is basically a stranger who happens to have your SSH keys. "
        "And that stranger will mass-delete your clever abstractions without remorse."
    )


@pytest.fixture
def sample_paragraphs():
    return [
        (
            "The thing about distributed systems is they distribute your problems too. "
            "You trade one big failure for a thousand small ones. "
            "And each small failure has its own personality."
        ),
        (
            "I rewrote the parser three times before I realized the grammar was ambiguous. "
            "Not my parser. The actual grammar. "
            "Sometimes the spec is the bug."
        ),
        (
            "Debugging is just reverse engineering your own past decisions. "
            "Except past-you left no documentation. "
            "And the git blame points at a squash commit with the message 'stuff'."
        ),
    ]
