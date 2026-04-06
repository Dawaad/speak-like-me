import os
import pytest
from slm.rewriter import Rewriter

pytestmark = pytest.mark.live


@pytest.fixture
def rewriter():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return Rewriter(api_key=api_key, model="claude-sonnet-4-20250514")


STYLE_CONTEXT = """\
[Source: blog.txt]
I've always thought the best code reads like a conversation. You don't need \
comments when the names tell the story. That said, sometimes you gotta leave \
a note for future-you. Because future-you is basically a stranger who happens \
to have your SSH keys.

---

[Source: notes.txt]
The thing about distributed systems is they distribute your problems too. \
You trade one big failure for a thousand small ones. And each small failure \
has its own personality."""


def test_rewrite_streaming(rewriter):
    chunks = list(rewriter.rewrite(
        style_context=STYLE_CONTEXT,
        input_text="Machine learning models require careful hyperparameter tuning to achieve optimal performance on validation datasets.",
    ))
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 20
    # Should be rewritten, not identical to input
    assert full_text != "Machine learning models require careful hyperparameter tuning to achieve optimal performance on validation datasets."


def test_rewrite_full(rewriter):
    result = rewriter.rewrite_full(
        style_context=STYLE_CONTEXT,
        input_text="Containerization provides isolation between application dependencies and enables reproducible deployments across environments.",
    )
    assert len(result) > 20
    assert isinstance(result, str)
