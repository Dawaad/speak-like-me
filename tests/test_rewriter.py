import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from slm.rewriter import Rewriter, build_rewrite_prompt
from slm.exceptions import RewriteError


def test_build_rewrite_prompt():
    style_context = "[Source: blog.txt]\nI've always thought...\n\n---\n\n[Source: notes.txt]\nThe thing about..."
    input_text = "Distributed systems are complex architectures."

    system, user = build_rewrite_prompt(style_context, input_text)

    assert "writing style" in system.lower() or "voice" in system.lower()
    assert style_context in user
    assert input_text in user


def test_build_rewrite_prompt_empty_context():
    system, user = build_rewrite_prompt("", "Some text to rewrite.")
    assert "Some text to rewrite." in user
    # Should still work, just without style examples


def test_rewriter_init_without_api_key():
    with pytest.raises(RewriteError, match="API key"):
        Rewriter(api_key=None, model="claude-sonnet-4-20250514")


def test_rewriter_rewrite_calls_api():
    with patch("slm.rewriter.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock streaming response
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Rewritten ", "text ", "here."])
        mock_client.messages.stream.return_value = mock_stream

        rewriter = Rewriter(api_key="sk-test", model="claude-sonnet-4-20250514")
        chunks = list(rewriter.rewrite(
            style_context="Some style examples.",
            input_text="Text to rewrite.",
        ))

        assert chunks == ["Rewritten ", "text ", "here."]
        mock_client.messages.stream.assert_called_once()
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 4096


def test_rewriter_rewrite_full_text():
    with patch("slm.rewriter.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Full ", "response."])
        mock_client.messages.stream.return_value = mock_stream

        rewriter = Rewriter(api_key="sk-test", model="claude-sonnet-4-20250514")
        result = rewriter.rewrite_full(
            style_context="Style.",
            input_text="Input.",
        )

        assert result == "Full response."
