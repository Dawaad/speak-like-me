from collections.abc import Iterator

import anthropic

from slm.exceptions import RewriteError

_SYSTEM_PROMPT = """\
You are a writing style adapter. Your job is to rewrite text so it matches \
the voice, rhythm, vocabulary, and sentence patterns shown in the style \
examples below. Preserve the original meaning and information content \
exactly — change only how it's expressed.

Rules:
- Match sentence length patterns from the examples
- Use the same level of formality, contractions, and colloquialisms
- Mirror transition patterns between ideas
- Keep the same tone (casual, technical, sardonic, etc.)
- Do NOT add information not in the original text
- Do NOT remove information from the original text
- Output ONLY the rewritten text — no commentary, no preamble"""

_USER_TEMPLATE = """\
## Style Examples

These are excerpts of the target writing style. Match this voice:

{style_context}

## Text to Rewrite

Rewrite the following text to match the style above:

{input_text}"""


def build_rewrite_prompt(
    style_context: str, input_text: str
) -> tuple[str, str]:
    """Build the system and user prompts for the rewrite call.

    Returns:
        (system_prompt, user_message)
    """
    user_msg = _USER_TEMPLATE.format(
        style_context=style_context or "(No style examples available.)",
        input_text=input_text,
    )
    return _SYSTEM_PROMPT, user_msg


class Rewriter:
    """Rewrites text using the Claude API to match a target writing style."""

    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise RewriteError(
                "API key required for rewriting. "
                "Set ANTHROPIC_API_KEY environment variable."
            )
        self._model = model
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise RewriteError(f"Failed to initialize Anthropic client: {e}") from e

    def rewrite(
        self, style_context: str, input_text: str
    ) -> Iterator[str]:
        """Rewrite text with streaming output. Yields text chunks."""
        system, user = build_rewrite_prompt(style_context, input_text)

        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                yield from stream.text_stream
        except anthropic.APIError as e:
            raise RewriteError(f"Claude API error: {e}") from e
        except Exception as e:
            raise RewriteError(f"Rewrite failed: {e}") from e

    def rewrite_full(
        self, style_context: str, input_text: str
    ) -> str:
        """Rewrite text and return the complete result as a string."""
        return "".join(self.rewrite(style_context, input_text))
