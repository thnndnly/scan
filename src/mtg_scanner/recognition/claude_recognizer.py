"""Claude Vision card recogniser using Anthropic API."""

from __future__ import annotations

import logging
import os
from typing import Optional

from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.recognition.base import BaseRecognizer
from mtg_scanner.utils.fuzzy_search import best_match, load_card_names
from mtg_scanner.utils.image_utils import encode_image_base64

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are an expert Magic: The Gathering card identifier. "
    "Look at this card image and respond with ONLY the exact English card name — nothing else. "
    "If the card name is in another language, translate it to English. "
    "If you cannot identify the card, respond with exactly: UNKNOWN"
)


class ClaudeRecognizer(BaseRecognizer):
    """Card recogniser that sends the card image to Claude for identification.

    Uses the Anthropic API (claude-sonnet-4-6 by default). The ``anthropic``
    package is imported lazily. The ``ANTHROPIC_API_KEY`` environment variable
    must be set at runtime.

    Args:
        model: Claude model ID (default: ``claude-sonnet-4-6``).
        names_file: Path to the card-names JSON for post-hoc fuzzy verification.
        fuzzy_cutoff: Minimum fuzzy-match score (0–100) to accept a result.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        names_file: str = "data/card_names.json",
        fuzzy_cutoff: float = 80.0,
    ) -> None:
        try:
            from mtg_scanner.config import get_config
            cfg = get_config()
            self._model = model or cfg.claude.model
        except Exception:
            self._model = model
        self._names_file = names_file
        self._fuzzy_cutoff = fuzzy_cutoff
        self._client = None
        self._card_names: list[str] = []

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for Claude recognition.\n"
                "Install it with:  pip install 'mtg-card-scanner[claude]'\n"
                "or:               pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please export your Anthropic API key before using the Claude recogniser."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _get_card_names(self) -> list[str]:
        if not self._card_names:
            self._card_names = load_card_names(self._names_file)
        return self._card_names

    def _call_claude(self, image_b64: str) -> Optional[str]:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=64,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": _PROMPT},
                        ],
                    }
                ],
            )
            content = response.content[0].text.strip()
            if content.upper() == "UNKNOWN" or not content:
                return None
            return content
        except Exception as exc:
            logger.error("Claude API request failed: %s", exc)
            return None

    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Identify the card in *patch* by sending its image to Claude.

        Args:
            patch: Detected card patch.

        Returns:
            ``(card_name, confidence)`` or ``(None, 0.0)`` on failure.
        """
        try:
            image_b64 = encode_image_base64(patch.image, fmt=".png")
        except Exception as exc:
            logger.warning("Failed to encode patch image: %s", exc)
            return None, 0.0

        try:
            raw_name = self._call_claude(image_b64)
        except (ImportError, EnvironmentError):
            raise
        except Exception as exc:
            logger.error("Claude recognition failed: %s", exc)
            return None, 0.0

        if raw_name is None:
            logger.debug("Claude returned no card name for patch %d", patch.patch_index)
            return None, 0.0

        logger.debug("Claude raw response: %r", raw_name)

        names = self._get_card_names()
        if names:
            card_name, confidence = best_match(
                raw_name, names, score_cutoff=self._fuzzy_cutoff, names_file=self._names_file
            )
        else:
            card_name, confidence = raw_name, 0.85

        if card_name is None:
            logger.info("Claude: %r not found in known card list", raw_name)
            return None, 0.0

        logger.info(
            "Claude: identified %r → %r (confidence=%.2f)", raw_name, card_name, confidence
        )
        return card_name, float(confidence)
