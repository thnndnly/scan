"""LLM-based card recogniser using GPT-4o vision."""

from __future__ import annotations

import logging
import os
from typing import Optional

from mtg_scanner.config import get_config
from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.recognition.base import BaseRecognizer
from mtg_scanner.utils.fuzzy_search import best_match, load_card_names
from mtg_scanner.utils.image_utils import encode_image_base64

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Magic: The Gathering card identifier. "
    "When shown a card image, respond with ONLY the exact card name and nothing else. "
    "If you cannot identify the card, respond with exactly: UNKNOWN"
)

_USER_PROMPT = (
    "What is the exact name of this Magic: The Gathering card? "
    "Respond with only the card name."
)


class LLMRecognizer(BaseRecognizer):
    """Card recogniser that sends the card image to GPT-4o for identification.

    The ``openai`` package is imported lazily so that the rest of the project
    works without it.  The ``OPENAI_API_KEY`` environment variable must be set
    at runtime.

    Args:
        model: OpenAI model identifier (default: ``gpt-4o``).
        names_file: Path to the card-names JSON used for post-hoc fuzzy
            verification.
        fuzzy_cutoff: Minimum fuzzy-match score (0-100) used to verify the
            model's output against the known card list.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        names_file: str = "data/card_names.json",
        fuzzy_cutoff: float = 80.0,
    ) -> None:
        self._model = model
        self._names_file = names_file
        self._fuzzy_cutoff = fuzzy_cutoff
        self._client = None  # lazy
        self._card_names: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily create the OpenAI client.

        Raises:
            ImportError: When the ``openai`` package is not installed.
            EnvironmentError: When ``OPENAI_API_KEY`` is not set.
        """
        if self._client is not None:
            return self._client

        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLM recognition.\n"
                "Install it with:  pip install 'mtg-card-scanner[llm]'\n"
                "or:               pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.  "
                "Please export your OpenAI API key before using the LLM recogniser."
            )

        self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def _get_card_names(self) -> list[str]:
        if not self._card_names:
            self._card_names = load_card_names(self._names_file)
        return self._card_names

    def _call_gpt4o(self, image_b64: str) -> Optional[str]:
        """Send the encoded image to GPT-4o and parse the response.

        Args:
            image_b64: Base64-encoded PNG string.

        Returns:
            Card name string returned by the model, or ``None`` on error.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": _USER_PROMPT},
                        ],
                    },
                ],
                max_tokens=60,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content.upper() == "UNKNOWN" or not content:
                return None
            return content
        except Exception as exc:
            logger.error("GPT-4o request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # BaseRecognizer interface
    # ------------------------------------------------------------------

    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Identify the card in *patch* by sending its image to GPT-4o.

        The raw model output is verified against the known-names list via
        fuzzy matching.  If the output doesn't match any known card name above
        *fuzzy_cutoff*, the result is discarded.

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
            raw_name = self._call_gpt4o(image_b64)
        except (ImportError, EnvironmentError):
            raise
        except Exception as exc:
            logger.error("LLM recognition failed: %s", exc)
            return None, 0.0

        if raw_name is None:
            logger.debug("LLM returned no card name for patch %d", patch.patch_index)
            return None, 0.0

        logger.debug("LLM raw response: %r", raw_name)

        # Verify via fuzzy search
        names = self._get_card_names()
        if names:
            card_name, confidence = best_match(
                raw_name,
                names,
                score_cutoff=self._fuzzy_cutoff,
                names_file=self._names_file,
            )
        else:
            # No names file available; trust the model at 0.8 confidence
            card_name, confidence = raw_name, 0.80

        if card_name is None:
            logger.info("LLM: %r not found in known card list", raw_name)
            return None, 0.0

        logger.info(
            "LLM: identified %r → %r (confidence=%.2f)", raw_name, card_name, confidence
        )
        return card_name, float(confidence)
