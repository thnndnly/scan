"""Abstract base class for card recognizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from mtg_scanner.models.card_patch import CardPatch


class BaseRecognizer(ABC):
    """Abstract interface for MTG card recognition.

    A recognizer receives a detected :class:`~mtg_scanner.models.card_patch.CardPatch`
    and attempts to identify the card name, returning a confidence score.
    """

    @abstractmethod
    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Attempt to identify the card in *patch*.

        Args:
            patch: A detected and perspective-corrected card image patch.

        Returns:
            A ``(card_name, confidence)`` tuple.  *card_name* is ``None`` when
            recognition fails.  *confidence* is in ``[0.0, 1.0]``.
        """
        ...
