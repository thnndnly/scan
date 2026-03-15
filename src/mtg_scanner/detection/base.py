"""Abstract base class for card detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mtg_scanner.models.card_patch import CardPatch


class BaseDetector(ABC):
    """Abstract interface for MTG card detection.

    Subclasses must implement :meth:`detect` which accepts a path to an image
    file and returns a list of :class:`~mtg_scanner.models.card_patch.CardPatch`
    objects representing each detected card.
    """

    @abstractmethod
    def detect(self, image_path: str) -> list[CardPatch]:
        """Detect all MTG cards in *image_path*.

        Args:
            image_path: Filesystem path to the input image.

        Returns:
            List of :class:`~mtg_scanner.models.card_patch.CardPatch` objects,
            one per detected card, ordered from top-left to bottom-right.
            Returns an empty list if no cards are found.
        """
        ...
