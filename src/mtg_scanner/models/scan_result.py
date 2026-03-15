from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mtg_scanner.models.recognized_card import RecognizedCard


@dataclass
class ScanResult:
    """The complete result of scanning a single image for MTG cards.

    Attributes:
        image_path: Path to the original scanned image.
        scan_timestamp: ISO 8601 timestamp of when the scan was performed.
        cards: List of all recognised (and unrecognised) cards found.
        total_detected: How many card patches were detected.
        total_recognized: How many patches were successfully named.
        total_unknown: How many patches could not be identified.
    """

    image_path: str
    scan_timestamp: str  # ISO 8601
    cards: list[RecognizedCard] = field(default_factory=list)
    total_detected: int = 0
    total_recognized: int = 0
    total_unknown: int = 0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_csv_rows(self) -> list[dict]:
        """Return a list of flat dicts suitable for csv.DictWriter.

        Each dict represents one recognised card with its patch metadata,
        recognition result, and Scryfall card data fields flattened into a
        single record.
        """
        rows: list[dict] = []
        for card in self.cards:
            row: dict = {
                "image_path": self.image_path,
                "scan_timestamp": self.scan_timestamp,
                "patch_index": card.patch.patch_index,
                "bbox_x": card.patch.bbox[0],
                "bbox_y": card.patch.bbox[1],
                "bbox_w": card.patch.bbox[2],
                "bbox_h": card.patch.bbox[3],
                "detection_confidence": card.patch.detection_confidence,
                "card_name": card.card_name or "",
                "recognition_confidence": card.recognition_confidence,
                "recognition_method": card.recognition_method,
                # CardData fields (may be absent)
                "set_code": "",
                "collector_number": "",
                "rarity": "",
                "type_line": "",
                "price_eur": "",
                "price_usd": "",
                "scryfall_uri": "",
                "image_uri": "",
            }
            if card.card_data is not None:
                cd = card.card_data
                row.update(
                    {
                        "set_code": cd.set_code,
                        "collector_number": cd.collector_number,
                        "rarity": cd.rarity,
                        "type_line": cd.type_line,
                        "price_eur": cd.price_eur if cd.price_eur is not None else "",
                        "price_usd": cd.price_usd if cd.price_usd is not None else "",
                        "scryfall_uri": cd.scryfall_uri,
                        "image_uri": cd.image_uri,
                    }
                )
            rows.append(row)
        return rows

    def to_json(self) -> dict:
        """Return a JSON-serialisable dict representation of the scan result."""

        def _card_data_dict(cd) -> dict | None:
            if cd is None:
                return None
            return {
                "name": cd.name,
                "set_code": cd.set_code,
                "collector_number": cd.collector_number,
                "rarity": cd.rarity,
                "type_line": cd.type_line,
                "price_eur": cd.price_eur,
                "price_usd": cd.price_usd,
                "scryfall_uri": cd.scryfall_uri,
                "image_uri": cd.image_uri,
            }

        cards_list = []
        for c in self.cards:
            cards_list.append(
                {
                    "patch_index": c.patch.patch_index,
                    "bbox": list(c.patch.bbox),
                    "detection_confidence": c.patch.detection_confidence,
                    "card_name": c.card_name,
                    "recognition_confidence": c.recognition_confidence,
                    "recognition_method": c.recognition_method,
                    "card_data": _card_data_dict(c.card_data),
                }
            )

        return {
            "image_path": self.image_path,
            "scan_timestamp": self.scan_timestamp,
            "total_detected": self.total_detected,
            "total_recognized": self.total_recognized,
            "total_unknown": self.total_unknown,
            "cards": cards_list,
        }

    def summary(self) -> str:
        """Return a human-readable one-line summary of the scan result."""
        lines = [
            f"Scan: {self.image_path}  [{self.scan_timestamp}]",
            f"  Detected : {self.total_detected}",
            f"  Recognised: {self.total_recognized}",
            f"  Unknown  : {self.total_unknown}",
        ]
        for c in self.cards:
            name_str = c.card_name or "<unknown>"
            price_str = ""
            if c.card_data and c.card_data.price_eur is not None:
                price_str = f"  €{c.card_data.price_eur:.2f}"
            lines.append(
                f"  [{c.patch.patch_index}] {name_str}"
                f"  ({c.recognition_method}, conf={c.recognition_confidence:.2f}){price_str}"
            )
        return "\n".join(lines)
