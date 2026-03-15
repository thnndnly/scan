"""Ground-truth labeling and evaluation utilities for mtg-card-scanner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_GT_PATH = "data/ground_truth.json"


def load_ground_truth(path: str = _DEFAULT_GT_PATH) -> dict:
    """Load ground truth JSON. Returns empty dict if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def save_ground_truth(data: dict, path: str = _DEFAULT_GT_PATH) -> None:
    """Save ground truth dict to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def label_image(
    image_path: str,
    expected_count: Optional[int] = None,
    expected_cards: Optional[list[str]] = None,
    gt_path: str = _DEFAULT_GT_PATH,
) -> None:
    """Add or update a ground-truth entry for image_path."""
    data = load_ground_truth(gt_path)
    entry: dict = data.get(image_path, {})
    if expected_count is not None:
        entry["expected_count"] = expected_count
    if expected_cards is not None:
        entry["expected_cards"] = expected_cards
    data[image_path] = entry
    save_ground_truth(data, gt_path)


def evaluate_scan(scan_result, gt_path: str = _DEFAULT_GT_PATH) -> dict:
    """Compare a ScanResult against stored ground truth.

    Returns a dict with keys:
        image_path, expected_count, detected_count, recognised_count,
        expected_cards, detected_cards, matched, missed, extra,
        precision, recall, detection_rate
    """
    data = load_ground_truth(gt_path)
    image_path = scan_result.image_path
    gt = data.get(image_path, {})

    expected_count = gt.get("expected_count")
    expected_cards = [c.lower() for c in gt.get("expected_cards", [])]
    detected_cards = [
        c.card_name.lower() for c in scan_result.cards if c.card_name
    ]

    detected_count = scan_result.total_detected
    recognised_count = scan_result.total_recognized

    matched = [c for c in detected_cards if c in expected_cards]
    missed = [c for c in expected_cards if c not in detected_cards]
    extra = [c for c in detected_cards if c not in expected_cards]

    precision = len(matched) / len(detected_cards) if detected_cards else 0.0
    recall = len(matched) / len(expected_cards) if expected_cards else 0.0
    detection_rate = (
        detected_count / expected_count if expected_count else None
    )

    return {
        "image_path": image_path,
        "expected_count": expected_count,
        "detected_count": detected_count,
        "recognised_count": recognised_count,
        "expected_cards": gt.get("expected_cards", []),
        "detected_cards": [c.card_name for c in scan_result.cards if c.card_name],
        "matched": matched,
        "missed": missed,
        "extra": extra,
        "precision": precision,
        "recall": recall,
        "detection_rate": detection_rate,
    }
