"""Generate synthetic training data for YOLOv8 card detection.

Usage
-----
    python scripts/generate_training_data.py --count 200 --output data/training

For each synthetic image the script:
1. Creates a random background (solid colour or noise).
2. Places 5–20 randomly selected card images from Scryfall.
3. Applies per-card augmentation: rotation ±25°, brightness ±30%, slight blur.
4. Allows partial overlaps between cards.
5. Writes the image and a YOLO-format label file (.txt).
6. Creates a ``dataset.yaml`` file compatible with Ultralytics YOLOv8.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm

try:
    from PIL import Image, ImageFilter
except ImportError:
    print("Pillow is required: pip install pillow", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# Scryfall random card endpoint
_RANDOM_CARD_URL = "https://api.scryfall.com/cards/random"
_RATE_LIMIT_S = 0.11

# Output dimensions
_CANVAS_W = 1280
_CANVAS_H = 720

# Card display size range (pixels wide)
_CARD_W_MIN = 80
_CARD_W_MAX = 200
_CARD_ASPECT = 1.40  # h / w (standard card portrait)

# Augmentation parameters
_ROT_MAX_DEG = 25
_BRIGHTNESS_DELTA = 0.30
_BLUR_PROB = 0.3
_BLUR_MAX_RADIUS = 1.5


def _random_background(w: int, h: int) -> np.ndarray:
    """Create a random background (solid colour or Gaussian noise)."""
    if random.random() < 0.5:
        color = [random.randint(20, 230) for _ in range(3)]
        bg = np.full((h, w, 3), color, dtype=np.uint8)
    else:
        bg = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return bg


def _fetch_card_image(session: requests.Session) -> Optional[np.ndarray]:
    """Fetch a random card image from Scryfall and return a BGR numpy array."""
    try:
        resp = session.get(_RANDOM_CARD_URL, timeout=15)
        resp.raise_for_status()
        card_data = resp.json()
        image_uris = card_data.get("image_uris") or {}
        url = image_uris.get("normal") or image_uris.get("small")
        if not url:
            return None
        img_resp = session.get(url, timeout=30)
        img_resp.raise_for_status()
        img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
        bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return bgr
    except Exception as exc:
        logger.warning("Failed to fetch card image: %s", exc)
        return None


def _augment_card(card: np.ndarray) -> np.ndarray:
    """Apply random brightness change and optional blur."""
    factor = 1.0 + random.uniform(-_BRIGHTNESS_DELTA, _BRIGHTNESS_DELTA)
    card = np.clip(card.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if random.random() < _BLUR_PROB:
        radius = random.uniform(0.5, _BLUR_MAX_RADIUS)
        ksize = max(1, int(radius * 2) | 1)  # must be odd
        card = cv2.GaussianBlur(card, (ksize, ksize), 0)
    return card


def _rotate_card(
    card: np.ndarray, angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate *card* by *angle_deg* and return (rotated_image, corners).

    *corners* is a (4, 2) float32 array with the corner coordinates of the
    card in the rotated image space.
    """
    h, w = card.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    rotated = cv2.warpAffine(card, M, (new_w, new_h))

    # Transform original corners
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])
    rotated_corners = (M @ corners_h.T).T
    return rotated, rotated_corners


def _place_card_on_canvas(
    canvas: np.ndarray,
    card: np.ndarray,
    x: int,
    y: int,
    angle_deg: float,
) -> tuple[np.ndarray, list[float]]:
    """Composite *card* onto *canvas* and return (updated_canvas, yolo_bbox).

    The YOLO bounding box is [class_id, cx, cy, w, h] normalised to [0, 1].
    """
    rotated, corners = _rotate_card(card, angle_deg)
    rh, rw = rotated.shape[:2]
    cw, ch = canvas.shape[1], canvas.shape[0]

    x = int(max(0, min(x, cw - rw)))
    y = int(max(0, min(y, ch - rh)))

    # Paste the non-black pixels (the rotated card has black padding)
    roi = canvas[y : y + rh, x : x + rw]
    if roi.shape[:2] != (rh, rw):
        return canvas, []

    mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    roi_copy = roi.copy()
    roi_copy[mask > 0] = rotated[mask > 0]
    canvas[y : y + rh, x : x + rw] = roi_copy

    # YOLO bbox from rotated bounding box of corners
    shifted_corners = corners + np.array([x, y])
    x_coords = shifted_corners[:, 0]
    y_coords = shifted_corners[:, 1]
    bx1, bx2 = max(0, float(x_coords.min())), min(cw, float(x_coords.max()))
    by1, by2 = max(0, float(y_coords.min())), min(ch, float(y_coords.max()))
    bbox_cx = ((bx1 + bx2) / 2.0) / cw
    bbox_cy = ((by1 + by2) / 2.0) / ch
    bbox_w = (bx2 - bx1) / cw
    bbox_h = (by2 - by1) / ch

    if bbox_w <= 0 or bbox_h <= 0:
        return canvas, []

    return canvas, [0, bbox_cx, bbox_cy, bbox_w, bbox_h]


def generate_training_images(
    count: int,
    output_dir: Path,
    card_pool_size: int = 50,
) -> None:
    """Generate *count* synthetic training images with YOLO labels.

    Args:
        count: Number of images to generate.
        output_dir: Root directory; ``images/`` and ``labels/`` subdirs are
            created automatically.
        card_pool_size: How many unique card images to pre-fetch from Scryfall
            (they are reused across generated images to reduce API calls).
    """
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "mtg-card-scanner/0.1"

    # Pre-fetch a pool of card images
    logger.info("Pre-fetching %d card images from Scryfall…", card_pool_size)
    card_pool: list[np.ndarray] = []
    with tqdm(total=card_pool_size, desc="Fetching cards") as bar:
        while len(card_pool) < card_pool_size:
            img = _fetch_card_image(session)
            if img is not None:
                card_pool.append(img)
                bar.update(1)
            time.sleep(_RATE_LIMIT_S)

    logger.info("Generating %d training images…", count)
    for i in tqdm(range(count), desc="Generating images"):
        canvas = _random_background(_CANVAS_W, _CANVAS_H)
        labels: list[list[float]] = []

        n_cards = random.randint(5, 20)
        for _ in range(n_cards):
            card_src = random.choice(card_pool).copy()
            card_w = random.randint(_CARD_W_MIN, _CARD_W_MAX)
            card_h = int(card_w * _CARD_ASPECT)
            card_resized = cv2.resize(card_src, (card_w, card_h))
            card_aug = _augment_card(card_resized)

            angle = random.uniform(-_ROT_MAX_DEG, _ROT_MAX_DEG)
            x = random.randint(-card_w // 4, _CANVAS_W - card_w // 4)
            y = random.randint(-card_h // 4, _CANVAS_H - card_h // 4)

            canvas, bbox = _place_card_on_canvas(canvas, card_aug, x, y, angle)
            if bbox:
                labels.append(bbox)

        img_path = images_dir / f"synthetic_{i:05d}.jpg"
        lbl_path = labels_dir / f"synthetic_{i:05d}.txt"

        cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])

        with open(lbl_path, "w") as fh:
            for lbl in labels:
                fh.write(" ".join(f"{v:.6f}" if j > 0 else str(int(v)) for j, v in enumerate(lbl)) + "\n")

    # Write dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/train",  # for now reuse; split manually if needed
        "nc": 1,
        "names": ["mtg_card"],
    }
    import yaml

    with open(yaml_path, "w") as fh:
        yaml.dump(dataset_yaml, fh, default_flow_style=False)

    logger.info("Training data written to %s", output_dir)
    logger.info("dataset.yaml: %s", yaml_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLOv8 training data.")
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate.")
    parser.add_argument(
        "--output", default="data/training", help="Output directory for images and labels."
    )
    parser.add_argument(
        "--pool", type=int, default=50, help="Number of unique card images to prefetch."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    args = _parse_args()
    generate_training_images(
        count=args.count,
        output_dir=Path(args.output),
        card_pool_size=args.pool,
    )


if __name__ == "__main__":
    main()
