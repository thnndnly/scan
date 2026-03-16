"""CLIP embedding-based card recogniser.

Compares the artwork of a card patch against a prebuilt database of CLIP
image embeddings using cosine similarity.  Build the database first with:

    mtg-scan db build-clip
    # or:
    python scripts/build_clip_db.py --sets m21,lea
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.recognition.base import BaseRecognizer

logger = logging.getLogger(__name__)


class ClipRecognizer(BaseRecognizer):
    """Card recogniser using CLIP image embeddings.

    Crops the artwork region (20%–65% of card height) from each patch,
    computes a CLIP embedding, and finds the nearest neighbour in the
    prebuilt embedding database using cosine similarity.

    Args:
        db_path: Path to the SQLite CLIP embedding database.
        model_name: HuggingFace model ID for CLIP.
        top_k: Number of nearest neighbours to inspect.
        similarity_threshold: Minimum cosine similarity to accept a match.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        try:
            from mtg_scanner.config import get_config
            cfg = get_config()
            self._db_path = db_path or cfg.clip.db_path
            self._model_name = model_name or cfg.clip.model_name
            self._threshold = similarity_threshold if similarity_threshold is not None else cfg.clip.similarity_threshold
        except Exception:
            self._db_path = db_path or "data/clip_embeddings.db"
            self._model_name = model_name or "openai/clip-vit-base-patch32"
            self._threshold = similarity_threshold if similarity_threshold is not None else 0.25

        self._top_k = top_k
        self._model = None
        self._processor = None
        self._embeddings: Optional[np.ndarray] = None
        self._card_names: Optional[list[str]] = None
        self._scryfall_ids: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is not None:
            return self._model, self._processor
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for CLIP recognition.\n"
                "Install it with:  pip install 'mtg-card-scanner[clip]'"
            ) from exc
        logger.info("Loading CLIP model: %s", self._model_name)
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model = CLIPModel.from_pretrained(self._model_name)
        self._model.eval()
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Embedding database
    # ------------------------------------------------------------------

    def _load_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        db = Path(self._db_path)
        if not db.exists():
            raise FileNotFoundError(
                f"CLIP embedding DB not found at '{self._db_path}'. "
                "Run: mtg-scan db build-clip"
            )
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT scryfall_id, card_name, embedding FROM clip_embeddings ORDER BY rowid"
        ).fetchall()
        conn.close()

        if not rows:
            raise RuntimeError("CLIP embedding DB is empty. Run: mtg-scan db build-clip")

        self._scryfall_ids = [r[0] for r in rows]
        self._card_names = [r[1] for r in rows]

        # Reconstruct float32 matrix from binary blobs
        raw = b"".join(r[2] for r in rows)
        n = len(rows)
        dim = len(raw) // (n * 4)  # float32 = 4 bytes
        embs = np.frombuffer(raw, dtype=np.float32).reshape(n, dim).copy()

        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self._embeddings = embs / np.maximum(norms, 1e-8)
        logger.info("Loaded %d CLIP embeddings from %s", n, self._db_path)

    # ------------------------------------------------------------------
    # Patch embedding
    # ------------------------------------------------------------------

    def _embed_patch(self, patch_image: np.ndarray) -> np.ndarray:
        """Compute a normalised CLIP embedding for the artwork region of *patch_image*."""
        import cv2  # type: ignore
        import torch  # type: ignore
        from PIL import Image as PILImage  # type: ignore

        h, w = patch_image.shape[:2]
        y0 = int(h * 0.20)
        y1 = int(h * 0.65)
        margin = 5
        crop = patch_image[y0:y1, margin:w - margin]
        if crop.size == 0:
            crop = patch_image

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(crop_rgb)

        model, processor = self._get_model()
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)

        emb = features[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-8)

    # ------------------------------------------------------------------
    # BaseRecognizer interface
    # ------------------------------------------------------------------

    def recognize(self, patch: CardPatch) -> tuple[Optional[str], float]:
        """Identify the card via CLIP nearest-neighbour search.

        Args:
            patch: Detected card patch.

        Returns:
            ``(card_name, similarity_score)`` or ``(None, 0.0)`` on failure.
        """
        try:
            self._load_embeddings()
        except FileNotFoundError as exc:
            logger.warning("CLIP: %s", exc)
            return None, 0.0
        except Exception as exc:
            logger.error("CLIP: failed to load embeddings: %s", exc)
            return None, 0.0

        try:
            query = self._embed_patch(patch.image)
        except Exception as exc:
            logger.warning("CLIP: failed to embed patch: %s", exc)
            return None, 0.0

        # Cosine similarity — embeddings are already L2-normalised
        sims = self._embeddings @ query
        top_indices = np.argsort(sims)[::-1][: self._top_k]
        best_idx = int(top_indices[0])
        best_sim = float(sims[best_idx])

        if best_sim < self._threshold:
            logger.debug(
                "CLIP: best similarity %.3f below threshold %.3f", best_sim, self._threshold
            )
            return None, best_sim

        card_name = self._card_names[best_idx]  # type: ignore[index]
        scryfall_id = self._scryfall_ids[best_idx]  # type: ignore[index]
        logger.info(
            "CLIP: matched → %r (similarity=%.3f, id=%s)", card_name, best_sim, scryfall_id
        )
        return card_name, best_sim

    def stats(self) -> dict:
        """Return embedding database statistics."""
        db = Path(self._db_path)
        if not db.exists():
            return {"exists": False, "count": 0, "path": self._db_path}
        conn = sqlite3.connect(self._db_path)
        count = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
        sets = conn.execute(
            "SELECT DISTINCT set_code FROM clip_embeddings"
        ).fetchall()
        conn.close()
        return {
            "exists": True,
            "count": count,
            "sets": len(sets),
            "path": self._db_path,
            "model": self._model_name,
        }
