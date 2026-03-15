"""Configuration management for mtg-card-scanner.

Loads config.yaml from the project root (or a path specified via the
MTG_SCANNER_CONFIG environment variable) and exposes typed Pydantic v2 models.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DetectionConfig(BaseModel):
    """Settings that control the card-detection stage."""

    method: Literal["opencv", "yolo"] = "opencv"
    yolo_model_path: str = "data/models/yolo_mtg.pt"
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    aspect_ratio_min: float = Field(0.60, ge=0.0)
    aspect_ratio_max: float = Field(0.85, ge=0.0)
    min_card_area_px: int = Field(5000, ge=0)
    save_debug: bool = False


class RecognitionConfig(BaseModel):
    """Settings that control the card-recognition stage."""

    primary_method: Literal["ocr", "hash"] = "ocr"
    fallback_method: Literal["ocr", "hash", "llm"] = "hash"
    ocr_confidence_threshold: float = Field(0.70, ge=0.0, le=1.0)
    hash_max_hamming_distance: int = Field(12, ge=0)
    llm_fallback_enabled: bool = False
    ocr_languages: list[str] = Field(default_factory=lambda: ["en", "de"])
    ocr_languages_cjk: list[str] = Field(default_factory=lambda: ["ja"])


class ScryfallConfig(BaseModel):
    """Settings for Scryfall API access and local caching."""

    cache_ttl_hours: int = Field(24, ge=0)
    rate_limit_ms: int = Field(110, ge=0)
    cache_db_path: str = "data/scryfall_cache.db"


class OutputConfig(BaseModel):
    """Settings for scan result output."""

    default_format: Literal["csv", "json", "both"] = "both"
    output_dir: str = "./output"
    save_card_patches: bool = False
    low_confidence_threshold: float = Field(0.60, ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
    """Settings for the dataset logger."""

    enabled: bool = True
    db_path: str = "data/dataset.db"
    save_patches: bool = True


class ArchiveConfig(BaseModel):
    """Settings for the permanent image archive."""

    enabled: bool = True
    db_path: str = "data/image_archive.db"
    index_path: str = "data/image_archive_index.json"
    max_dimension: int = Field(1920, ge=100)
    jpeg_quality: int = Field(70, ge=10, le=95)
    thumbnail_size: int = Field(300, ge=50)
    thumbnail_quality: int = Field(50, ge=10, le=95)


class CatalogConfig(BaseModel):
    """Settings for the local Scryfall card catalog."""

    enabled: bool = True
    db_path: str = "data/card_catalog.db"
    bulk_type: str = "default_cards"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    scryfall: ScryfallConfig = Field(default_factory=ScryfallConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATHS: list[Path] = [
    Path("config.yaml"),
    Path(__file__).parent.parent.parent.parent / "config.yaml",  # repo root
]


def _find_config_file() -> Optional[Path]:
    """Return the first existing config file path, or None."""
    env_path = os.environ.get("MTG_SCANNER_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate

    return None


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and return the application configuration.

    Looks for ``config.yaml`` in the current directory, the repository root,
    or the path set in the ``MTG_SCANNER_CONFIG`` environment variable.  Falls
    back to defaults when no file is found.

    Returns:
        Fully validated :class:`AppConfig` instance.
    """
    config_path = _find_config_file()
    if config_path is None:
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(raw)


def reload_config() -> AppConfig:
    """Clear the cache and reload the configuration from disk.

    Useful in tests or after the config file has been modified at runtime.
    """
    get_config.cache_clear()
    return get_config()
