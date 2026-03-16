# CLAUDE.md — MTG Card Scanner

## Project overview

Python tool that detects and identifies Magic: The Gathering cards in photos
(eBay listings, bulk lots, etc.) via OpenCV + EasyOCR + Scryfall API.
Includes a Gradio web interface, SQLite-based scan dataset logger, permanent
image archive, full card catalog (Scryfall bulk data), and CLI tools for data
management.

---

## Environment setup

- **Python**: 3.12 (PyTorch does not support 3.13+ yet)
- **Virtual env**: `.venv/` — always use `.venv/Scripts/python` / `.venv/Scripts/pip`
- **PyTorch**: CPU-only build `2.2.2+cpu` (avoid the default pip version which breaks on Windows)
- **NumPy**: pinned to `<2` for torch 2.2 compatibility — Gradio upgrades it, must re-pin after

### Setup from scratch

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install "numpy<2"
.venv\Scripts\pip install -e ".[ui]"
.venv\Scripts\pip install "numpy<2"          # Re-pin after Gradio install
.venv\Scripts\python scripts/download_card_names.py
.venv\Scripts\python scripts/download_card_names.py --lang de ja
.venv\Scripts\python scripts/build_card_catalog.py   # ~155 MB, ~2 min
```

### Optional extras

```bash
# YOLOv8 detector
.venv\Scripts\pip install "mtg-card-scanner[yolo]"

# GPT-4o vision fallback
.venv\Scripts\pip install "mtg-card-scanner[llm]"

# Claude Vision fallback (needs ANTHROPIC_API_KEY env var)
.venv\Scripts\pip install "mtg-card-scanner[claude]"

# CLIP artwork embedding recognition (includes OpenCLIP)
.venv\Scripts\pip install "mtg-card-scanner[clip]"

# PaddleOCR recognizer (~5-10% better accuracy than EasyOCR)
.venv\Scripts\pip install "mtg-card-scanner[paddle]"
```

---

## All available commands

### Scan images

```bash
mtg-scan scan <image_or_dir>
mtg-scan scan <image> --verbose
mtg-scan scan <image> --format csv
mtg-scan scan <image> --detector yolo
mtg-scan scan <image> --recognizer hash
mtg-scan scan <image> --recognizer paddle
mtg-scan scan <image> --save-patches
```

### Gradio web UI

```bash
mtg-scan ui                     # Launch at http://localhost:7860
mtg-scan ui --port 8080
mtg-scan ui --share             # Create public Gradio link
```

### Dataset management

```bash
mtg-scan dataset stats                      # Scan history overview
mtg-scan dataset export-csv [PATH]          # Export all results as CSV
mtg-scan dataset export-yolo [DIR]          # Export patches + bboxes as YOLO training data
mtg-scan dataset corrections                # Show manually corrected cards
```

### Card catalog

```bash
mtg-scan catalog build                      # Download & import (~155 MB, einmalig)
mtg-scan catalog build --force              # Force re-download
mtg-scan catalog build --check             # Only check freshness
mtg-scan catalog stats                      # Show counts, Scryfall timestamp
mtg-scan catalog search "Lightning Bolt"   # Search all printings
```

### Image archive

```bash
mtg-scan archive stats          # Images, sizes, savings
mtg-scan archive verify         # SHA256 integrity check
mtg-scan archive export [DIR]   # Export all images as JPEG
```

### Ground-truth labeling & evaluation

```bash
mtg-scan label set foto.jpg -n 9 --cards "Karte1,Karte2,..."
mtg-scan label show             # All stored labels
mtg-scan label eval foto.jpg    # Scan + compare against label
```

### Database / data management

```bash
mtg-scan db stats               # Show stats for all local databases
mtg-scan db update-names        # Re-download card names from Scryfall
mtg-scan db build-hashes        # Build perceptual hash DB (long-running)
mtg-scan db build-hashes --sets m21,lea
mtg-scan db build-hashes --limit 500
mtg-scan db build-hashes --dry-run
mtg-scan db build-clip          # Build CLIP embedding DB from Scryfall artwork
mtg-scan db build-clip --sets m21,lea
mtg-scan db build-clip --limit 500 --dry-run
mtg-scan db train-yolo          # Fine-tune YOLOv8 on validated patches
mtg-scan db train-yolo --epochs 50 --base-model yolov8s.pt
mtg-scan db train-yolo --dry-run
```

### Configuration

```bash
mtg-scan config show            # Print effective config as YAML
mtg-scan config init            # Write default config.yaml
```

### Help system

```bash
mtg-scan --help                 # or -h
mtg-scan help                   # Overview with quick-start examples
mtg-scan help scan              # Detailed help for scan
mtg-scan help db                # Detailed help for db
mtg-scan help catalog           # Detailed help for catalog
mtg-scan help archive           # Detailed help for archive
mtg-scan help label             # Detailed help for label
mtg-scan help ui                # UI tab overview
mtg-scan help config            # Config reference
mtg-scan <group> help           # e.g. mtg-scan archive help
mtg-scan <group> -h             # short form works everywhere
```

### Scripts (direct)

```bash
.venv\Scripts\python scripts/download_card_names.py
.venv\Scripts\python scripts/download_card_names.py --lang de ja
.venv\Scripts\python scripts/build_card_catalog.py
.venv\Scripts\python scripts/build_card_catalog.py --force
.venv\Scripts\python scripts/build_card_catalog.py --check
.venv\Scripts\python scripts/build_hash_db.py
.venv\Scripts\python scripts/build_hash_db.py --sets m21,lea --limit 500
.venv\Scripts\python scripts/build_hash_db.py --dry-run

# Multilingual card names via MTGJSON (alternative to Scryfall names endpoint)
.venv\Scripts\python scripts/download_mtgjson.py
.venv\Scripts\python scripts/download_mtgjson.py --lang de ja fr es
# Downloads AllPrintings.json.bz2 (~700 MB, cached), writes data/card_names_{lang}.json
# Available lang codes: de, fr, es, it, pt, ja, ko, ru, zhs, zht
```

### Tests

```bash
.venv\Scripts\pytest
.venv\Scripts\pytest -v
```

---

## Architecture

```
pipeline.py           # Orchestrates: detect → recognise → lookup → dataset log → archive
dataset.py            # SQLite dataset logger: scans, detections, attempts, results
image_archive.py      # Permanent compressed image store (SHA256, JPEG, thumbnails)
evaluation.py         # Ground-truth labeling + scan evaluation
ui.py                 # Gradio web interface (9 tabs, all labels in German)
cli.py                # Click CLI: scan, ui, dataset, db, catalog, archive, label, config
detection/
  opencv_detector.py  # Canny edges → IoU-NMS (configurable threshold) → perspective patches
  yolo_detector.py    # YOLOv8 (optional, needs [yolo] extra)
recognition/
  ocr_recognizer.py    # EasyOCR on top-13%/left-70% title strip (mana cost excluded); tries raw +
                       # inverted-grayscale variants, picks best rapidfuzz match; CJK fallback last
  hash_recognizer.py   # pHash of artwork region vs. SQLite DB (path from config.recognition.hash_db_path)
  llm_recognizer.py    # GPT-4o vision fallback (needs OPENAI_API_KEY)
  claude_recognizer.py # Claude Vision fallback (needs ANTHROPIC_API_KEY)
  clip_recognizer.py   # CLIP artwork embedding cosine similarity vs. clip_embeddings.db
  paddle_recognizer.py # PaddleOCR fallback (optional, needs [paddle] extra)
lookup/
  scryfall_client.py  # GET /cards/named?fuzzy=... with SQLite cache
  card_catalog.py     # Query interface over local card_catalog.db
models/               # CardPatch, RecognizedCard, CardData, ScanResult
utils/
  image_utils.py      # load_image() — OpenCV + Pillow fallback (AVIF/HEIC) + EXIF rotation correction
  fuzzy_search.py     # rapidfuzz wrapper against data/card_names.json; O(1) lower-cache
config.py             # Pydantic v2 models: AppConfig + all sub-configs
```

---

## Data files

| File | Created by | Purpose |
|------|-----------|---------|
| `data/card_names.json` | `scripts/download_card_names.py` | ~33k English card names for OCR fuzzy match |
| `data/card_names_de.json` | `scripts/download_card_names.py --lang de` | DE→EN name mapping (~28k) |
| `data/card_names_ja.json` | `scripts/download_card_names.py --lang ja` | JA→EN name mapping (~29k) |
| `data/card_names_{lang}.json` | `scripts/download_mtgjson.py --lang {lang}` | Any language via MTGJSON foreignData |
| `data/card_catalog.db` | `scripts/build_card_catalog.py` | Full card catalog: ~130k printings, all metadata, no images (~80 MB) |
| `data/card_hashes.db` | `scripts/build_hash_db.py` | pHash DB for hash recognition |
| `data/clip_embeddings.db` | `scripts/build_clip_db.py` | CLIP embedding DB for artwork recognition |
| `data/models/yolo_mtg.pt` | `scripts/train_yolo.py` | Fine-tuned YOLOv8 model |
| `data/scryfall_cache.db` | runtime | Scryfall API response cache (TTL 24h) |
| `data/dataset.db` | runtime (pipeline) | Scan history: scans, detections, recognition attempts, results |
| `data/image_archive.db` | runtime (pipeline) | Permanent compressed image store (SHA256, JPEG 70%) |
| `data/image_archive_index.json` | runtime (pipeline) | **Git-trackable** manifest: hashes, filenames, card assignments — NO image bytes |
| `data/ground_truth.json` | `mtg-scan label set` | Expected card counts/names per image for evaluation |
| `data/patches/YYYY-MM-DD/` | runtime (pipeline) | Patch PNGs saved per scan date |
| `data/test_images/` | manual | Example scan photos for testing (Alpha deck, single cards, etc.) |
| `config.yaml` | `mtg-scan config init` | Application configuration |

**Git tracking:**
- `data/image_archive_index.json` — commit this (metadata only, no images)
- Everything else in `data/` — gitignored (large binaries)

---

## Dataset DB schema

`data/dataset.db` tables (auto-migrated on first open):

**`scans`** — one row per scanned image:
- `corrected_count INTEGER` — user-corrected card count (set in Nachkontrolle)
- `reviewed INTEGER DEFAULT 0` — set to 1 when user clicks "Korrektur abschließen"; prevents re-editing in UI

**`detections`** — one row per detected card patch:
- `detection_approved INTEGER` — NULL=unchecked, 1=approved, 0=rejected (set in Nachkontrolle)

**`results`** — one row per detection, recognition output:
- `scryfall_id TEXT`, `oracle_id TEXT` — set when patch is assigned to a catalog entry

---

## Gradio UI tabs

| Tab | Purpose |
|-----|---------|
| Scanner | Upload **one or more** images simultaneously, run scan, patch gallery with card names + prices |
| Scan-Historie | Browse past scans, click row to view patches |
| Nachkontrolle | Post-scan review: correct card count, approve/reject detections, correct card names with catalog suggestions. Each scan can be reviewed **exactly once** — clicking "Korrektur abschließen" locks it |
| Hash-DB | Build/status the perceptual hash DB with dry-run preview |
| Training | Fine-tune YOLO + build CLIP embedding DB; shows dataset stats and process log |
| Karten-Zuordnung | Assign patches to specific catalog entries (set/printing/finish) + card image preview + add to collection |
| Auswertung | Ground-truth labels, scan vs. expectation comparison |
| Archiv | Browse image archive, view thumbnails, export all images |
| Einstellungen | Edit detection method, OCR threshold, dataset toggle |

---

## Nachkontrolle workflow (one-time review)

1. Load scan list — by default only **unreviewed** scans shown; checkbox "Erledigte Scans anzeigen" reveals finished ones
2. Click a scan row → patches load into gallery, card count field pre-fills
3. Correct the **card count** if needed → "Anzahl speichern"
4. Click a **patch** in the gallery → shows OCR raw text, recognised name, confidence
5. Click **"✓ Karte korrekt erkannt"** or **"✗ Keine Karte / Fehler"** to mark detection
6. Or: enter search term → "Katalog durchsuchen" → click a suggestion → "Korrektur speichern"
7. When done with the whole scan: **"Korrektur abschließen ✓"** → scan is marked `reviewed=1`, all controls lock, scan disappears from default list

---

## Card catalog

Built from Scryfall's `default_cards` bulk export (~130,000 printings, no images).

```bash
# Build once (~155 MB download, ~2 min import):
mtg-scan catalog build

# Subsequent runs: only re-downloads if Scryfall published a newer version
mtg-scan catalog build

# Force re-download:
mtg-scan catalog build --force
```

**Fields per card:** `id` (Scryfall UUID), `oracle_id`, `name`, `set_code`, `set_name`,
`collector_number`, `released_at`, `rarity`, `finishes` (foil/nonfoil/etched),
`artist`, `frame_effects` (showcase/borderless/...), `promo`, `prices` (EUR/USD),
`image_uris` (URLs only), `type_line`, `oracle_text`, `lang`

**Freshness:** Poll `/bulk-data` API daily; re-download only when `updated_at` changes.

---

## Patch-to-card assignment workflow

After scanning, patches can be assigned to a specific catalog entry:

1. **UI tab "Karten-Zuordnung"**: Lists all patches without a catalog assignment
2. Click a patch → preview loads, card name pre-fills search
3. Search the catalog → see all printings (set, year, rarity, finish, price, artist)
4. Click the correct printing → "Zuordnung speichern"
5. Result: `scryfall_id` + `oracle_id` saved in both `dataset.db` and `image_archive.db`

This builds a verified dataset over time:
- Each confirmed patch is stored in `image_archive.db` with `patch_type = 'card_patch'`
- Basis for hash-DB refinement and YOLO fine-tuning with real labels

---

## Image archive

Every scanned image is automatically compressed and stored permanently:

- **Deduplication**: SHA256 content-addressing — same photo never stored twice
- **Compression**: JPEG 70%, max 1920px long side (~150–300 KB per photo)
- **Thumbnails**: 300px JPEG 50% stored separately for fast UI display
- **Index**: `data/image_archive_index.json` is git-trackable (no image bytes)
- **Patch storage**: Confirmed card patches stored with `scryfall_id` + `patch_type = 'card_patch'`

```bash
mtg-scan archive stats    # How many images, how much space saved
mtg-scan archive verify   # SHA256 integrity check
mtg-scan archive export . # Extract all images to folder
```

---

## Hash DB build — time and storage estimates

The hash DB downloads every card's artwork image from Scryfall and computes a
perceptual hash. This is a one-time process.

| Scope | Cards | Est. time | Est. DB size |
|-------|-------|-----------|--------------|
| Full collection (~30k cards) | 30,000 | ~50 min | ~30 MB |
| Single set (e.g. m21, ~270 cards) | 270 | ~30 sec | ~300 KB |
| English only | ~28,000 | ~47 min | ~28 MB |

Use `--dry-run` to preview before downloading:

```bash
.venv\Scripts\python scripts/build_hash_db.py --dry-run
.venv\Scripts\python scripts/build_hash_db.py --dry-run --sets m21
```

The build is resumable — cards already in the DB are skipped automatically.

---

## Test images

`data/test_images/` contains example scan photos from the
[tmikonen/magic_card_detector](https://github.com/tmikonen/magic_card_detector) repo (MIT licence).
All are Alpha/Beta-era cards.

| File | Cards | Notes |
|------|-------|-------|
| `alpha_deck.jpg` | 40 | Cards laid out in rows — best stress test for detection |
| `black.jpg` | 8 | Black cards spread loosely with dice on top |
| `geyser_twister_fireball.jpg` | 3 | Cards side by side, good contrast, white background |
| `lands_and_fatties.jpg` | 5 | Cards at angles on dark background — hard case |
| `dragon_whelp.jpg` | 1 | Single card on patterned fabric background |
| `evil_eye.jpg` | 1 | Card in BGS slab (hard plastic case) |
| `counterspell_bgs.jpg` | 1 | Card in Beckett slab with grading label |
| `instill_energy.jpg` | 1 | Single card, clean white background |
| `ruby.jpg` | 1 | Card at an angle on dark surface |

**Observed performance on `alpha_deck.jpg`**: 39/40 cards detected, OCR identification
rate is low because Alpha card frames have low title-text contrast. The hash recognizer
(`--recognizer hash`) with Alpha set hashes (`mtg-scan db build-hashes --sets lea`) is
the recommended approach for old-set cards.

---

## Detection: IoU-based NMS

The OpenCV detector uses **non-maximum suppression** to remove duplicate detections.
RETR_LIST finds all contours including inner card frames; NMS keeps only the largest
non-overlapping box per card.

The IoU threshold is **configurable** via `detection.iou_nms_threshold` (default 0.3).

**Debug mode** — saves edge map + annotated detection image to `output/debug/`:
```yaml
# config.yaml
detection:
  save_debug: true
```

---

## Config reference

`config.yaml` in the project root is loaded at startup.
Override path via the `MTG_SCANNER_CONFIG` environment variable.

```yaml
detection:
  method: opencv            # 'opencv' or 'yolo'
  yolo_model_path: data/models/yolo_mtg.pt
  confidence_threshold: 0.5
  aspect_ratio_min: 0.60
  aspect_ratio_max: 0.85
  min_card_area_px: 5000
  max_card_area_frac: 0.50
  iou_nms_threshold: 0.3    # IoU threshold for non-maximum suppression
  save_debug: false         # Save edge map + detection image to output/debug/

recognition:
  primary_method: ocr       # 'ocr', 'hash', 'clip', 'claude', 'paddle'
  fallback_method: hash     # single fallback (legacy)
  # OR: use fallback_chain for a sequence (overrides fallback_method):
  # fallback_chain: [hash, clip, claude]
  # Valid methods: ocr, hash, clip, claude, llm, paddle
  ocr_confidence_threshold: 0.70
  hash_max_hamming_distance: 12
  hash_db_path: data/card_hashes.db   # explicit path for hash DB
  paddle_confidence_threshold: 0.70   # combined OCR+fuzzy score threshold for PaddleOCR
  llm_fallback_enabled: false
  ocr_languages: [en, de]
  ocr_languages_cjk: [ja]

scryfall:
  cache_ttl_hours: 24
  rate_limit_ms: 110
  cache_db_path: data/scryfall_cache.db
  prefer_local: true        # Use local card_catalog.db before hitting the API

output:
  default_format: both      # 'csv', 'json', or 'both'
  output_dir: ./output
  save_card_patches: false
  low_confidence_threshold: 0.60   # Fallback results below this are discarded
  recognition_timeout_seconds: 0   # Max seconds per card recognition; 0 = disabled

dataset:
  enabled: true             # Log all scans to SQLite
  db_path: data/dataset.db
  save_patches: true        # Save patch PNGs to data/patches/YYYY-MM-DD/

archive:
  enabled: true             # Store every scanned image permanently
  db_path: data/image_archive.db
  index_path: data/image_archive_index.json
  max_dimension: 1920       # Resize to max 1920px on longest side
  jpeg_quality: 70          # JPEG compression quality (10–95)
  thumbnail_size: 300
  thumbnail_quality: 50

catalog:
  enabled: true
  db_path: data/card_catalog.db
  bulk_type: default_cards  # oracle_cards | default_cards | all_cards

collection:
  db_path: data/collection.db

claude:
  model: claude-sonnet-4-6      # Claude model for Vision recognition
  names_file: data/card_names.json
  fuzzy_cutoff: 80.0            # Minimum fuzzy-match score to accept result

clip:
  db_path: data/clip_embeddings.db
  model_name: openai/clip-vit-base-patch32
  similarity_threshold: 0.25   # Minimum cosine similarity to accept match
  top_k: 5
  device: cpu                  # 'cpu' or 'cuda'
  use_open_clip: false         # Use open_clip_torch instead of transformers
```

### Recognition fallback chain

```yaml
# config.yaml
recognition:
  primary_method: ocr          # 'ocr', 'hash', 'clip', 'claude', 'paddle'
  fallback_method: hash        # single fallback (legacy)
  # OR: use fallback_chain for a sequence (overrides fallback_method):
  fallback_chain: [hash, clip, claude]
  # Valid values: ocr, hash, clip, claude, llm, paddle
  # Invalid entries raise a validation error at startup
```

---

## Known constraints

- **AVIF support**: OpenCV cannot read AVIF; Pillow fallback handles it automatically.
- **EXIF rotation**: `load_image()` auto-corrects EXIF orientation (phone photos in portrait mode). OpenCV path reads EXIF via a lightweight Pillow header read; Pillow fallback uses `ImageOps.exif_transpose()`.
- **PyTorch version**: torch 2.10+ causes `c10.dll` init failure on Windows — use 2.2.2+cpu.
- **NumPy version**: torch 2.2 requires `numpy<2`. Gradio upgrades numpy — always re-pin with `pip install "numpy<2"` after installing/updating Gradio.
- **Hash DB**: not included in repo (requires large card image downloads). OCR works without it. Path is now `data/card_hashes.db` (from `config.recognition.hash_db_path`), no longer derived from scryfall cache path. Hash candidates are loaded once into RAM on first `recognize()` call and cached for all subsequent calls.
- **Card catalog**: not included in repo (~80 MB SQLite). Build with `mtg-scan catalog build`.
- **Scryfall rate limit**: 110 ms between requests — enforced globally across all `ScryfallClient` instances via a class-level lock, so parallel pipelines cannot exceed it.
- **JA OCR limitation**: Japanese OCR (EasyOCR `ja` model) has lower accuracy than Latin script. Hash-based recognition is more reliable for Japanese cards.
- **Gradio**: requires the `[ui]` extra (`pip install -e ".[ui]"`). After install always re-pin numpy.
- **Dataset logger**: enabled by default. Set `dataset.enabled: false` in `config.yaml` to disable. Pipeline continues normally if the DB cannot be opened. The UI shows a `gr.Warning()` toast when the DB file is missing.
- **Dataset DB concurrency**: uses `PRAGMA foreign_keys = ON` and `PRAGMA journal_mode = WAL`. All write methods are protected by a `threading.Lock`. Each scan's logging (log_scan → log_detections → log_results → finish_scan) runs in a single atomic transaction via `begin_batch()` / `commit_batch()`.
- **Card catalog in UI**: catalog-dependent tabs (Karten-Zuordnung) show a warning if `card_catalog.db` does not exist. Run `mtg-scan catalog build` first.
- **PaddleOCR**: requires `[paddle]` extra. Lazy-loaded — missing install gives a clear error message. Does not require GPU.
- **Nachkontrolle locking**: once a scan is marked `reviewed=1` via "Korrektur abschließen", all UI controls for that scan are disabled. This is intentional — re-editing reviewed scans would corrupt training data consistency.
- **MTGJSON download**: `scripts/download_mtgjson.py` downloads ~700 MB (`AllPrintings.json.bz2`) on first run, cached locally. Subsequent runs reuse the cache.
- **Recognition timeout**: set `output.recognition_timeout_seconds` (default `0` = off) to limit per-card recognition time. Uses `concurrent.futures.ThreadPoolExecutor`; timed-out cards get `method="timeout"` and `confidence=0.0`.
- **EasyOCR thread safety**: lazy reader initialisation uses double-checked locking (`threading.Lock`). Two threads can safely share one `OCRRecognizer` instance.
- **OCR on old card frames (Alpha/Beta/Unlimited)**: Low contrast between title text and card frame colour (e.g. white text on teal blue border) causes poor OCR accuracy. EasyOCR may read mana cost symbols instead of the card name, or produce near-random output. The `OCRRecognizer` mitigates this via (1) cropping out the right 30 % of the title strip to exclude mana cost, and (2) trying both raw and inverted-grayscale preprocessing and keeping the best match. For old-set cards the **hash recognizer** is significantly more reliable — build it with `mtg-scan db build-hashes --sets lea` (Alpha) or the relevant set codes.
- **OCR preprocessing pipeline**: `_extract_title_region()` returns a `width × 70%` crop (mana cost excluded). `_preprocess_variants()` generates [raw, inverted-grayscale] variants. `recognize()` runs OCR on each variant, fuzzy-matches all results, and returns the highest-confidence match. CJK OCR fallback runs only after all Latin variants fail.
- **Tests**: `tests/` use mock detectors, recognizers, and Scryfall clients. The `_build_pipeline_with_mocks` helper in `test_pipeline.py` mocks **both** primary and fallback recognizers — if only primary is mocked, the pipeline builds a real `HashRecognizer` which may match against `data/card_hashes.db` if it exists locally.

---

## Recognition method names

All valid method strings are defined as `RecognitionMethod` in `config.py`:

```python
RecognitionMethod = Literal["ocr", "hash", "clip", "claude", "llm", "paddle"]
```

Use this type everywhere a method name is passed programmatically. Invalid values raise a Pydantic `ValidationError` at startup.
