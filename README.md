# MTG Card Scanner

A Python pipeline for detecting and identifying **Magic: The Gathering** cards
in photographs.  It combines classical computer vision (OpenCV contour
detection), OCR (EasyOCR), perceptual hashing, and optional LLM identification
(GPT-4o) with real-time Scryfall price lookups.

---

## Features

| Stage | Methods |
|-------|---------|
| Detection | OpenCV (contour + perspective correction), YOLOv8 |
| Recognition | OCR title strip (EasyOCR + rapidfuzz), pHash artwork matching, GPT-4o vision |
| Lookup | Scryfall REST API with SQLite cache |
| Output | JSON, CSV, or both |

---

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/yourname/mtg-card-scanner.git
cd mtg-card-scanner

# Create a virtual environment — Python 3.12 required (3.13+ not yet supported by PyTorch)
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install core dependencies
pip install -e .

# Note: if pip installed torch>=2.10, downgrade to the CPU build:
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2"

# Optional extras
pip install -e ".[yolo]"   # YOLOv8 detection
pip install -e ".[llm]"    # GPT-4o recognition
pip install -e ".[dev]"    # development tools
```

### 2. Download card names (required for OCR recognition)

```bash
python scripts/download_card_names.py
```

### 3. Build the hash database (required for hash recognition)

```bash
python scripts/build_hash_db.py --sets m21,lea --limit 1000
```

### 4. Scan a photo

```bash
mtg-scan scan path/to/photo.jpg
mtg-scan scan path/to/folder/ --format csv
```

---

## CLI Reference

```
mtg-scan scan <path> [OPTIONS]
    --output, -o PATH         Output directory
    --format, -f [csv|json|both]
    --detector [opencv|yolo]
    --recognizer [ocr|hash]
    --verbose, -v
    --save-patches

mtg-scan db update-names      Download latest card name list
mtg-scan db build-hashes      Build pHash database
    --sets CODES              Comma-separated set codes
    --limit N                 Max cards to process
mtg-scan db stats             Show DB statistics

mtg-scan config show          Print effective configuration
mtg-scan config init          Write default config.yaml
```

---

## Configuration

Copy `config.yaml` to your working directory (or run `mtg-scan config init`).
The most important settings:

```yaml
detection:
  method: opencv          # or: yolo
  confidence_threshold: 0.5

recognition:
  primary_method: ocr     # or: hash
  fallback_method: hash
  llm_fallback_enabled: false   # set true + OPENAI_API_KEY for GPT-4o

scryfall:
  cache_ttl_hours: 24
  rate_limit_ms: 110
```

Set `MTG_SCANNER_CONFIG=/path/to/config.yaml` to override the default search
path.

---

## Project Structure

```
mtg-card-scanner/
├── src/mtg_scanner/
│   ├── config.py          Pydantic v2 configuration models
│   ├── pipeline.py        Orchestration: detect → recognise → lookup
│   ├── cli.py             Click CLI entry point (mtg-scan)
│   ├── detection/         Card boundary detectors
│   ├── recognition/       OCR / hash / LLM recognisers
│   ├── lookup/            Scryfall client + SQLite cache
│   ├── models/            Data classes (CardPatch, RecognizedCard, ScanResult)
│   └── utils/             Image helpers, fuzzy search
├── scripts/
│   ├── download_card_names.py
│   ├── build_hash_db.py
│   └── generate_training_data.py
├── tests/
├── data/                  Local databases (gitignored except .gitkeep)
├── output/                Scan results
└── config.yaml
```

---

## Generating YOLO Training Data

```bash
python scripts/generate_training_data.py --count 200 --output data/training
```

This creates `data/training/images/train/`, `data/training/labels/train/`, and
`data/training/dataset.yaml` ready for Ultralytics YOLOv8 training.

---

## Running Tests

```bash
pytest
pytest --cov=mtg_scanner --cov-report=term-missing
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `MTG_SCANNER_CONFIG` | Path to a custom `config.yaml` |
| `OPENAI_API_KEY` | Required when `llm_fallback_enabled: true` |

---

## License

MIT
