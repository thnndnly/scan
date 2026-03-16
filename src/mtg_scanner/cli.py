"""Click-based command-line interface for mtg-card-scanner."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="mtg-card-scanner")
def cli() -> None:
    """MTG Card Scanner – Magickarten in Fotos erkennen und identifizieren.

    \b
    Schnellstart:
      mtg-scan scan foto.jpg          Einzelbild scannen
      mtg-scan scan ./fotos/          Ganzen Ordner scannen
      mtg-scan ui                     Weboberfläche starten (localhost:7860)

    \b
    Daten verwalten:
      mtg-scan db update-names        Kartennamen aktualisieren
      mtg-scan db build-hashes        Hash-Datenbank aufbauen
      mtg-scan dataset stats          Scan-Verlauf anzeigen
      mtg-scan archive stats          Bildarchiv-Statistiken

    \b
    Auswertung:
      mtg-scan label set foto.jpg -n 9 --cards "Blitz,Gegenzauber"
      mtg-scan label eval foto.jpg    Scan mit Erwartung vergleichen

    Hilfe zu einem Unterbefehl:  mtg-scan BEFEHL --help
    """


@cli.command("help", hidden=False)
@click.argument("topic", required=False)
@click.pass_context
def help_cmd(ctx: click.Context, topic: str | None) -> None:
    """Hilfe anzeigen. Optionaler TOPIC: scan, db, dataset, archive, label, ui, config, catalog."""
    topics = {
        "scan": """\
SCAN — Bilder nach MTG-Karten durchsuchen

  mtg-scan scan PFAD [OPTIONEN]

  PFAD kann eine einzelne Bilddatei oder ein Ordner sein.
  Unterstützte Formate: JPG, PNG, AVIF, HEIC, WEBP, BMP, TIFF

Optionen:
  -o, --output DIR       Ausgabeordner (Standard: ./output)
  -f, --format FORMAT    Ausgabeformat: csv | json | both  (Standard: both)
  --detector ENGINE      opencv | yolo
  --recognizer METHOD    ocr | hash
  --save-patches         Kartenausschnitte als PNG speichern
  -v, --verbose          Debug-Logging aktivieren

Beispiele:
  mtg-scan scan foto.jpg
  mtg-scan scan ./fotos/ -o ./ergebnisse --format json
  mtg-scan scan foto.jpg --detector opencv --recognizer ocr -v
  mtg-scan scan foto.jpg --save-patches
""",
        "db": """\
DB — Lokale Datenbanken verwalten

  mtg-scan db update-names          Kartennamen von Scryfall herunterladen
  mtg-scan db build-hashes          Perceptual-Hash-DB aufbauen (dauert ~50 Min.)
  mtg-scan db build-hashes --dry-run  Vorschau: Umfang und Zeit schätzen
  mtg-scan db build-hashes --sets m21,lea  Nur bestimmte Sets
  mtg-scan db build-hashes --limit 500     Max. 500 Karten
  mtg-scan db stats                 Statistiken aller lokalen DBs

Dateien:
  data/card_names.json     ~33.000 englische Kartennamen
  data/card_names_de.json  ~28.000 deutsche Kartennamen
  data/card_hashes.db      Perceptual-Hash-DB (~30 MB für alle Sets)
  data/scryfall_cache.db   API-Cache (TTL: 24h)
""",
        "dataset": """\
DATASET — Scan-Verlauf und Telemetrie

  mtg-scan dataset stats             Übersicht aller Scans
  mtg-scan dataset export-csv [PFAD] Alle Ergebnisse als CSV
  mtg-scan dataset export-yolo [DIR] Patches + Boxen als YOLO-Trainingsdaten
  mtg-scan dataset corrections       Manuell korrigierte Karten anzeigen

Datenbank: data/dataset.db
Patches:   data/patches/YYYY-MM-DD/
""",
        "archive": """\
ARCHIVE — Permanentes Bildarchiv

  mtg-scan archive stats           Statistiken (Bilder, Größe, Einsparung)
  mtg-scan archive verify          SHA256-Integrität aller Bilder prüfen
  mtg-scan archive export [PFAD]   Alle Bilder als JPEG exportieren

Jedes gescannte Bild wird automatisch komprimiert (JPEG 70%, max 1920px)
und content-addressed (SHA256) gespeichert — kein Duplikat wird je
zweimal gespeichert.

Datei:  data/image_archive.db      (gitignored)
Index:  data/image_archive_index.json  (in git committen)
""",
        "label": """\
LABEL — Ground-Truth für Auswertung

  mtg-scan label set PFAD -n 9 --cards "Karte1,Karte2,..."
                              Erwartete Karten für ein Bild festlegen
  mtg-scan label show         Alle gespeicherten Labels anzeigen
  mtg-scan label eval PFAD    Scan durchführen und mit Label vergleichen

Ausgabe von eval:
  Erkennungsrate, Precision, Recall, fehlende und falsch erkannte Karten

Datei: data/ground_truth.json
""",
        "ui": """\
UI — Gradio-Weboberfläche

  mtg-scan ui                  Starten auf http://localhost:7860
  mtg-scan ui --port 8080      Anderen Port verwenden
  mtg-scan ui --share          Öffentlichen Link erstellen (über Gradio-Server)

Tabs:
  Scanner          Bild hochladen, Scan starten, Galerie mit Preisen
  Scan-Historie    Vergangene Scans durchsuchen
  Dataset Explorer Niedrig-Konfidenz-Patches manuell korrigieren
  Hash-DB          Hash-Datenbank aufbauen und verwalten
  Auswertung       Ground-Truth Labels setzen und Scans evaluieren
  Archiv           Bildarchiv durchsuchen und exportieren
  Einstellungen    Konfiguration bearbeiten
""",
        "config": """\
CONFIG — Konfiguration verwalten

  mtg-scan config show         Aktuelle Konfiguration als YAML ausgeben
  mtg-scan config init         Standard-config.yaml erstellen

Konfigurationsdatei: config.yaml im Projektverzeichnis
Überschreiben per Umgebungsvariable: MTG_SCANNER_CONFIG=/pfad/config.yaml

Wichtige Einstellungen:
  detection.method             opencv | yolo
  recognition.primary_method  ocr | hash
  recognition.ocr_confidence_threshold  (Standard: 0.70)
  dataset.enabled              Scan-Logging an/aus (Standard: true)
  archive.enabled              Bildarchiv an/aus (Standard: true)
  archive.jpeg_quality         Komprimierungsqualität 10-95 (Standard: 70)
  catalog.db_path              Pfad zur Karten-Katalog-DB
""",
        "collection": """\
COLLECTION — Persönliche Kartensammlung verwalten

  mtg-scan collection stats                    Statistiken anzeigen
  mtg-scan collection list                     Alle Karten auflisten
  mtg-scan collection list --name "Bolt"       Nach Name filtern
  mtg-scan collection list --set m21           Nach Set filtern
  mtg-scan collection add --scryfall-id UUID --name "Lightning Bolt" --set-code lea
  mtg-scan collection add ... --foil --qty 2 --condition LP --buy-price 4.50
  mtg-scan collection remove 42                Eintrag #42 entfernen
  mtg-scan collection export                   Generisches CSV
  mtg-scan collection export --format moxfield output/moxfield.csv
  mtg-scan collection export --format tcgplayer
  mtg-scan collection export --format cardmarket
  mtg-scan collection export --format arena output/arena.dek

Zustände: NM (Near Mint), LP (Lightly Played), MP (Moderately Played),
          HP (Heavily Played), DMG (Damaged)

Datenbank: data/collection.db
""",
        "catalog": """\
CATALOG — Lokaler Karten-Katalog (Scryfall Bulk-Daten)

  mtg-scan catalog build         Katalog herunterladen (~155 MB, einmalig)
  mtg-scan catalog build --force Erzwinge Neu-Download
  mtg-scan catalog build --check Nur Aktualität prüfen
  mtg-scan catalog stats         Statistiken anzeigen
  mtg-scan catalog search NAME   Nach Karte suchen

Der Katalog enthält ~130.000 Karten-Drucke mit vollständigen Metadaten
(Set, Nummer, Seltenheit, Preise, Finish, Künstler, usw.) aber keine Bilder.
Bilder werden per URL geladen wenn benötigt.

Datenbank: data/card_catalog.db  (gitignored)
""",
    }

    if topic and topic.lower() in topics:
        click.echo(topics[topic.lower()])
    else:
        root_ctx = ctx.find_root()
        click.echo(root_ctx.get_help())
        click.echo("\nDetaillierte Hilfe zu einem Thema:")
        for t in topics:
            click.echo(f"  mtg-scan help {t}")


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------


@cli.command("scan", context_settings=CONTEXT_SETTINGS)
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output directory (overrides config).")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["csv", "json", "both"], case_sensitive=False),
    default=None,
    help="Output format (default: both).",
)
@click.option(
    "--detector",
    type=click.Choice(["opencv", "yolo"], case_sensitive=False),
    default=None,
    help="Detection backend.",
)
@click.option(
    "--recognizer",
    type=click.Choice(["ocr", "hash"], case_sensitive=False),
    default=None,
    help="Primary recognition method.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.option(
    "--save-patches", is_flag=True, default=False, help="Save individual card patch images."
)
def scan_cmd(
    path: str,
    output: str | None,
    fmt: str | None,
    detector: str | None,
    recognizer: str | None,
    verbose: bool,
    save_patches: bool,
) -> None:
    """Bild oder Ordner nach MTG-Karten durchsuchen.

    \b
    PATH kann eine einzelne Bilddatei oder ein Ordner sein.
    Unterstützte Formate: JPG, PNG, AVIF, HEIC, WEBP, BMP, TIFF

    \b
    Beispiele:
      mtg-scan scan foto.jpg
      mtg-scan scan ./fotos/ -o ./ergebnisse
      mtg-scan scan foto.jpg --format json -v
      mtg-scan scan foto.jpg --detector opencv --recognizer ocr
      mtg-scan scan foto.jpg --save-patches
    """
    _setup_logging(verbose)

    from mtg_scanner.config import get_config
    from mtg_scanner.pipeline import Pipeline

    cfg = get_config()
    out_dir = output or cfg.output.output_dir
    out_fmt = fmt or cfg.output.default_format

    # Build detector
    det_instance = None
    if detector == "yolo":
        from mtg_scanner.detection.yolo_detector import YOLODetector

        det_instance = YOLODetector()
    elif detector == "opencv":
        from mtg_scanner.detection.opencv_detector import OpenCVDetector

        det_instance = OpenCVDetector()

    # Build recognizer
    rec_instance = None
    if recognizer == "hash":
        from mtg_scanner.recognition.hash_recognizer import HashRecognizer

        rec_instance = HashRecognizer()
    elif recognizer == "ocr":
        from mtg_scanner.recognition.ocr_recognizer import OCRRecognizer

        rec_instance = OCRRecognizer()

    pipeline = Pipeline(
        detector=det_instance,
        primary_recognizer=rec_instance,
        save_patches=save_patches or cfg.output.save_card_patches,
        output_dir=out_dir,
    )

    p = Path(path)
    if p.is_dir():
        def _progress(current: int, total: int, img_path: str) -> None:
            click.echo(f"[{current}/{total}] {img_path}")

        results = pipeline.process_directory(str(p), progress_callback=_progress)
    else:
        results = [pipeline.process_image(str(p))]

    written = pipeline.save_results(results, output_dir=out_dir, fmt=out_fmt)

    for result in results:
        click.echo(result.summary())

    if written:
        click.echo("\nOutput files:")
        for f in written:
            click.echo(f"  {f}")


# ---------------------------------------------------------------------------
# db command group
# ---------------------------------------------------------------------------


@cli.group("db", context_settings=CONTEXT_SETTINGS)
def db_group() -> None:
    """Lokale Datenbanken verwalten (Kartennamen, Hash-DB, Cache).

    \b
    Befehle:
      update-names   Kartennamen von Scryfall herunterladen
      build-hashes   Perceptual-Hash-DB aufbauen
      stats          Statistiken aller lokalen DBs
    """


@db_group.command("help", hidden=False)
@click.pass_context
def db_help(ctx: click.Context) -> None:
    """Hilfe zu db-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@db_group.command("update-names")
@click.option("--verbose", "-v", is_flag=True)
def db_update_names(verbose: bool) -> None:
    """Download the latest card name list from MTG JSON."""
    _setup_logging(verbose)
    click.echo("Downloading card names from MTG JSON…")
    try:
        import subprocess

        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        script = str(scripts_dir / "download_card_names.py")
        result = subprocess.run([sys.executable, script], check=False)
        sys.exit(result.returncode)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@db_group.command("build-hashes")
@click.option("--sets", default=None, help="Comma-separated set codes to include.")
@click.option("--limit", default=None, type=int, help="Maximum number of cards to process.")
@click.option("--dry-run", is_flag=True, default=False, help="Preview scope and time estimate without downloading.")
@click.option("--verbose", "-v", is_flag=True)
def db_build_hashes(sets: str | None, limit: int | None, dry_run: bool, verbose: bool) -> None:
    """Build the perceptual hash database from Scryfall card images."""
    _setup_logging(verbose)
    click.echo("Building hash database…")
    try:
        import subprocess

        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        script = str(scripts_dir / "build_hash_db.py")
        args = [sys.executable, script]
        if sets:
            args += ["--sets", sets]
        if limit is not None:
            args += ["--limit", str(limit)]
        if dry_run:
            args += ["--dry-run"]
        result = subprocess.run(args, check=False)
        sys.exit(result.returncode)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@db_group.command("stats")
@click.option("--verbose", "-v", is_flag=True)
def db_stats(verbose: bool) -> None:
    """Show statistics for the local databases."""
    _setup_logging(verbose)

    from mtg_scanner.config import get_config
    from mtg_scanner.lookup.cache import ScryfallCache

    cfg = get_config()

    # Scryfall cache stats
    cache = ScryfallCache(
        db_path=cfg.scryfall.cache_db_path, ttl_hours=cfg.scryfall.cache_ttl_hours
    )
    stats = cache.stats()
    click.echo("Scryfall cache:")
    click.echo(f"  Total entries  : {stats['total_entries']}")
    click.echo(f"  Expired entries: {stats['expired_entries']}")

    # Hash DB stats
    hash_db = cfg.recognition.hash_db_path
    if Path(hash_db).exists():
        import sqlite3

        conn = sqlite3.connect(hash_db)
        count = conn.execute("SELECT COUNT(*) FROM card_hashes;").fetchone()[0]
        conn.close()
        click.echo(f"\nHash DB ({hash_db}):")
        click.echo(f"  Total hashes: {count}")
    else:
        click.echo(f"\nHash DB not found at {hash_db}.")

    # Card names
    names_file = Path("data/card_names.json")
    if names_file.exists():
        import json as _json

        with open(names_file, encoding="utf-8") as fh:
            data = _json.load(fh)
        n = len(data) if isinstance(data, list) else len(data.get("data", []))
        click.echo(f"\nCard names (data/card_names.json): {n} entries")
    else:
        click.echo("\nCard names file not found (data/card_names.json).")

    # CLIP embedding DB
    import sqlite3 as _sqlite3
    clip_db = Path("data/clip_embeddings.db")
    if clip_db.exists():
        conn = _sqlite3.connect(str(clip_db))
        count = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
        conn.close()
        click.echo(f"\nCLIP embedding DB ({clip_db}): {count} embeddings")
    else:
        click.echo(f"\nCLIP embedding DB not found (run: mtg-scan db build-clip).")


@db_group.command("build-clip")
@click.option("--sets", default=None, help="Comma-separated Scryfall set codes (e.g. m21,lea).")
@click.option("--limit", default=None, type=int, help="Maximum number of cards to embed.")
@click.option("--lang", multiple=True, default=["en"], show_default=True, help="Language codes.")
@click.option("--db-path", default="data/clip_embeddings.db", show_default=True)
@click.option("--dry-run", is_flag=True, help="Show stats without downloading.")
@click.option("--verbose", "-v", is_flag=True)
def db_build_clip(
    sets: str | None,
    limit: int | None,
    lang: tuple[str, ...],
    db_path: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Build the CLIP embedding database for artwork-based recognition."""
    _setup_logging(verbose)
    try:
        import subprocess

        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        script = str(scripts_dir / "build_clip_db.py")
        args = [sys.executable, script, "--db-path", db_path]
        if sets:
            args += ["--sets", sets]
        if limit is not None:
            args += ["--limit", str(limit)]
        for lc in lang:
            args += ["--lang", lc]
        if dry_run:
            args.append("--dry-run")
        result = subprocess.run(args, check=False)
        sys.exit(result.returncode)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@db_group.command("train-yolo")
@click.option("--base-model", default="yolov8n.pt", show_default=True)
@click.option("--epochs", default=100, show_default=True, type=int)
@click.option("--imgsz", default=640, show_default=True, type=int)
@click.option("--batch", default=16, show_default=True, type=int)
@click.option("--output-model", default="data/models/yolo_mtg.pt", show_default=True)
@click.option("--dataset-db", default="data/dataset.db", show_default=True)
@click.option("--val-split", default=0.15, show_default=True, type=float)
@click.option("--min-confidence", default=0.0, show_default=True, type=float)
@click.option("--dry-run", is_flag=True, help="Show dataset stats without training.")
@click.option("--verbose", "-v", is_flag=True)
def db_train_yolo(
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    output_model: str,
    dataset_db: str,
    val_split: float,
    min_confidence: float,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Fine-tune YOLOv8 on validated patches from the dataset DB."""
    _setup_logging(verbose)
    try:
        import subprocess

        scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
        script = str(scripts_dir / "train_yolo.py")
        args = [
            sys.executable, script,
            "--base-model", base_model,
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--output-model", output_model,
            "--dataset-db", dataset_db,
            "--val-split", str(val_split),
            "--min-confidence", str(min_confidence),
        ]
        if dry_run:
            args.append("--dry-run")
        result = subprocess.run(args, check=False)
        sys.exit(result.returncode)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# config command group
# ---------------------------------------------------------------------------


@cli.group("config", context_settings=CONTEXT_SETTINGS)
def config_group() -> None:
    """Konfiguration verwalten.

    \b
    Befehle:
      show   Aktuelle Konfiguration als YAML ausgeben
      init   Standard-config.yaml erstellen
    """


@config_group.command("help", hidden=False)
@click.pass_context
def config_help(ctx: click.Context) -> None:
    """Hilfe zu config-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@config_group.command("show")
def config_show() -> None:
    """Print the current effective configuration as YAML."""
    from mtg_scanner.config import get_config

    cfg = get_config()
    click.echo(yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False))


@config_group.command("init")
@click.option(
    "--output",
    "-o",
    default="config.yaml",
    show_default=True,
    help="Destination path for the new config file.",
)
def config_init(output: str) -> None:
    """Write a default config.yaml to the current directory."""
    dest = Path(output)
    if dest.exists():
        if not click.confirm(f"{dest} already exists. Overwrite?", default=False):
            click.echo("Aborted.")
            return

    from mtg_scanner.config import AppConfig

    cfg = AppConfig()
    with open(dest, "w", encoding="utf-8") as fh:
        yaml.dump(cfg.model_dump(), fh, default_flow_style=False, sort_keys=False)

    click.echo(f"Default configuration written to {dest}")


# ---------------------------------------------------------------------------
# dataset command group
# ---------------------------------------------------------------------------


@cli.group("dataset", context_settings=CONTEXT_SETTINGS)
def dataset_group() -> None:
    """Scan-Verlauf und Telemetrie verwalten.

    \b
    Befehle:
      stats          Übersicht aller Scans
      export-csv     Alle Ergebnisse als CSV exportieren
      export-yolo    Patches als YOLO-Trainingsdaten exportieren
      corrections    Manuell korrigierte Karten anzeigen
    """


@dataset_group.command("help", hidden=False)
@click.pass_context
def dataset_help(ctx: click.Context) -> None:
    """Hilfe zu dataset-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@dataset_group.command("stats")
@click.option("--verbose", "-v", is_flag=True)
def dataset_stats(verbose: bool) -> None:
    """Sammlung-Übersicht: Statistiken über alle gespeicherten Scans."""
    _setup_logging(verbose)

    from mtg_scanner.config import get_config
    from mtg_scanner.dataset import DatasetLogger

    cfg = get_config()
    if not cfg.dataset.enabled:
        click.echo("Dataset-Logger ist deaktiviert (dataset.enabled: false in config.yaml).")
        return

    db_path = cfg.dataset.db_path
    if not Path(db_path).exists():
        click.echo(f"Keine Datenbank gefunden unter: {db_path}")
        click.echo("Führen Sie zunächst einen Scan durch, um Daten zu erfassen.")
        return

    dl = DatasetLogger(db_path=db_path, save_patches=cfg.dataset.save_patches)
    s = dl.stats()
    dl.close()

    click.echo("Dataset-Statistiken:")
    click.echo(f"  Scans gesamt       : {s['total_scans']}")
    click.echo(f"  Erkennungen gesamt : {s['total_detections']}")
    click.echo(f"  Identifiziert      : {s['total_recognised']}")
    click.echo(f"  Unbekannt          : {s['total_unknown']}")
    click.echo(f"  Korrekturen        : {s['total_corrections']}")
    click.echo(f"  Gesamtwert (EUR)   : €{s['total_value_eur']:.2f}")
    click.echo(f"  Datenbank          : {s['db_path']}")


@dataset_group.command("export-csv")
@click.argument("path", default="output/dataset_export.csv", required=False)
@click.option("--verbose", "-v", is_flag=True)
def dataset_export_csv(path: str, verbose: bool) -> None:
    """Alle Ergebnisse als CSV exportieren."""
    _setup_logging(verbose)

    from mtg_scanner.config import get_config
    from mtg_scanner.dataset import DatasetLogger

    cfg = get_config()
    db_path = cfg.dataset.db_path
    if not Path(db_path).exists():
        click.echo(f"Keine Datenbank gefunden unter: {db_path}", err=True)
        sys.exit(1)

    dl = DatasetLogger(db_path=db_path, save_patches=False)
    count = dl.export_csv(path)
    dl.close()
    click.echo(f"{count} Zeilen exportiert nach: {path}")


@dataset_group.command("export-yolo")
@click.argument("directory", default="output/yolo_training", required=False)
@click.option("--verbose", "-v", is_flag=True)
def dataset_export_yolo(directory: str, verbose: bool) -> None:
    """Patches + Bounding-Boxen als YOLO-Trainingsdaten exportieren."""
    _setup_logging(verbose)

    from mtg_scanner.config import get_config
    from mtg_scanner.dataset import DatasetLogger

    cfg = get_config()
    db_path = cfg.dataset.db_path
    if not Path(db_path).exists():
        click.echo(f"Keine Datenbank gefunden unter: {db_path}", err=True)
        sys.exit(1)

    dl = DatasetLogger(db_path=db_path, save_patches=False)
    out_dir = Path(directory)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT d.id, d.patch_image_path, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
               s.image_width, s.image_height
        FROM detections d
        JOIN scans s ON s.id = d.scan_id
        WHERE d.patch_image_path IS NOT NULL
        """
    ).fetchall()
    conn.close()
    dl.close()

    exported = 0
    for row in rows:
        patch_path = Path(row["patch_image_path"])
        if not patch_path.exists():
            continue
        dest_img = images_dir / patch_path.name
        shutil.copy2(str(patch_path), str(dest_img))

        # YOLO label: class 0 = MTG card, bbox normalized to image size
        iw = row["image_width"] or 1
        ih = row["image_height"] or 1
        x_c = (row["bbox_x"] + row["bbox_w"] / 2) / iw
        y_c = (row["bbox_y"] + row["bbox_h"] / 2) / ih
        bw = row["bbox_w"] / iw
        bh = row["bbox_h"] / ih
        label_file = labels_dir / (patch_path.stem + ".txt")
        with open(label_file, "w") as lf:
            lf.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
        exported += 1

    click.echo(f"{exported} Patches nach {directory} exportiert.")


@dataset_group.command("corrections")
@click.option("--verbose", "-v", is_flag=True)
def dataset_corrections(verbose: bool) -> None:
    """Zeige alle manuell korrigierten Karten."""
    _setup_logging(verbose)

    from mtg_scanner.config import get_config
    from mtg_scanner.dataset import DatasetLogger

    cfg = get_config()
    db_path = cfg.dataset.db_path
    if not Path(db_path).exists():
        click.echo(f"Keine Datenbank gefunden unter: {db_path}", err=True)
        sys.exit(1)

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT r.detection_id, r.card_name, r.corrected_name,
               d.patch_index, s.image_path, s.scan_timestamp
        FROM results r
        JOIN detections d ON d.id = r.detection_id
        JOIN scans s ON s.id = d.scan_id
        WHERE r.manually_corrected = 1
        ORDER BY s.scan_timestamp DESC
        """
    ).fetchall()
    conn.close()

    if not rows:
        click.echo("Keine manuellen Korrekturen vorhanden.")
        return

    click.echo(f"{len(rows)} Korrektur(en):")
    for row in rows:
        click.echo(
            f"  [{row['scan_timestamp'][:10]}] {row['image_path']}"
            f" patch#{row['patch_index']}: "
            f"'{row['card_name']}' → '{row['corrected_name']}'"
        )


# ---------------------------------------------------------------------------
# label command group
# ---------------------------------------------------------------------------


@cli.group("label", context_settings=CONTEXT_SETTINGS)
def label_group() -> None:
    """Ground-Truth Labels setzen und Scan-Genauigkeit auswerten.

    \b
    Befehle:
      set    Erwartete Karten für ein Bild festlegen
      show   Alle gespeicherten Labels anzeigen
      eval   Scan mit gespeichertem Label vergleichen
    """


@label_group.command("help", hidden=False)
@click.pass_context
def label_help(ctx: click.Context) -> None:
    """Hilfe zu label-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@label_group.command("set")
@click.argument("path", type=click.Path(exists=True))
@click.option("--count", "-n", default=None, type=int, help="Expected number of cards.")
@click.option("--cards", "-c", default=None, help="Comma-separated expected card names.")
@click.option("--verbose", "-v", is_flag=True)
def label_set(path: str, count: int | None, cards: str | None, verbose: bool) -> None:
    """Set expected card count/names for an image (ground truth)."""
    _setup_logging(verbose)
    from mtg_scanner.evaluation import label_image

    card_list = [c.strip() for c in cards.split(",")] if cards else None
    label_image(str(Path(path).resolve()), expected_count=count, expected_cards=card_list)
    click.echo(f"Ground truth saved for: {path}")
    if count:
        click.echo(f"  Expected count: {count}")
    if card_list:
        click.echo(f"  Expected cards: {', '.join(card_list)}")


@label_group.command("show")
@click.option("--verbose", "-v", is_flag=True)
def label_show(verbose: bool) -> None:
    """Show all ground-truth labels."""
    _setup_logging(verbose)
    from mtg_scanner.evaluation import load_ground_truth

    data = load_ground_truth()
    if not data:
        click.echo("No ground truth labels found (data/ground_truth.json).")
        return
    for img_path, entry in data.items():
        click.echo(f"\n{img_path}")
        if "expected_count" in entry:
            click.echo(f"  Expected count : {entry['expected_count']}")
        if "expected_cards" in entry:
            click.echo(f"  Expected cards : {', '.join(entry['expected_cards'])}")


@label_group.command("eval")
@click.argument("path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def label_eval(path: str, verbose: bool) -> None:
    """Scan an image and compare results to its ground truth label."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.evaluation import evaluate_scan
    from mtg_scanner.pipeline import Pipeline

    cfg = get_config()
    pipeline = Pipeline()
    result = pipeline.process_image(str(Path(path).resolve()))

    report = evaluate_scan(result)

    click.echo(f"\nEvaluation: {path}")
    click.echo(f"  Detected      : {report['detected_count']}"
               + (f" / {report['expected_count']} expected" if report['expected_count'] else ""))
    click.echo(f"  Recognised    : {report['recognised_count']}")
    if report['detection_rate'] is not None:
        click.echo(f"  Detection rate: {report['detection_rate']:.0%}")
    if report['expected_cards']:
        click.echo(f"  Matched       : {len(report['matched'])} / {len(report['expected_cards'])}")
        click.echo(f"  Precision     : {report['precision']:.0%}")
        click.echo(f"  Recall        : {report['recall']:.0%}")
        if report['missed']:
            click.echo(f"  Missed cards  : {', '.join(report['missed'])}")
        if report['extra']:
            click.echo(f"  Extra (wrong) : {', '.join(report['extra'])}")
    if report['detected_cards']:
        click.echo(f"  Detected cards: {', '.join(report['detected_cards'])}")


# ---------------------------------------------------------------------------
# archive command group
# ---------------------------------------------------------------------------


@cli.group("archive", context_settings=CONTEXT_SETTINGS)
def archive_group() -> None:
    """Permanentes komprimiertes Bildarchiv verwalten.

    \b
    Befehle:
      stats    Statistiken (Bilder, Größe, Einsparung)
      verify   SHA256-Integrität aller Bilder prüfen
      export   Alle Bilder als JPEG in einen Ordner exportieren
    """


@archive_group.command("help", hidden=False)
@click.pass_context
def archive_help(ctx: click.Context) -> None:
    """Hilfe zu archive-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@archive_group.command("stats")
@click.option("--verbose", "-v", is_flag=True)
def archive_stats(verbose: bool) -> None:
    """Show image archive statistics."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.image_archive import ImageArchive

    cfg = get_config()
    if not Path(cfg.archive.db_path).exists():
        click.echo("Kein Archiv gefunden. Führe zunächst einen Scan durch.")
        return
    arch = ImageArchive(
        db_path=cfg.archive.db_path,
        index_path=cfg.archive.index_path,
    )
    s = arch.stats()
    arch.close()
    click.echo("Bild-Archiv Statistiken:")
    click.echo(f"  Gespeicherte Bilder  : {s['total_images']}")
    click.echo(f"  Originalgröße        : {s['original_size_mb']:.1f} MB")
    click.echo(f"  Gespeichert (komprim.): {s['stored_size_mb']:.1f} MB")
    click.echo(f"  Komprimierungsrate   : {s['compression_ratio']:.0%}")
    click.echo(f"  Einsparung           : {s['space_saved_mb']:.1f} MB")
    click.echo(f"  Datenbank            : {s['db_path']}")


@archive_group.command("export")
@click.argument("dest", default="output/archive_export", required=False)
@click.option("--verbose", "-v", is_flag=True)
def archive_export(dest: str, verbose: bool) -> None:
    """Export all archived images as JPEG files to DEST directory."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.image_archive import ImageArchive

    cfg = get_config()
    if not Path(cfg.archive.db_path).exists():
        click.echo("Kein Archiv gefunden.", err=True)
        sys.exit(1)
    arch = ImageArchive(db_path=cfg.archive.db_path, index_path=cfg.archive.index_path)
    count = arch.export_all(dest)
    arch.close()
    click.echo(f"{count} Bilder exportiert nach: {dest}")


@archive_group.command("verify")
@click.option("--verbose", "-v", is_flag=True)
def archive_verify(verbose: bool) -> None:
    """Verify integrity of all archived images."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.image_archive import ImageArchive

    cfg = get_config()
    if not Path(cfg.archive.db_path).exists():
        click.echo("Kein Archiv gefunden.", err=True)
        sys.exit(1)
    arch = ImageArchive(db_path=cfg.archive.db_path, index_path=cfg.archive.index_path)
    ok, corrupt = arch.verify()
    arch.close()
    click.echo(f"Integrität: {ok} OK, {corrupt} beschädigt")


# ---------------------------------------------------------------------------
# catalog command group
# ---------------------------------------------------------------------------


@cli.group("catalog", context_settings=CONTEXT_SETTINGS)
def catalog_group() -> None:
    """Lokalen Karten-Katalog verwalten (Scryfall Bulk-Daten).

    \b
    Befehle:
      build    Katalog herunterladen und aufbauen (~155 MB)
      stats    Statistiken anzeigen
      search   Nach Karten suchen
    """


@catalog_group.command("help", hidden=False)
@click.pass_context
def catalog_help(ctx: click.Context) -> None:
    """Hilfe zu catalog-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@catalog_group.command("build")
@click.option("--force", is_flag=True, help="Neu laden auch wenn aktuell.")
@click.option("--check", is_flag=True, help="Nur Aktualität prüfen, kein Download.")
@click.option(
    "--bulk-type",
    default=None,
    type=click.Choice(["oracle_cards", "default_cards", "all_cards"]),
    help="Bulk-Datentyp (Standard: aus config.yaml).",
)
@click.option("--verbose", "-v", is_flag=True)
def catalog_build(force: bool, check: bool, bulk_type: str | None, verbose: bool) -> None:
    """Karten-Katalog von Scryfall herunterladen und aufbauen.

    \b
    Beim ersten Aufruf: ~155 MB Download, ~2 Minuten Import.
    Folgeaufrufe: nur neu laden wenn Scryfall eine neue Version hat.
    """
    _setup_logging(verbose)
    import subprocess

    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    script = str(scripts_dir / "build_card_catalog.py")

    from mtg_scanner.config import get_config
    cfg = get_config()
    bt = bulk_type or cfg.catalog.bulk_type

    args_list = [sys.executable, script, "--bulk-type", bt, "--db", cfg.catalog.db_path]
    if force:
        args_list.append("--force")
    if check:
        args_list.append("--check")

    result = subprocess.run(args_list, check=False)
    sys.exit(result.returncode)


@catalog_group.command("stats")
@click.option("--verbose", "-v", is_flag=True)
def catalog_stats(verbose: bool) -> None:
    """Statistiken des lokalen Katalogs anzeigen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.lookup.card_catalog import CardCatalog

    cfg = get_config()
    if not Path(cfg.catalog.db_path).exists():
        click.echo("Kein Katalog gefunden. Aufbauen mit: mtg-scan catalog build")
        return
    cat = CardCatalog(db_path=cfg.catalog.db_path)
    s = cat.stats()
    cat.close()
    click.echo("Karten-Katalog:")
    click.echo(f"  Karten gesamt : {s['total_cards']:,}")
    click.echo(f"  Sets          : {s['total_sets']:,}")
    click.echo(f"  Bulk-Typ      : {s['bulk_type']}")
    click.echo(f"  Scryfall Stand: {s['updated_at']}")
    click.echo(f"  Importiert am : {s['imported_at']}")
    click.echo(f"  Datenbank     : {s['db_path']}")


@catalog_group.command("search")
@click.argument("name")
@click.option("--limit", default=20, show_default=True, help="Maximale Treffer.")
@click.option("--verbose", "-v", is_flag=True)
def catalog_search(name: str, limit: int, verbose: bool) -> None:
    """Nach Karten im lokalen Katalog suchen.

    \b
    Beispiele:
      mtg-scan catalog search "Lightning Bolt"
      mtg-scan catalog search "Blitz" --limit 5
    """
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.lookup.card_catalog import CardCatalog

    cfg = get_config()
    if not Path(cfg.catalog.db_path).exists():
        click.echo("Kein Katalog gefunden. Aufbauen mit: mtg-scan catalog build")
        sys.exit(1)
    cat = CardCatalog(db_path=cfg.catalog.db_path)
    results = cat.search_by_name(name, limit=limit)
    cat.close()

    if not results:
        click.echo(f"Keine Treffer für '{name}'.")
        return

    click.echo(f"{len(results)} Treffer für '{name}':\n")
    for c in results:
        prices = c.get("prices") or {}
        eur = prices.get("eur") or "—"
        eur_foil = prices.get("eur_foil") or "—"
        finishes = ", ".join(c.get("finishes") or [])
        click.echo(
            f"  [{c['set_code'].upper():4}] {c['collector_number']:>4}  "
            f"{c['name']:<40}  "
            f"{c.get('rarity','?'):<8}  "
            f"EUR {eur:>6} / Foil {eur_foil:>6}  "
            f"({finishes})  "
            f"{c.get('released_at','?')[:4]}"
        )


# ---------------------------------------------------------------------------
# collection command group
# ---------------------------------------------------------------------------


@cli.group("collection", context_settings=CONTEXT_SETTINGS)
def collection_group() -> None:
    """Eigene Kartensammlung verwalten.

    \b
    Befehle:
      stats    Statistiken der Sammlung
      list     Karten auflisten (optional gefiltert)
      add      Karte zur Sammlung hinzufügen
      remove   Eintrag aus der Sammlung entfernen
      export   Sammlung exportieren (Moxfield, TCGplayer, Cardmarket, Arena, CSV)
    """


@collection_group.command("help", hidden=False)
@click.pass_context
def collection_help(ctx: click.Context) -> None:
    """Hilfe zu collection-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@collection_group.command("stats")
@click.option("--verbose", "-v", is_flag=True)
def collection_stats(verbose: bool) -> None:
    """Statistiken der Kartensammlung anzeigen."""
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config

    cfg = get_config()
    db_path = cfg.collection.db_path
    col = CollectionManager(db_path=db_path)
    s = col.stats()
    col.close()

    click.echo("Sammlungs-Statistiken:")
    click.echo(f"  Einträge gesamt    : {s['total_entries']}")
    click.echo(f"  Karten gesamt      : {s['total_cards']}")
    click.echo(f"  Einzigartige Karten: {s['unique_cards']}")
    click.echo(f"  Davon Foil         : {s['foil_count']}")
    click.echo(f"  Kaufwert gesamt    : €{s['total_value_eur']:.2f}")
    click.echo(f"  Marktwert (EUR)    : €{s['market_value_eur']:.2f}")
    click.echo(f"  Preishistorie-Tage : {s['price_history_days']}")
    if s["by_condition"]:
        click.echo("  Nach Zustand:")
        for cond, cnt in sorted(s["by_condition"].items()):
            click.echo(f"    {cond:<4}: {cnt}")
    click.echo(f"  Datenbank          : {s['db_path']}")


@collection_group.command("list")
@click.option("--name", default="", help="Namensfilter (Teilstring).")
@click.option("--set", "set_code", default="", help="Set-Code-Filter (z.B. m21).")
@click.option("--limit", default=100, show_default=True, help="Maximale Treffer.")
@click.option("--verbose", "-v", is_flag=True)
def collection_list(name: str, set_code: str, limit: int, verbose: bool) -> None:
    """Karten der Sammlung auflisten."""
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config

    cfg = get_config()
    col = CollectionManager(db_path=cfg.collection.db_path)
    entries = col.get_collection(name_filter=name, set_filter=set_code, limit=limit)
    col.close()

    if not entries:
        click.echo("Keine Einträge gefunden.")
        return

    click.echo(f"{'ID':>4}  {'Qty':>3}  {'Cond':<4}  {'Foil':<4}  "
               f"{'Set':>4}  {'Nr.':>4}  Name")
    click.echo("—" * 72)
    for e in entries:
        foil = "✓" if e["foil"] else ""
        click.echo(
            f"{e['id']:>4}  {e['quantity']:>3}x  {e['condition']:<4}  "
            f"{foil:<4}  {(e['set_code'] or '').upper():>4}  "
            f"{(e['collector_number'] or ''):>4}  {e['name']}"
        )
    click.echo(f"\n{len(entries)} Einträge.")


@collection_group.command("add")
@click.option("--scryfall-id", required=True, help="Scryfall-UUID der Karte.")
@click.option("--name", required=True, help="Kartenname.")
@click.option("--oracle-id", default="", help="Oracle-ID.")
@click.option("--set-code", default="", help="Set-Code (z.B. m21).")
@click.option("--set-name", default="", help="Set-Name.")
@click.option("--collector-number", default="", help="Collector-Nummer.")
@click.option("--lang", default="en", show_default=True, help="Sprachcode.")
@click.option("--foil", is_flag=True, default=False, help="Foil-Karte.")
@click.option(
    "--condition",
    default="NM",
    show_default=True,
    type=click.Choice(["NM", "LP", "MP", "HP", "DMG"], case_sensitive=False),
    help="Zustand.",
)
@click.option("--qty", default=1, show_default=True, help="Anzahl.")
@click.option("--buy-price", default=None, type=float, help="Kaufpreis in EUR.")
@click.option("--buy-date", default=None, help="Kaufdatum (YYYY-MM-DD).")
@click.option("--notes", default="", help="Notizen.")
@click.option("--verbose", "-v", is_flag=True)
def collection_add(
    scryfall_id: str,
    name: str,
    oracle_id: str,
    set_code: str,
    set_name: str,
    collector_number: str,
    lang: str,
    foil: bool,
    condition: str,
    qty: int,
    buy_price: float | None,
    buy_date: str | None,
    notes: str,
    verbose: bool,
) -> None:
    """Karte zur Sammlung hinzufügen (oder Anzahl erhöhen wenn bereits vorhanden)."""
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config

    cfg = get_config()
    col = CollectionManager(db_path=cfg.collection.db_path)
    entry_id = col.add_card(
        scryfall_id=scryfall_id,
        name=name,
        oracle_id=oracle_id,
        set_code=set_code,
        set_name=set_name,
        collector_number=collector_number,
        lang=lang,
        foil=foil,
        condition=condition,
        quantity=qty,
        buy_price=buy_price,
        buy_date=buy_date,
        notes=notes,
    )
    col.close()
    foil_str = " (Foil)" if foil else ""
    click.echo(f"Hinzugefügt: {qty}x {name}{foil_str} [{condition}] → Eintrag #{entry_id}")


@collection_group.command("remove")
@click.argument("entry_id", type=int)
@click.option("--verbose", "-v", is_flag=True)
def collection_remove(entry_id: int, verbose: bool) -> None:
    """Eintrag aus der Sammlung entfernen (anhand der Eintrag-ID)."""
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config

    cfg = get_config()
    col = CollectionManager(db_path=cfg.collection.db_path)
    entry = col.get_entry(entry_id)
    if entry is None:
        click.echo(f"Eintrag #{entry_id} nicht gefunden.", err=True)
        col.close()
        sys.exit(1)
    ok = col.remove_card(entry_id)
    col.close()
    if ok:
        click.echo(f"Eintrag #{entry_id} ({entry['name']}) entfernt.")
    else:
        click.echo(f"Eintrag #{entry_id} konnte nicht entfernt werden.", err=True)
        sys.exit(1)


@collection_group.command("price-update")
@click.option("--verbose", "-v", is_flag=True)
def collection_price_update(verbose: bool) -> None:
    """Aktuelle Preise für alle Sammlungskarten aus dem lokalen Katalog holen.

    \b
    Benötigt: mtg-scan catalog build (einmalig)
    Schreibt täglich einen neuen Preis-Datenpunkt pro Karte.
    """
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config
    from mtg_scanner.lookup.card_catalog import CardCatalog

    cfg = get_config()
    if not Path(cfg.catalog.db_path).exists():
        click.echo("Kein Katalog gefunden. Bitte zuerst: mtg-scan catalog build", err=True)
        sys.exit(1)

    col = CollectionManager(db_path=cfg.collection.db_path)
    cat = CardCatalog(db_path=cfg.catalog.db_path)
    click.echo("Preise werden aktualisiert…")
    inserted = col.update_prices_from_catalog(cat)
    cat.close()
    s = col.stats()
    col.close()
    click.echo(f"{inserted} neue Preisdatenpunkte gespeichert.")
    click.echo(f"Aktueller Marktwert (EUR): €{s['market_value_eur']:.2f}")


@collection_group.command("export")
@click.argument("path", default=None, required=False)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["moxfield", "tcgplayer", "cardmarket", "arena", "csv"],
                      case_sensitive=False),
    default="csv",
    show_default=True,
    help="Exportformat.",
)
@click.option("--verbose", "-v", is_flag=True)
def collection_export(path: str | None, fmt: str, verbose: bool) -> None:
    """Sammlung exportieren.

    \b
    Formate:
      moxfield   CSV für Moxfield / Archidekt
      tcgplayer  CSV für TCGplayer
      cardmarket CSV für Cardmarket
      arena      MTG-Arena-Format (.dek)
      csv        Generisches CSV (Standard)

    \b
    Beispiele:
      mtg-scan collection export --format moxfield output/moxfield.csv
      mtg-scan collection export --format arena output/arena.dek
    """
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config

    cfg = get_config()
    col = CollectionManager(db_path=cfg.collection.db_path)

    ext_map = {"moxfield": "csv", "tcgplayer": "csv", "cardmarket": "csv",
               "arena": "dek", "csv": "csv"}
    dest = path or f"output/collection_{fmt}.{ext_map[fmt]}"

    export_fn = {
        "moxfield": col.export_moxfield,
        "tcgplayer": col.export_tcgplayer,
        "cardmarket": col.export_cardmarket,
        "arena": col.export_arena,
        "csv": col.export_csv,
    }[fmt]

    count = export_fn(dest)
    col.close()
    click.echo(f"{count} Einträge als {fmt.capitalize()} exportiert nach: {dest}")


# ---------------------------------------------------------------------------
# wishlist command group
# ---------------------------------------------------------------------------


@cli.group("wishlist", context_settings=CONTEXT_SETTINGS)
def wishlist_group() -> None:
    """Wunschliste und Tauschbörse verwalten.

    \b
    Befehle:
      stats         Statistiken beider Listen
      want-add      Karte zur Wunschliste hinzufügen
      want-list     Wunschliste anzeigen
      want-remove   Eintrag aus Wunschliste entfernen
      have-add      Karte zur Tauschbörse hinzufügen
      have-list     Tauschbörse anzeigen
      have-remove   Eintrag aus Tauschbörse entfernen
      compare       Wunschliste mit Sammlung abgleichen
      export        Beide Listen als CSV exportieren
    """


@wishlist_group.command("help", hidden=False)
@click.pass_context
def wishlist_help(ctx: click.Context) -> None:
    """Hilfe zu wishlist-Befehlen anzeigen."""
    click.echo(ctx.parent.get_help())


@wishlist_group.command("stats")
@click.option("--verbose", "-v", is_flag=True)
def wishlist_stats(verbose: bool) -> None:
    """Statistiken der Wunschliste und Tauschbörse."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    s = wl.stats()
    wl.close()
    click.echo("Wunschliste / Tauschbörse:")
    click.echo(f"  Wunschliste Einträge : {s['want_entries']}")
    click.echo(f"  Wunschliste Karten   : {s['want_cards']}")
    click.echo(f"  Tauschbörse Einträge : {s['have_entries']}")
    click.echo(f"  Tauschbörse Karten   : {s['have_cards']}")
    click.echo(f"  Datenbank            : {s['db_path']}")


@wishlist_group.command("want-add")
@click.option("--name", required=True, help="Kartenname.")
@click.option("--scryfall-id", default="", help="Scryfall-UUID (optional).")
@click.option("--set-code", default="", help="Set-Code (optional).")
@click.option("--foil", is_flag=True, default=False)
@click.option("--condition", default="NM",
              type=click.Choice(["NM", "LP", "MP", "HP", "DMG"], case_sensitive=False))
@click.option("--qty", default=1, show_default=True)
@click.option("--max-price", default=None, type=float, help="Maximaler Kaufpreis EUR.")
@click.option("--priority", default=2, show_default=True,
              type=click.Choice(["1", "2", "3"]), help="Priorität: 1=Hoch 2=Mittel 3=Niedrig.")
@click.option("--notes", default="")
@click.option("--verbose", "-v", is_flag=True)
def wishlist_want_add(name, scryfall_id, set_code, foil, condition,
                      qty, max_price, priority, notes, verbose):
    """Karte zur Wunschliste hinzufügen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    eid = wl.add_want(
        name=name, scryfall_id=scryfall_id, set_code=set_code,
        foil=foil, condition=condition, quantity=qty,
        max_price_eur=max_price, priority=int(priority), notes=notes,
    )
    wl.close()
    foil_str = " (Foil)" if foil else ""
    click.echo(f"Zur Wunschliste hinzugefügt: {qty}x {name}{foil_str} → Eintrag #{eid}")


@wishlist_group.command("want-list")
@click.option("--name", default="", help="Namensfilter.")
@click.option("--verbose", "-v", is_flag=True)
def wishlist_want_list(name, verbose):
    """Wunschliste anzeigen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager, PRIORITIES

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    entries = wl.get_want_list(name_filter=name)
    wl.close()
    if not entries:
        click.echo("Wunschliste ist leer.")
        return
    click.echo(f"{'ID':>4}  {'Qty':>3}  {'Prio':<6}  {'Cond':<4}  {'Foil':<4}  Name")
    click.echo("—" * 60)
    for e in entries:
        foil = "✓" if e["foil"] else ""
        prio = PRIORITIES.get(e["priority"], str(e["priority"]))
        click.echo(f"{e['id']:>4}  {e['quantity_wanted']:>3}x  {prio:<6}  "
                   f"{e['condition']:<4}  {foil:<4}  {e['name']}")


@wishlist_group.command("want-remove")
@click.argument("entry_id", type=int)
@click.option("--verbose", "-v", is_flag=True)
def wishlist_want_remove(entry_id, verbose):
    """Eintrag aus der Wunschliste entfernen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    ok = wl.remove_want(entry_id)
    wl.close()
    click.echo(f"Eintrag #{entry_id} {'entfernt' if ok else 'nicht gefunden'}.")


@wishlist_group.command("have-add")
@click.option("--name", required=True)
@click.option("--scryfall-id", default="")
@click.option("--set-code", default="")
@click.option("--collector-number", default="")
@click.option("--foil", is_flag=True, default=False)
@click.option("--condition", default="NM",
              type=click.Choice(["NM", "LP", "MP", "HP", "DMG"], case_sensitive=False))
@click.option("--qty", default=1, show_default=True)
@click.option("--ask-price", default=None, type=float, help="Wunschpreis EUR.")
@click.option("--notes", default="")
@click.option("--verbose", "-v", is_flag=True)
def wishlist_have_add(name, scryfall_id, set_code, collector_number,
                      foil, condition, qty, ask_price, notes, verbose):
    """Karte zur Tauschbörse hinzufügen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    eid = wl.add_have(
        name=name, scryfall_id=scryfall_id, set_code=set_code,
        collector_number=collector_number, foil=foil, condition=condition,
        quantity=qty, ask_price_eur=ask_price, notes=notes,
    )
    wl.close()
    click.echo(f"Zur Tauschbörse hinzugefügt: {qty}x {name} → Eintrag #{eid}")


@wishlist_group.command("have-list")
@click.option("--name", default="")
@click.option("--verbose", "-v", is_flag=True)
def wishlist_have_list(name, verbose):
    """Tauschbörse anzeigen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    entries = wl.get_have_list(name_filter=name)
    wl.close()
    if not entries:
        click.echo("Tauschbörse ist leer.")
        return
    click.echo(f"{'ID':>4}  {'Qty':>3}  {'Cond':<4}  {'Foil':<4}  {'Set':>4}  Name")
    click.echo("—" * 60)
    for e in entries:
        foil = "✓" if e["foil"] else ""
        click.echo(f"{e['id']:>4}  {e['quantity']:>3}x  {e['condition']:<4}  "
                   f"{foil:<4}  {(e['set_code'] or '').upper():>4}  {e['name']}")


@wishlist_group.command("have-remove")
@click.argument("entry_id", type=int)
@click.option("--verbose", "-v", is_flag=True)
def wishlist_have_remove(entry_id, verbose):
    """Eintrag aus der Tauschbörse entfernen."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    ok = wl.remove_have(entry_id)
    wl.close()
    click.echo(f"Eintrag #{entry_id} {'entfernt' if ok else 'nicht gefunden'}.")


@wishlist_group.command("compare")
@click.option("--verbose", "-v", is_flag=True)
def wishlist_compare(verbose):
    """Wunschliste mit Sammlung abgleichen — was fehlt noch?"""
    _setup_logging(verbose)
    from mtg_scanner.collection import CollectionManager
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    col = CollectionManager(db_path=cfg.collection.db_path)
    results = wl.compare_want_vs_collection(col)
    col.close()
    wl.close()

    if not results:
        click.echo("Wunschliste ist leer.")
        return

    click.echo(f"{'Name':<35}  {'Prio':<6}  {'Gewünscht':>9}  {'Besitz':>6}  {'Fehlt':>5}")
    click.echo("—" * 70)
    for r in results:
        click.echo(
            f"{r['name'][:34]:<35}  {r['priority']:<6}  "
            f"{r['quantity_wanted']:>9}  {r['quantity_owned']:>6}  {r['still_missing']:>5}"
        )
    missing = sum(r["still_missing"] for r in results)
    click.echo(f"\n{missing} Karten noch gesucht.")


@wishlist_group.command("export")
@click.option("--want-path", default="output/want_list.csv", show_default=True)
@click.option("--have-path", default="output/have_list.csv", show_default=True)
@click.option("--verbose", "-v", is_flag=True)
def wishlist_export(want_path, have_path, verbose):
    """Wunschliste und Tauschbörse als CSV exportieren."""
    _setup_logging(verbose)
    from mtg_scanner.config import get_config
    from mtg_scanner.wishlist import WishlistManager

    cfg = get_config()
    wl = WishlistManager(db_path=cfg.wishlist.db_path)
    w = wl.export_want_list(want_path)
    h = wl.export_have_list(have_path)
    wl.close()
    click.echo(f"Wunschliste: {w} Einträge → {want_path}")
    click.echo(f"Tauschbörse: {h} Einträge → {have_path}")


# ---------------------------------------------------------------------------
# ui command
# ---------------------------------------------------------------------------


@cli.command("ui")
@click.option("--port", default=7860, show_default=True, help="Port für den Gradio-Server.")
@click.option("--share", is_flag=True, default=False, help="Öffentlichen Gradio-Link erstellen.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def ui_cmd(port: int, share: bool, verbose: bool) -> None:
    """Gradio-Weboberfläche starten (Standard: http://localhost:7860)."""
    _setup_logging(verbose)
    try:
        from mtg_scanner.ui import launch_ui
    except ImportError as exc:
        click.echo(
            f"Gradio ist nicht installiert: {exc}\n"
            "Installieren Sie es mit:  pip install 'mtg-card-scanner[ui]'",
            err=True,
        )
        sys.exit(1)
    launch_ui(share=share, port=port)


if __name__ == "__main__":
    cli()
