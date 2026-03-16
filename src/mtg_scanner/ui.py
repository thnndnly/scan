"""Gradio web interface for MTG Card Scanner.

All labels, buttons and headings are in German.

Launch with::

    mtg-scan ui [--port 7860] [--share]

or programmatically::

    from mtg_scanner.ui import launch_ui
    launch_ui()
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Gradio import with helpful error message
# ---------------------------------------------------------------------------

try:
    import gradio as gr  # type: ignore
except ImportError as _gradio_import_error:
    raise ImportError(
        "Gradio ist nicht installiert. Installieren Sie es mit:\n"
        "  pip install 'mtg-card-scanner[ui]'\n"
        "oder:\n"
        "  pip install gradio>=4.0"
    ) from _gradio_import_error

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_dataset_logger():
    """Return an initialised DatasetLogger or None."""
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.dataset import DatasetLogger

        cfg = get_config()
        if not cfg.dataset.enabled:
            return None
        if not Path(cfg.dataset.db_path).exists():
            return None
        return DatasetLogger(db_path=cfg.dataset.db_path, save_patches=cfg.dataset.save_patches)
    except Exception as exc:
        logger.warning("Konnte DatasetLogger nicht initialisieren: %s", exc)
        return None


def _patch_caption(card) -> str:
    """Build a gallery caption string for a RecognizedCard."""
    name = card.card_name or "Unbekannt"
    method = card.recognition_method
    conf = card.recognition_confidence
    price = ""
    if card.card_data and card.card_data.price_eur is not None:
        price = f" — €{card.card_data.price_eur:.2f}"
    return f"{name} ({method}, {conf:.0%}){price}"


def _patch_to_pil(patch_image):
    """Convert a BGR NumPy array to a PIL Image."""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        rgb = cv2.cvtColor(patch_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tab 1: Scanner
# ---------------------------------------------------------------------------


def _run_scan(files, progress=gr.Progress()):
    """Run the pipeline on one or more uploaded images."""
    if not files:
        return "Bitte mindestens ein Bild auswählen.", [], "Kein Bild ausgewählt."

    if not isinstance(files, list):
        files = [files]

    # Normalize to path strings
    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)

    if not paths:
        return "Keine gültigen Bilddateien.", [], "Keine Bilder."

    try:
        from mtg_scanner.pipeline import Pipeline

        progress(0, desc="Pipeline wird initialisiert…")
        pipeline = Pipeline()

        gallery_items = []
        total_detected = 0
        total_recognized = 0
        total_unknown = 0
        total_value = 0.0

        for i, path in enumerate(paths):
            progress(
                i / len(paths) * 0.9,
                desc=f"Bild {i + 1}/{len(paths)}: {Path(path).name}…",
            )
            result = pipeline.process_image(path)

            for card in result.cards:
                pil_img = _patch_to_pil(card.patch.image)
                if pil_img is not None:
                    gallery_items.append((pil_img, _patch_caption(card)))

            total_detected += result.total_detected
            total_recognized += result.total_recognized
            total_unknown += result.total_unknown
            total_value += sum(
                c.card_data.price_eur
                for c in result.cards
                if c.card_data and c.card_data.price_eur is not None
            )

        progress(1.0, desc="Fertig")
        n = len(paths)
        suffix = "er" if n != 1 else ""
        summary = (
            f"Bilder: {n}  |  Erkannt: {total_detected}  |  "
            f"Identifiziert: {total_recognized}  |  "
            f"Unbekannt: {total_unknown}  |  "
            f"Gesamtwert: €{total_value:.2f}"
        )
        return f"Scan abgeschlossen ({n} Bild{suffix}).", gallery_items, summary

    except Exception as exc:
        logger.exception("Scan fehlgeschlagen")
        return f"Fehler beim Scan: {exc}", [], "Scan fehlgeschlagen."


def _export_scan_csv():
    """Export dataset to CSV and return path for download."""
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.dataset import DatasetLogger

        cfg = get_config()
        if not Path(cfg.dataset.db_path).exists():
            return None
        dl = DatasetLogger(db_path=cfg.dataset.db_path, save_patches=False)
        out_path = "output/dataset_export.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        dl.export_csv(out_path)
        dl.close()
        return out_path
    except Exception as exc:
        logger.warning("CSV-Export fehlgeschlagen: %s", exc)
        return None


def _build_scanner_tab():
    with gr.Tab("Scanner"):
        gr.Markdown("## Karten scannen")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.File(
                    file_count="multiple",
                    file_types=["image", ".avif", ".heic", ".heif"],
                    label="Bilder hochladen (mehrere möglich)",
                )
                scan_btn = gr.Button("Scan starten", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False, lines=2)
                export_btn = gr.Button("CSV herunterladen")
                csv_download = gr.File(label="CSV-Download", visible=False)
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Erkannte Karten",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )
                summary_box = gr.Textbox(
                    label="Zusammenfassung", interactive=False, lines=2
                )

        scan_btn.click(
            fn=_run_scan,
            inputs=[image_input],
            outputs=[status_box, gallery, summary_box],
        )
        export_btn.click(
            fn=_export_scan_csv,
            inputs=[],
            outputs=[csv_download],
        )
        export_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[csv_download],
        )


# ---------------------------------------------------------------------------
# Tab 2: Scan-Historie
# ---------------------------------------------------------------------------


def _load_history():
    import gradio as gr  # noqa: PLC0415
    dl = _get_dataset_logger()
    if dl is None:
        from mtg_scanner.config import get_config
        cfg = get_config()
        if cfg.dataset.enabled and not Path(cfg.dataset.db_path).exists():
            gr.Warning(
                "Keine Scan-Datenbank gefunden. "
                "Starte einen Scan, damit die Datenbank angelegt wird."
            )
        return []
    history = dl.get_scan_history(limit=100)
    dl.close()
    rows = []
    for h in history:
        rows.append([
            h.get("scan_timestamp", "")[:19].replace("T", " "),
            h.get("image_path", ""),
            h.get("total_detected", 0),
            h.get("total_recognised", 0),
            f"€{h.get('total_value_eur', 0.0):.2f}",
        ])
    return rows


def _load_scan_detail(evt: gr.SelectData, history_data):
    """Load patch gallery for the selected scan row."""
    try:
        row_idx = evt.index[0]
        if history_data is None or len(history_data) == 0 or row_idx >= len(history_data):
            return [], "Kein Scan ausgewählt."

        dl = _get_dataset_logger()
        if dl is None:
            return [], "Keine Datenbank."

        # history_data rows: [date, image_path, detected, recognised, value]
        # Gradio passes a pandas DataFrame — use .iloc for row access
        try:
            image_path = history_data.iloc[row_idx, 1]
        except AttributeError:
            image_path = history_data[row_idx][1]
        # Find scan by image_path
        import sqlite3
        from mtg_scanner.config import get_config
        cfg = get_config()
        conn = sqlite3.connect(cfg.dataset.db_path)
        conn.row_factory = sqlite3.Row
        scan_row = conn.execute(
            "SELECT id FROM scans WHERE image_path=? ORDER BY id DESC LIMIT 1",
            (image_path,)
        ).fetchone()
        if scan_row is None:
            conn.close()
            dl.close()
            return [], "Scan nicht gefunden."

        scan_id = scan_row["id"]
        detail = dl.get_scan_detail(scan_id)
        dl.close()
        conn.close()

        gallery_items = []
        for det in detail.get("detections", []):
            patch_path = det.get("patch_image_path")
            if patch_path and Path(patch_path).exists():
                from PIL import Image
                try:
                    pil_img = Image.open(patch_path)
                    result = det.get("result")
                    if result:
                        caption = (
                            f"{result.get('card_name') or 'Unbekannt'} "
                            f"({result.get('recognition_method', '')}, "
                            f"{(result.get('confidence') or 0):.0%})"
                        )
                    else:
                        caption = f"Patch #{det.get('patch_index', '?')}"
                    gallery_items.append((pil_img, caption))
                except Exception:
                    pass

        info = (
            f"Scan {scan_id}: {image_path}\n"
            f"Erkannt: {detail.get('total_detected', 0)}, "
            f"Identifiziert: {detail.get('total_recognised', 0)}"
        )
        return gallery_items, info
    except Exception as exc:
        logger.warning("Scan-Detail fehlgeschlagen: %s", exc)
        return [], f"Fehler: {exc}"


def _build_history_tab():
    with gr.Tab("Scan-Historie"):
        gr.Markdown("## Scan-Verlauf")
        refresh_btn = gr.Button("Aktualisieren")
        history_table = gr.Dataframe(
            headers=["Datum", "Bild", "Erkannt", "Identifiziert", "Gesamtwert"],
            datatype=["str", "str", "number", "number", "str"],
            label="Scans",
            interactive=False,
        )
        with gr.Row():
            detail_gallery = gr.Gallery(
                label="Patches des ausgewählten Scans",
                columns=4,
                object_fit="contain",
            )
            detail_info = gr.Textbox(label="Detail", interactive=False, lines=3)

        refresh_btn.click(fn=_load_history, inputs=[], outputs=[history_table])
        history_table.select(
            fn=_load_scan_detail,
            inputs=[history_table],
            outputs=[detail_gallery, detail_info],
        )


# ---------------------------------------------------------------------------
# Tab 3: Nachkontrolle
# ---------------------------------------------------------------------------


def _suggest_from_catalog(query_text: str) -> list:
    """Fuzzy-match query_text against card names and return catalog suggestions."""
    if not query_text or not query_text.strip():
        return []
    try:
        import json
        from rapidfuzz import fuzz, process

        names_path = Path("data/card_names.json")
        if not names_path.exists():
            return []
        with open(names_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        name_list = raw if isinstance(raw, list) else list(raw.keys())

        matches = process.extract(
            query_text.strip(), name_list, scorer=fuzz.WRatio, limit=8
        )

        from mtg_scanner.config import get_config
        cfg = get_config()
        rows = []
        if Path(cfg.catalog.db_path).exists():
            from mtg_scanner.lookup.card_catalog import CardCatalog
            cat = CardCatalog(db_path=cfg.catalog.db_path)
            for name, score, _ in matches:
                if score < 40:
                    continue
                printings = cat.search_by_name(name, limit=5)
                for p in printings:
                    prices = p.get("prices") or {}
                    eur = prices.get("eur") or "—"
                    rows.append([
                        p.get("name", name),
                        (p.get("set_code") or "").upper(),
                        (p.get("released_at") or "")[:4],
                        p.get("rarity", "—"),
                        f"€{eur}" if eur != "—" else "—",
                        p.get("id", ""),
                        f"{score:.0f}%",
                    ])
                if not printings:
                    rows.append([name, "—", "—", "—", "—", "", f"{score:.0f}%"])
            cat.close()
        else:
            for name, score, _ in matches:
                if score >= 40:
                    rows.append([name, "—", "—", "—", "—", "", f"{score:.0f}%"])
        return rows[:20]
    except Exception as exc:
        logger.warning("Katalog-Vorschläge fehlgeschlagen: %s", exc)
        return []


def _build_nachkontrolle_tab():
    with gr.Tab("Nachkontrolle"):
        gr.Markdown(
            "## Scans nachkontrollieren\n"
            "Wähle einen Scan, prüfe jeden erkannten Patch und korrigiere Kartenname oder Erkennung."
        )

        nc_scan_id_hidden = gr.Textbox(visible=False)
        nc_detection_id_hidden = gr.Textbox(visible=False)

        with gr.Row():
            refresh_scans_btn = gr.Button("Scans laden", variant="primary")
            nc_show_reviewed = gr.Checkbox(label="Erledigte Scans anzeigen", value=False)

        nc_scans_table = gr.Dataframe(
            headers=["ID", "Datum", "Bild", "Erkannt", "Identifiziert", "Korr. Anzahl", "Fertig"],
            datatype=["number", "str", "str", "number", "number", "str", "str"],
            label="Scans — Zeile anklicken",
            interactive=False,
        )

        nc_reviewed_banner = gr.Markdown("", visible=False)

        with gr.Row():
            corrected_count_input = gr.Number(
                label="Tatsächliche Anzahl Karten im Bild", precision=0, minimum=0
            )
            save_count_btn = gr.Button("Anzahl speichern")
            count_status = gr.Textbox(label="Status", interactive=False)
            finish_review_btn = gr.Button("Korrektur abschließen ✓", variant="primary")

        nc_patches_gallery = gr.Gallery(
            label="Erkannte Patches — anklicken zum Überprüfen",
            columns=5,
            object_fit="contain",
            height=320,
        )

        gr.Markdown("---")
        gr.Markdown("### Patch überprüfen")

        with gr.Row():
            with gr.Column(scale=1):
                nc_patch_img = gr.Image(label="Patch-Bild", type="pil")
                nc_ocr_text = gr.Textbox(label="OCR-Rohtext", interactive=False)
                nc_current_name = gr.Textbox(label="Erkannte Karte", interactive=False)
                nc_current_conf = gr.Textbox(label="Konfidenz & Methode", interactive=False)
                with gr.Row():
                    nc_approve_btn = gr.Button("✓ Karte korrekt erkannt", variant="primary")
                    nc_reject_btn = gr.Button("✗ Keine Karte / Fehler", variant="stop")
                nc_det_status = gr.Textbox(label="Erkennungs-Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Kartenname + Edition korrigieren")
                nc_suggest_input = gr.Textbox(
                    label="Suchbegriff für Vorschläge",
                    placeholder="Wird automatisch aus OCR-Text vorgeschlagen",
                )
                nc_suggest_btn = gr.Button("Katalog durchsuchen")
                nc_suggestions_table = gr.Dataframe(
                    headers=["Name", "Set", "Jahr", "Seltenheit", "EUR", "Scryfall-ID", "Score"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    label="Vorschläge — Zeile anklicken zum Übernehmen",
                    interactive=False,
                )
                nc_correction_name = gr.Textbox(label="Korrekter Kartenname")
                nc_correction_sf_id = gr.Textbox(
                    label="Scryfall-ID (aus Vorschlägen)", interactive=False
                )
                nc_save_correction_btn = gr.Button("Korrektur speichern", variant="primary")
                nc_correction_status = gr.Textbox(label="Status", interactive=False)

        # --- Event handlers ---

        def load_scans_for_nc(show_reviewed=False):
            import sqlite3 as _sq
            from mtg_scanner.config import get_config
            cfg = get_config()
            if not Path(cfg.dataset.db_path).exists():
                return []
            conn = _sq.connect(cfg.dataset.db_path)
            conn.row_factory = _sq.Row
            try:
                where = "" if show_reviewed else "WHERE (reviewed IS NULL OR reviewed = 0)"
                rows = conn.execute(
                    f"SELECT id, scan_timestamp, image_path, total_detected, "
                    f"total_recognised, corrected_count, reviewed FROM scans {where} "
                    f"ORDER BY id DESC LIMIT 200"
                ).fetchall()
            except Exception:
                rows = conn.execute(
                    "SELECT id, scan_timestamp, image_path, total_detected, "
                    "total_recognised FROM scans ORDER BY id DESC LIMIT 200"
                ).fetchall()
            conn.close()
            result = []
            for r in rows:
                d = dict(r)
                corrected = d.get("corrected_count")
                reviewed = d.get("reviewed", 0)
                result.append([
                    d["id"],
                    str(d.get("scan_timestamp", ""))[:19].replace("T", " "),
                    Path(d.get("image_path", "")).name,
                    d.get("total_detected", 0),
                    d.get("total_recognised", 0),
                    str(corrected) if corrected is not None else "—",
                    "✓ Fertig" if reviewed else "—",
                ])
            return result

        def on_nc_scan_select(evt: gr.SelectData, table_data):
            def _updates(detected, locked):
                return (
                    gr.update(value=detected, interactive=not locked),  # corrected_count_input
                    gr.update(interactive=not locked),  # save_count_btn
                    gr.update(interactive=not locked),  # finish_review_btn
                    gr.update(interactive=not locked),  # nc_approve_btn
                    gr.update(interactive=not locked),  # nc_reject_btn
                    gr.update(interactive=not locked),  # nc_suggest_btn
                    gr.update(interactive=not locked),  # nc_save_correction_btn
                )
            try:
                if table_data is None or len(table_data) == 0:
                    return ("", [], gr.update(visible=False), *_updates(0, False))
                try:
                    row = list(table_data.iloc[evt.index[0]])
                except AttributeError:
                    row = table_data[evt.index[0]]
                scan_id = int(row[0])
                detected = int(row[3]) if row[3] else 0
                is_reviewed = str(row[6]).strip() == "✓ Fertig" if len(row) > 6 else False

                import sqlite3 as _sq
                from mtg_scanner.config import get_config
                cfg = get_config()
                conn = _sq.connect(cfg.dataset.db_path)
                conn.row_factory = _sq.Row
                dets = conn.execute(
                    "SELECT d.id, d.patch_image_path, r.card_name, r.confidence "
                    "FROM detections d LEFT JOIN results r ON r.detection_id = d.id "
                    "WHERE d.scan_id = ? ORDER BY d.patch_index",
                    (scan_id,),
                ).fetchall()
                conn.close()

                gallery = []
                for det in dets:
                    path = det["patch_image_path"]
                    if path and Path(path).exists():
                        from PIL import Image as _PIL
                        try:
                            img = _PIL.open(path)
                            name = det["card_name"] or "Unbekannt"
                            conf = f"{(det['confidence'] or 0):.0%}"
                            gallery.append((img, f"{name} ({conf})"))
                        except Exception:
                            pass

                if is_reviewed:
                    banner = gr.update(
                        value="⚠️ **Dieser Scan wurde bereits abgeschlossen und kann nicht mehr bearbeitet werden.**",
                        visible=True,
                    )
                else:
                    banner = gr.update(visible=False)

                return (str(scan_id), gallery, banner, *_updates(detected, is_reviewed))
            except Exception as exc:
                logger.warning("Scan-Auswahl fehlgeschlagen: %s", exc)
                return ("", [], gr.update(visible=False), *_updates(0, False))

        def save_corrected_count(scan_id_str, count):
            if not scan_id_str:
                return "Kein Scan ausgewählt."
            try:
                scan_id = int(scan_id_str)
                count_int = int(count) if count is not None else None
                import sqlite3 as _sq
                from mtg_scanner.config import get_config
                cfg = get_config()
                conn = _sq.connect(cfg.dataset.db_path)
                conn.execute(
                    "UPDATE scans SET corrected_count = ? WHERE id = ?",
                    (count_int, scan_id),
                )
                conn.commit()
                conn.close()
                return f"Anzahl für Scan #{scan_id} gespeichert: {count_int}"
            except Exception as exc:
                return f"Fehler: {exc}"

        def on_nc_patch_select(evt: gr.SelectData, scan_id_str):
            try:
                if not scan_id_str:
                    return None, "", "", "", "", ""
                scan_id = int(scan_id_str)
                gallery_idx = evt.index

                import sqlite3 as _sq
                from mtg_scanner.config import get_config
                cfg = get_config()
                conn = _sq.connect(cfg.dataset.db_path)
                conn.row_factory = _sq.Row

                dets = conn.execute(
                    "SELECT d.id, d.patch_image_path "
                    "FROM detections d WHERE d.scan_id = ? ORDER BY d.patch_index",
                    (scan_id,),
                ).fetchall()

                # Filter to those with valid images (same as gallery)
                valid_dets = [d for d in dets if d["patch_image_path"] and Path(d["patch_image_path"]).exists()]

                if gallery_idx >= len(valid_dets):
                    conn.close()
                    return None, "", "", "", "", ""

                det = valid_dets[gallery_idx]
                det_id = det["id"]
                patch_path = det["patch_image_path"]

                attempt = conn.execute(
                    "SELECT raw_text, matched_name, confidence, method "
                    "FROM recognition_attempts WHERE detection_id = ? ORDER BY id ASC LIMIT 1",
                    (det_id,),
                ).fetchone()

                result_row = conn.execute(
                    "SELECT card_name, corrected_name, recognition_method, confidence "
                    "FROM results WHERE detection_id = ?",
                    (det_id,),
                ).fetchone()
                conn.close()

                from PIL import Image as _PIL
                pil_img = _PIL.open(patch_path)

                raw_text = (attempt["raw_text"] or "") if attempt else ""
                card_name = ""
                conf_info = ""
                if result_row:
                    card_name = result_row["corrected_name"] or result_row["card_name"] or ""
                    conf = result_row["confidence"] or 0
                    method = result_row["recognition_method"] or "—"
                    conf_info = f"{method}, {conf:.0%}"

                return pil_img, raw_text, card_name, conf_info, str(det_id), raw_text
            except Exception as exc:
                logger.warning("Patch-Auswahl fehlgeschlagen: %s", exc)
                return None, "", "", "", "", ""

        def on_suggestion_select(evt: gr.SelectData, table_data):
            try:
                if table_data is None or len(table_data) == 0:
                    return "", ""
                try:
                    row = list(table_data.iloc[evt.index[0]])
                except AttributeError:
                    row = table_data[evt.index[0]]
                name = str(row[0])
                sf_id = str(row[5]) if len(row) > 5 else ""
                return name, sf_id
            except Exception:
                return "", ""

        def save_nc_correction(det_id_str, name, sf_id):
            if not det_id_str:
                return "Kein Patch ausgewählt."
            if not name.strip():
                return "Bitte Kartenname eingeben."
            try:
                det_id = int(det_id_str)
                dl = _get_dataset_logger()
                if dl is None:
                    return "Keine Datenbank verfügbar."
                if sf_id and sf_id.strip() and sf_id.strip() not in ("—", ""):
                    dl.assign_card(det_id, sf_id.strip(), "")
                dl.correct_card(det_id, name.strip())
                dl.close()
                return f"Korrektur gespeichert: #{det_id} → '{name.strip()}'"
            except Exception as exc:
                return f"Fehler: {exc}"

        def finish_review(scan_id_str, show_reviewed):
            if not scan_id_str:
                return "Kein Scan ausgewählt.", []
            try:
                scan_id = int(scan_id_str)
                import sqlite3 as _sq
                from mtg_scanner.config import get_config
                cfg = get_config()
                conn = _sq.connect(cfg.dataset.db_path)
                conn.execute("UPDATE scans SET reviewed = 1 WHERE id = ?", (scan_id,))
                conn.commit()
                conn.close()
                return f"Scan #{scan_id} als erledigt markiert.", load_scans_for_nc(show_reviewed)
            except Exception as exc:
                return f"Fehler: {exc}", []

        def set_det_approved(det_id_str, approved):
            if not det_id_str:
                return "Kein Patch ausgewählt."
            try:
                det_id = int(det_id_str)
                import sqlite3 as _sq
                from mtg_scanner.config import get_config
                cfg = get_config()
                conn = _sq.connect(cfg.dataset.db_path)
                conn.execute(
                    "UPDATE detections SET detection_approved = ? WHERE id = ?",
                    (1 if approved else 0, det_id),
                )
                conn.commit()
                conn.close()
                label = "✓ Korrekt markiert" if approved else "✗ Als Fehler markiert"
                return f"Patch #{det_id}: {label}"
            except Exception as exc:
                return f"Fehler: {exc}"

        # --- Wire up ---
        refresh_scans_btn.click(
            fn=load_scans_for_nc,
            inputs=[nc_show_reviewed],
            outputs=[nc_scans_table],
        )
        nc_show_reviewed.change(
            fn=load_scans_for_nc,
            inputs=[nc_show_reviewed],
            outputs=[nc_scans_table],
        )

        nc_scans_table.select(
            fn=on_nc_scan_select,
            inputs=[nc_scans_table],
            outputs=[
                nc_scan_id_hidden, nc_patches_gallery, nc_reviewed_banner,
                corrected_count_input, save_count_btn, finish_review_btn,
                nc_approve_btn, nc_reject_btn, nc_suggest_btn, nc_save_correction_btn,
            ],
        )

        save_count_btn.click(
            fn=save_corrected_count,
            inputs=[nc_scan_id_hidden, corrected_count_input],
            outputs=[count_status],
        )

        finish_review_btn.click(
            fn=finish_review,
            inputs=[nc_scan_id_hidden, nc_show_reviewed],
            outputs=[count_status, nc_scans_table],
        )

        nc_patches_gallery.select(
            fn=on_nc_patch_select,
            inputs=[nc_scan_id_hidden],
            outputs=[
                nc_patch_img, nc_ocr_text, nc_current_name, nc_current_conf,
                nc_detection_id_hidden, nc_suggest_input,
            ],
        )

        nc_suggest_btn.click(
            fn=_suggest_from_catalog,
            inputs=[nc_suggest_input],
            outputs=[nc_suggestions_table],
        )

        nc_suggestions_table.select(
            fn=on_suggestion_select,
            inputs=[nc_suggestions_table],
            outputs=[nc_correction_name, nc_correction_sf_id],
        )

        nc_approve_btn.click(
            fn=lambda det_id: set_det_approved(det_id, True),
            inputs=[nc_detection_id_hidden],
            outputs=[nc_det_status],
        )

        nc_reject_btn.click(
            fn=lambda det_id: set_det_approved(det_id, False),
            inputs=[nc_detection_id_hidden],
            outputs=[nc_det_status],
        )

        nc_save_correction_btn.click(
            fn=save_nc_correction,
            inputs=[nc_detection_id_hidden, nc_correction_name, nc_correction_sf_id],
            outputs=[nc_correction_status],
        )


# ---------------------------------------------------------------------------
# Tab 4: Hash-DB
# ---------------------------------------------------------------------------

_hash_build_thread: Optional[threading.Thread] = None
_hash_build_log: list[str] = []
_hash_build_running = False


def _hash_db_status():
    try:
        from mtg_scanner.config import get_config
        cfg = get_config()
        hash_db_path = cfg.recognition.hash_db_path
        p = Path(hash_db_path)
        if not p.exists():
            return "Hash-DB nicht vorhanden.", 0
        import sqlite3
        conn = sqlite3.connect(str(p))
        count = conn.execute("SELECT COUNT(*) FROM card_hashes").fetchone()[0]
        conn.close()
        return f"Hash-DB: {count} Einträge  ({p})", count
    except Exception as exc:
        return f"Fehler: {exc}", 0


def _hash_dry_run(sets_str: str, limit_str: str):
    try:
        import requests
        import json

        sets = [s.strip() for s in sets_str.split(",") if s.strip()] if sets_str else None
        limit = int(limit_str) if limit_str.strip() else None

        status_msg = "Lade Bulk-Daten von Scryfall…\n"

        resp = requests.get("https://api.scryfall.com/bulk-data", timeout=30,
                            headers={"User-Agent": "mtg-card-scanner/0.1"})
        resp.raise_for_status()
        bulk_url = None
        for item in resp.json().get("data", []):
            if item.get("type") == "default_cards":
                bulk_url = item["download_uri"]
                break
        if not bulk_url:
            return "Konnte Bulk-Daten-URL nicht finden."

        resp2 = requests.get(bulk_url, timeout=60,
                             headers={"User-Agent": "mtg-card-scanner/0.1"})
        resp2.raise_for_status()
        all_cards = resp2.json()
        total_all = len(all_cards)

        if sets:
            set_lower = {s.lower() for s in sets}
            all_cards = [c for c in all_cards if c.get("set", "").lower() in set_lower]

        if limit:
            all_cards = all_cards[:limit]

        card_count = len(all_cards)
        est_time_min = card_count * 0.1 / 60
        est_size_mb = card_count * 50 / 1024

        lines = [
            f"Bulk-Daten: {total_all} Karten gesamt",
            f"Nach Filter: {card_count} Karten",
            f"Geschätzte Download-Zeit: {est_time_min:.1f} Minuten",
            f"Geschätzter Speicherbedarf: {est_size_mb:.0f} MB (Hashes in SQLite)",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Fehler beim Dry-Run: {exc}"


def _build_hash_db_background(sets_str: str, limit_str: str):
    global _hash_build_running, _hash_build_log
    _hash_build_running = True
    _hash_build_log = ["Hash-DB-Aufbau gestartet…"]
    try:
        import subprocess, sys
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        script = str(scripts_dir / "build_hash_db.py")
        args = [sys.executable, script]
        if sets_str.strip():
            args += ["--sets", sets_str.strip()]
        if limit_str.strip():
            args += ["--limit", limit_str.strip()]
        result = subprocess.run(args, capture_output=True, text=True)
        _hash_build_log.append(result.stdout or "")
        if result.stderr:
            _hash_build_log.append(result.stderr)
        _hash_build_log.append("Fertig." if result.returncode == 0 else f"Fehlercode: {result.returncode}")
    except Exception as exc:
        _hash_build_log.append(f"Fehler: {exc}")
    finally:
        _hash_build_running = False


def _start_hash_build(sets_str: str, limit_str: str):
    global _hash_build_thread
    if _hash_build_running:
        return "Aufbau läuft bereits…"
    _hash_build_thread = threading.Thread(
        target=_build_hash_db_background, args=(sets_str, limit_str), daemon=True
    )
    _hash_build_thread.start()
    return "Hash-DB-Aufbau gestartet (Hintergrundprozess)."


def _get_hash_build_log():
    return "\n".join(_hash_build_log) if _hash_build_log else "Kein laufender Aufbau."


def _build_hash_db_tab():
    with gr.Tab("Hash-DB"):
        gr.Markdown("## Perceptual-Hash-Datenbank aufbauen")
        status_text, count = _hash_db_status()
        db_status = gr.Textbox(value=status_text, label="Status", interactive=False)
        refresh_status_btn = gr.Button("Status aktualisieren")

        with gr.Row():
            sets_input = gr.Textbox(
                label="Sets (kommagetrennt, z.B. m21,lea)",
                placeholder="Leer = alle Sets",
            )
            limit_input = gr.Textbox(
                label="Limit (max. Karten)",
                placeholder="Leer = keine Begrenzung",
            )
        with gr.Row():
            dry_run_btn = gr.Button("Dry Run")
            build_btn = gr.Button("Jetzt aufbauen", variant="primary")
        dry_run_output = gr.Textbox(label="Dry-Run-Ergebnis", interactive=False, lines=6)
        build_log = gr.Textbox(label="Aufbau-Log", interactive=False, lines=6)
        log_refresh_btn = gr.Button("Log aktualisieren")

        refresh_status_btn.click(
            fn=lambda: _hash_db_status()[0], inputs=[], outputs=[db_status]
        )
        dry_run_btn.click(
            fn=_hash_dry_run, inputs=[sets_input, limit_input], outputs=[dry_run_output]
        )
        build_btn.click(
            fn=_start_hash_build, inputs=[sets_input, limit_input], outputs=[build_log]
        )
        log_refresh_btn.click(
            fn=_get_hash_build_log, inputs=[], outputs=[build_log]
        )


# ---------------------------------------------------------------------------
# Tab 5: Training (YOLO + CLIP)
# ---------------------------------------------------------------------------

_training_log: list[str] = []


def _clip_db_status() -> str:
    """Return a status string for the CLIP embedding database."""
    try:
        from mtg_scanner.config import get_config
        cfg = get_config()
        db = Path(cfg.clip.db_path)
        if not db.exists():
            return "CLIP-DB nicht gefunden. Zuerst aufbauen."
        import sqlite3
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
        sets = conn.execute("SELECT COUNT(DISTINCT set_code) FROM clip_embeddings").fetchone()[0]
        conn.close()
        return f"CLIP-DB: {count} Embeddings, {sets} Sets  ({db})"
    except Exception as exc:
        return f"Fehler beim Lesen der CLIP-DB: {exc}"


def _yolo_model_status() -> str:
    """Return a status string for the fine-tuned YOLO model."""
    try:
        from mtg_scanner.config import get_config
        cfg = get_config()
        model_path = Path(cfg.detection.yolo_model_path)
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1_048_576
            return f"YOLO-Modell: {model_path}  ({size_mb:.1f} MB)"
        return f"Kein YOLO-Modell gefunden ({model_path}). Erst trainieren."
    except Exception as exc:
        return f"Fehler: {exc}"


def _yolo_dataset_stats() -> str:
    """Return training dataset statistics from dataset.db."""
    try:
        import sqlite3
        from mtg_scanner.config import get_config
        cfg = get_config()
        db = Path(cfg.dataset.db_path)
        if not db.exists():
            return "dataset.db nicht gefunden. Zuerst Karten scannen."
        conn = sqlite3.connect(str(db))
        total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        total_det = conn.execute("SELECT COUNT(*) FROM detections WHERE bbox_w > 0").fetchone()[0]
        validated = conn.execute(
            "SELECT COUNT(*) FROM results WHERE scryfall_id IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        train_n = int(total_det * 0.85)
        val_n = total_det - train_n
        return (
            f"Scans: {total_scans}  |  Detektionen: {total_det}  |  Validiert: {validated}\n"
            f"Train/Val-Split (85/15): {train_n} / {val_n}"
        )
    except Exception as exc:
        return f"Fehler: {exc}"


def _start_clip_build(sets_str: str, limit_str: str) -> str:
    """Start building the CLIP embedding DB in the background."""
    import subprocess, threading, sys
    global _training_log
    _training_log = []
    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    script = str(scripts_dir / "build_clip_db.py")
    args = [sys.executable, script]
    if sets_str.strip():
        args += ["--sets", sets_str.strip()]
    if limit_str.strip():
        args += ["--limit", limit_str.strip()]

    def _run():
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            _training_log.append(line.rstrip())
        proc.wait()

    threading.Thread(target=_run, daemon=True).start()
    return "CLIP-DB-Aufbau gestartet (Hintergrundprozess). Log unten aktualisieren."


def _start_yolo_train(epochs_str: str, base_model: str) -> str:
    """Start YOLO fine-tuning in the background."""
    import subprocess, threading, sys
    global _training_log
    _training_log = []
    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    script = str(scripts_dir / "train_yolo.py")
    try:
        epochs = int(epochs_str) if epochs_str.strip() else 100
    except ValueError:
        epochs = 100
    args = [sys.executable, script, "--epochs", str(epochs), "--base-model", base_model.strip() or "yolov8n.pt"]

    def _run():
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            _training_log.append(line.rstrip())
        proc.wait()

    threading.Thread(target=_run, daemon=True).start()
    return "YOLO-Training gestartet (Hintergrundprozess). Log unten aktualisieren."


def _get_training_log() -> str:
    return "\n".join(_training_log) if _training_log else "Kein laufender Prozess."


def _build_training_tab():
    with gr.Tab("Training"):
        gr.Markdown("## Datengetriebenes Training\n"
                    "Validierte Patches aus der Karten-Zuordnung werden verwendet um YOLO und CLIP zu verbessern.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### YOLO Fine-Tuning")
                yolo_status = gr.Textbox(
                    value=_yolo_model_status(), label="Modell-Status", interactive=False
                )
                dataset_stats = gr.Textbox(
                    value=_yolo_dataset_stats(), label="Dataset-Statistik", interactive=False, lines=3
                )
                yolo_refresh_btn = gr.Button("Status aktualisieren")
                yolo_epochs = gr.Textbox(label="Epochen", value="100", placeholder="100")
                yolo_base_model = gr.Textbox(
                    label="Basis-Modell", value="yolov8n.pt", placeholder="yolov8n.pt"
                )
                yolo_train_btn = gr.Button("YOLO trainieren", variant="primary")

            with gr.Column():
                gr.Markdown("### CLIP Embedding-Datenbank")
                clip_status = gr.Textbox(
                    value=_clip_db_status(), label="DB-Status", interactive=False
                )
                clip_refresh_btn = gr.Button("Status aktualisieren")
                clip_sets = gr.Textbox(
                    label="Sets (kommagetrennt)", placeholder="Leer = alle Sets"
                )
                clip_limit = gr.Textbox(
                    label="Limit (max. Karten)", placeholder="Leer = keine Begrenzung"
                )
                clip_build_btn = gr.Button("CLIP-DB aufbauen", variant="primary")

        training_log = gr.Textbox(label="Prozess-Log", interactive=False, lines=10)
        log_refresh_btn = gr.Button("Log aktualisieren")

        yolo_refresh_btn.click(
            fn=lambda: (_yolo_model_status(), _yolo_dataset_stats()),
            inputs=[],
            outputs=[yolo_status, dataset_stats],
        )
        clip_refresh_btn.click(
            fn=_clip_db_status, inputs=[], outputs=[clip_status]
        )
        yolo_train_btn.click(
            fn=_start_yolo_train,
            inputs=[yolo_epochs, yolo_base_model],
            outputs=[training_log],
        )
        clip_build_btn.click(
            fn=_start_clip_build,
            inputs=[clip_sets, clip_limit],
            outputs=[training_log],
        )
        log_refresh_btn.click(
            fn=_get_training_log, inputs=[], outputs=[training_log]
        )


# ---------------------------------------------------------------------------
# Tab 6: Einstellungen
# ---------------------------------------------------------------------------


def _load_current_settings():
    from mtg_scanner.config import get_config
    cfg = get_config()
    return (
        cfg.detection.method,
        cfg.recognition.ocr_confidence_threshold,
        cfg.output.low_confidence_threshold,
        cfg.dataset.enabled,
    )


def _save_settings(det_method: str, ocr_threshold: float, low_conf: float, dataset_enabled: bool):
    try:
        import yaml
        from mtg_scanner.config import _find_config_file, reload_config, get_config

        cfg_path = _find_config_file()
        if cfg_path is None:
            cfg_path = Path("config.yaml")

        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        else:
            raw = {}

        raw.setdefault("detection", {})["method"] = det_method
        raw.setdefault("recognition", {})["ocr_confidence_threshold"] = ocr_threshold
        raw.setdefault("output", {})["low_confidence_threshold"] = low_conf
        raw.setdefault("dataset", {})["enabled"] = dataset_enabled

        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(raw, fh, default_flow_style=False, sort_keys=False)

        reload_config()
        return f"Einstellungen gespeichert in {cfg_path}"
    except Exception as exc:
        return f"Fehler beim Speichern: {exc}"


def _build_settings_tab():
    with gr.Tab("Einstellungen"):
        gr.Markdown("## Konfiguration")
        det_method = gr.Dropdown(
            choices=["opencv", "yolo"],
            label="Erkennungsmethode",
        )
        ocr_threshold = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.05,
            label="OCR-Konfidenz-Schwelle",
        )
        low_conf_threshold = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.05,
            label="Niedrig-Konfidenz-Schwelle",
        )
        dataset_enabled = gr.Checkbox(label="Dataset-Logger aktiviert")

        load_btn = gr.Button("Aktuelle Einstellungen laden")
        save_btn = gr.Button("Speichern", variant="primary")
        save_status = gr.Textbox(label="Status", interactive=False)

        load_btn.click(
            fn=_load_current_settings,
            inputs=[],
            outputs=[det_method, ocr_threshold, low_conf_threshold, dataset_enabled],
        )
        save_btn.click(
            fn=_save_settings,
            inputs=[det_method, ocr_threshold, low_conf_threshold, dataset_enabled],
            outputs=[save_status],
        )


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _get_collection_manager():
    """Return an initialised CollectionManager."""
    try:
        from mtg_scanner.collection import CollectionManager
        from mtg_scanner.config import get_config
        cfg = get_config()
        return CollectionManager(db_path=cfg.collection.db_path)
    except Exception as exc:
        logger.warning("Konnte CollectionManager nicht initialisieren: %s", exc)
        return None


def _load_card_image_for_scryfall_id(scryfall_id: str):
    """Fetch the card image (normal size) from catalog image_uris. Returns PIL Image or None."""
    if not scryfall_id:
        return None
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.lookup.card_catalog import CardCatalog
        import requests
        from PIL import Image as PILImage

        cfg = get_config()
        if not Path(cfg.catalog.db_path).exists():
            return None
        cat = CardCatalog(db_path=cfg.catalog.db_path)
        card = cat.get_by_scryfall_id(scryfall_id)
        cat.close()
        if not card:
            return None
        image_uris = card.get("image_uris") or {}
        url = image_uris.get("normal") or image_uris.get("small") or image_uris.get("large")
        if not url:
            return None
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "mtg-card-scanner/0.1"})
        resp.raise_for_status()
        return PILImage.open(io.BytesIO(resp.content))
    except Exception as exc:
        logger.debug("Kartenbild konnte nicht geladen werden: %s", exc)
        return None


def _add_to_collection_from_ui(
    scryfall_id: str,
    display_text: str,
    printings_table,
    condition: str,
    foil: bool,
    qty,
    buy_price,
) -> str:
    """Add the currently selected card printing to the collection."""
    if not scryfall_id:
        return "Bitte zuerst einen Druck auswählen."
    try:
        quantity = max(1, int(qty or 1))
        price = float(buy_price) if buy_price else None

        # Resolve name/set from printings table
        name, set_code, set_name, collector_number, oracle_id, lang = "", "", "", "", "", "en"
        try:
            import pandas as _pd
            if printings_table is not None and hasattr(printings_table, 'iterrows'):
                for _, row in printings_table.iterrows():
                    if str(row.iloc[0]) == scryfall_id:
                        oracle_id = str(row.iloc[1])
                        set_code = str(row.iloc[2])
                        set_name = str(row.iloc[3])
                        collector_number = str(row.iloc[4])
                        break
        except Exception:
            pass

        # Fall back to catalog lookup for name
        try:
            from mtg_scanner.config import get_config
            from mtg_scanner.lookup.card_catalog import CardCatalog
            cfg = get_config()
            if Path(cfg.catalog.db_path).exists():
                cat = CardCatalog(db_path=cfg.catalog.db_path)
                card = cat.get_by_scryfall_id(scryfall_id)
                cat.close()
                if card:
                    name = card.get("name", "")
                    oracle_id = oracle_id or card.get("oracle_id", "")
                    set_code = set_code or card.get("set_code", "")
                    set_name = set_name or card.get("set_name", "")
                    collector_number = collector_number or card.get("collector_number", "")
                    lang = card.get("lang", "en")
        except Exception:
            pass

        if not name:
            name = display_text.split("|")[0].strip() if display_text else "Unbekannt"

        col = _get_collection_manager()
        if col is None:
            return "Sammlung nicht verfügbar."
        entry_id = col.add_card(
            scryfall_id=scryfall_id,
            name=name,
            oracle_id=oracle_id,
            set_code=set_code,
            set_name=set_name,
            collector_number=collector_number,
            lang=lang,
            foil=bool(foil),
            condition=condition or "NM",
            quantity=quantity,
            buy_price=price,
        )
        col.close()
        foil_str = " (Foil)" if foil else ""
        return f"✓ {quantity}x {name}{foil_str} [{condition}] zur Sammlung hinzugefügt (#{entry_id})"
    except Exception as exc:
        logger.warning("Sammlung hinzufügen fehlgeschlagen: %s", exc)
        return f"Fehler: {exc}"


# ---------------------------------------------------------------------------
# Tab: Sammlung
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Catalog helpers (shared across tabs)
# ---------------------------------------------------------------------------


def _catalog_search_printings(card_name: str) -> tuple[list, str]:
    """Search catalog for all printings of card_name. Returns (table_rows, status)."""
    if not card_name.strip():
        return [], "Bitte Kartenname eingeben."
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.lookup.card_catalog import CardCatalog
        cfg = get_config()
        if not Path(cfg.catalog.db_path).exists():
            return [], "Kein Katalog vorhanden. Bitte 'mtg-scan catalog build' ausführen."
        cat = CardCatalog(db_path=cfg.catalog.db_path)
        # First get oracle_id for exact or fuzzy match
        oracle_id = cat.get_oracle_id(card_name.strip())
        if oracle_id:
            printings = cat.get_printings(oracle_id)
        else:
            printings = cat.search_by_name(card_name.strip(), limit=100)
        cat.close()
        if not printings:
            return [], f"Keine Treffer für '{card_name}'."
        rows = []
        for p in printings:
            prices = p.get("prices") or {}
            eur = prices.get("eur") or "—"
            eur_foil = prices.get("eur_foil") or "—"
            finishes = ", ".join(p.get("finishes") or [])
            frame_fx = ", ".join(p.get("frame_effects") or [])
            rows.append([
                p["id"],           # hidden scryfall_id
                p.get("oracle_id", ""),  # hidden oracle_id
                p.get("set_code", "").upper(),
                p.get("set_name", ""),
                p.get("collector_number", ""),
                p.get("released_at", "")[:4] if p.get("released_at") else "—",
                p.get("rarity", "—"),
                finishes,
                frame_fx or "—",
                p.get("artist", "—"),
                f"€{eur}" if eur != "—" else "—",
                f"€{eur_foil}" if eur_foil != "—" else "—",
            ])
        return rows, f"{len(rows)} Drucke gefunden."
    except Exception as exc:
        logger.warning("Katalog-Suche fehlgeschlagen: %s", exc)
        return [], f"Fehler: {exc}"


def _do_assign_patch(
    detection_id_str: str,
    scryfall_id: str,
    oracle_id: str,
    card_name_for_archive: str = "",
) -> str:
    """Save card assignment to dataset.db and image_archive.db."""
    if not detection_id_str or not scryfall_id:
        return "Bitte einen Druck auswählen."
    try:
        detection_id = int(detection_id_str)
        from mtg_scanner.config import get_config
        from mtg_scanner.dataset import DatasetLogger
        cfg = get_config()
        if Path(cfg.dataset.db_path).exists():
            dl = DatasetLogger(db_path=cfg.dataset.db_path, save_patches=False)
            dl.assign_card(detection_id, scryfall_id, oracle_id)
            dl.close()
        # Also store patch in archive if patch image exists
        if Path(cfg.archive.db_path).exists():
            import sqlite3 as _sqlite3
            conn = _sqlite3.connect(cfg.dataset.db_path)
            row = conn.execute(
                "SELECT patch_image_path FROM detections WHERE id = ?", (detection_id,)
            ).fetchone()
            conn.close()
            if row and row[0] and Path(row[0]).exists():
                from mtg_scanner.image_archive import ImageArchive
                from mtg_scanner.utils.image_utils import load_image
                img = load_image(row[0])
                if img is not None:
                    arch = ImageArchive(
                        db_path=cfg.archive.db_path,
                        index_path=cfg.archive.index_path,
                    )
                    arch.store_patch(
                        img,
                        scryfall_id=scryfall_id,
                        oracle_id=oracle_id,
                        detection_id=detection_id,
                        original_filename=row[0],
                    )
                    arch.close()
        return f"Zuordnung gespeichert: {card_name_for_archive} ({scryfall_id[:8]}…)"
    except Exception as exc:
        logger.warning("Zuordnung fehlgeschlagen: %s", exc)
        return f"Fehler: {exc}"


# ---------------------------------------------------------------------------
# Tab: Karten-Zuordnung (Catalog + Patch Assignment)
# ---------------------------------------------------------------------------


def _build_catalog_tab():
    with gr.Tab("Karten-Zuordnung"):
        gr.Markdown("## Karten-Patches dem Katalog zuordnen")
        gr.Markdown(
            "Wähle einen Patch aus dem Scan-Verlauf, suche die Karte im Katalog "
            "und weise den genauen Druck zu. Der Patch wird mit der `scryfall_id` "
            "im Archiv gespeichert."
        )

        with gr.Row():
            # --- Left: patch selector ---
            with gr.Column(scale=1):
                gr.Markdown("### 1. Patch auswählen")
                refresh_patches_btn = gr.Button("Nicht zugeordnete Patches laden")
                patches_table = gr.Dataframe(
                    headers=["Det-ID", "Datum", "Erkannter Name", "Konfidenz", "Methode"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Patches ohne Katalog-Zuordnung",
                    interactive=False,
                )
                patch_preview = gr.Image(label="Patch-Vorschau", type="pil", height=250)
                selected_detection_id = gr.Textbox(visible=False)

            # --- Right: catalog search & assignment ---
            with gr.Column(scale=2):
                gr.Markdown("### 2. Im Katalog suchen")
                with gr.Row():
                    catalog_search_input = gr.Textbox(
                        label="Kartenname suchen",
                        placeholder="z.B. Lightning Bolt",
                    )
                    catalog_search_btn = gr.Button("Suchen")
                catalog_status = gr.Textbox(label="Status", interactive=False, lines=1)
                printings_table = gr.Dataframe(
                    headers=[
                        "ID", "Oracle-ID", "Set", "Set-Name", "Nr.",
                        "Jahr", "Seltenheit", "Finish", "Rahmen",
                        "Künstler", "EUR", "EUR Foil",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str",
                              "str", "str", "str", "str", "str", "str"],
                    label="Alle Drucke — Zeile anklicken zum Auswählen",
                    interactive=False,
                )

                gr.Markdown("### 3. Zuordnung bestätigen")
                with gr.Row():
                    selected_scryfall_id = gr.Textbox(
                        label="Ausgewählte Scryfall-ID", interactive=False
                    )
                    selected_card_display = gr.Textbox(
                        label="Ausgewählter Druck", interactive=False
                    )
                card_image_preview = gr.Image(
                    label="Kartenbild", type="pil", height=280
                )
                assign_btn = gr.Button("Zuordnung speichern", variant="primary")
                assign_status = gr.Textbox(label="Ergebnis", interactive=False)
                gr.Markdown("### 4. Zur Sammlung hinzufügen")
                with gr.Row():
                    coll_condition = gr.Dropdown(
                        choices=["NM", "LP", "MP", "HP", "DMG"],
                        value="NM",
                        label="Zustand",
                    )
                    coll_foil = gr.Checkbox(label="Foil")
                    coll_qty = gr.Number(value=1, label="Anzahl", precision=0, minimum=1)
                    coll_buy_price = gr.Number(
                        value=None, label="Kaufpreis (EUR)", precision=2
                    )
                add_to_collection_btn = gr.Button("Zur Sammlung hinzufügen")
                collection_status = gr.Textbox(label="Sammlungs-Status", interactive=False)

        # --- Wire up ---

        def load_unassigned_patches():
            try:
                from mtg_scanner.config import get_config
                import sqlite3 as _sq
                cfg = get_config()
                if not Path(cfg.dataset.db_path).exists():
                    return []
                conn = _sq.connect(cfg.dataset.db_path)
                conn.row_factory = _sq.Row
                rows = conn.execute("""
                    SELECT d.id, s.scan_timestamp, r.card_name,
                           r.confidence, r.recognition_method
                    FROM detections d
                    JOIN scans s ON s.id = d.scan_id
                    LEFT JOIN results r ON r.detection_id = d.id
                    WHERE r.scryfall_id IS NULL OR r.scryfall_id = ''
                    ORDER BY d.id DESC
                    LIMIT 200
                """).fetchall()
                conn.close()
                return [
                    [
                        str(r["id"]),
                        str(r["scan_timestamp"] or "")[:19].replace("T", " "),
                        r["card_name"] or "—",
                        f"{(r['confidence'] or 0):.0%}",
                        r["recognition_method"] or "—",
                    ]
                    for r in rows
                ]
            except Exception as exc:
                logger.warning("Patches laden fehlgeschlagen: %s", exc)
                return []

        def on_patch_select(evt: gr.SelectData, table_data):
            try:
                if table_data is None or len(table_data) == 0:
                    return None, "", ""
                try:
                    row = list(table_data.iloc[evt.index[0]])
                except AttributeError:
                    row = table_data[evt.index[0]]
                det_id = str(row[0])
                card_name = str(row[2]) if row[2] != "—" else ""

                # Load patch image
                pil_img = None
                try:
                    from mtg_scanner.config import get_config
                    import sqlite3 as _sq
                    cfg = get_config()
                    conn = _sq.connect(cfg.dataset.db_path)
                    pr = conn.execute(
                        "SELECT patch_image_path FROM detections WHERE id = ?", (det_id,)
                    ).fetchone()
                    conn.close()
                    if pr and pr[0] and Path(pr[0]).exists():
                        from PIL import Image as _PIL
                        pil_img = _PIL.open(pr[0])
                except Exception:
                    pass
                return pil_img, det_id, card_name
            except Exception as exc:
                return None, "", ""

        def on_printing_select(evt: gr.SelectData, table_data):
            try:
                if table_data is None or len(table_data) == 0:
                    return "", "", None
                try:
                    row = list(table_data.iloc[evt.index[0]])
                except AttributeError:
                    row = table_data[evt.index[0]]
                scryfall_id = str(row[0])
                display = (
                    f"{row[2]} {row[3]} #{row[4]} "
                    f"| {row[6]} | {row[7]} | {row[9]} | {row[10]}"
                )
                # Load card image from catalog image_uris
                card_img = _load_card_image_for_scryfall_id(scryfall_id)
                return scryfall_id, display, card_img
            except Exception:
                return "", "", None

        def do_assign(det_id, sf_id, table_data, display_text):
            oracle_id = ""
            try:
                if table_data is not None and len(table_data) > 0:
                    # Find the row matching sf_id to get oracle_id
                    import pandas as _pd
                    if hasattr(table_data, 'iterrows'):
                        for _, row in table_data.iterrows():
                            if str(row.iloc[0]) == sf_id:
                                oracle_id = str(row.iloc[1])
                                break
            except Exception:
                pass
            return _do_assign_patch(det_id, sf_id, oracle_id, display_text)

        refresh_patches_btn.click(
            fn=load_unassigned_patches,
            inputs=[],
            outputs=[patches_table],
        )
        patches_table.select(
            fn=on_patch_select,
            inputs=[patches_table],
            outputs=[patch_preview, selected_detection_id, catalog_search_input],
        )
        catalog_search_btn.click(
            fn=_catalog_search_printings,
            inputs=[catalog_search_input],
            outputs=[printings_table, catalog_status],
        )
        catalog_search_input.submit(
            fn=_catalog_search_printings,
            inputs=[catalog_search_input],
            outputs=[printings_table, catalog_status],
        )
        printings_table.select(
            fn=on_printing_select,
            inputs=[printings_table],
            outputs=[selected_scryfall_id, selected_card_display, card_image_preview],
        )
        assign_btn.click(
            fn=do_assign,
            inputs=[selected_detection_id, selected_scryfall_id, printings_table, selected_card_display],
            outputs=[assign_status],
        )
        add_to_collection_btn.click(
            fn=_add_to_collection_from_ui,
            inputs=[
                selected_scryfall_id, selected_card_display, printings_table,
                coll_condition, coll_foil, coll_qty, coll_buy_price,
            ],
            outputs=[collection_status],
        )


# ---------------------------------------------------------------------------
# Tab 6: Auswertung (Ground-Truth Labeling & Evaluation)
# ---------------------------------------------------------------------------


def _gt_load_all():
    """Return all ground-truth entries as table rows."""
    from mtg_scanner.evaluation import load_ground_truth
    data = load_ground_truth()
    rows = []
    for img_path, entry in data.items():
        rows.append([
            img_path,
            entry.get("expected_count", "—"),
            ", ".join(entry.get("expected_cards", [])) or "—",
        ])
    return rows


def _gt_save_label(image_path: str, count_str: str, cards_str: str):
    if not image_path.strip():
        return "Bitte Bildpfad eingeben.", _gt_load_all()
    from mtg_scanner.evaluation import label_image
    count = int(count_str) if count_str.strip().isdigit() else None
    card_list = [c.strip() for c in cards_str.split(",") if c.strip()] if cards_str.strip() else None
    label_image(image_path.strip(), expected_count=count, expected_cards=card_list)
    return f"Gespeichert: {image_path.strip()}", _gt_load_all()


def _gt_run_eval(image_path: str):
    if not image_path.strip():
        return "Bitte Bildpfad eingeben.", [], ""
    from pathlib import Path as _Path
    p = _Path(image_path.strip())
    if not p.exists():
        return f"Datei nicht gefunden: {image_path}", [], ""
    try:
        from mtg_scanner.evaluation import evaluate_scan
        from mtg_scanner.pipeline import Pipeline
        pipeline = Pipeline()
        result = pipeline.process_image(str(p))
        report = evaluate_scan(result)

        lines = [
            f"Bild: {report['image_path']}",
            f"Erkannt: {report['detected_count']}"
            + (f" / {report['expected_count']} erwartet" if report['expected_count'] else ""),
            f"Identifiziert: {report['recognised_count']}",
        ]
        if report['detection_rate'] is not None:
            lines.append(f"Erkennungsrate: {report['detection_rate']:.0%}")
        if report['expected_cards']:
            lines.append(f"Treffer: {len(report['matched'])} / {len(report['expected_cards'])}")
            lines.append(f"Precision: {report['precision']:.0%}  |  Recall: {report['recall']:.0%}")
        if report['missed']:
            lines.append(f"Fehlend: {', '.join(report['missed'])}")
        if report['extra']:
            lines.append(f"Falsch erkannt: {', '.join(report['extra'])}")

        summary = "\n".join(lines)

        # Build patch gallery
        gallery_items = []
        for card in result.cards:
            pil_img = _patch_to_pil(card.patch.image)
            if pil_img is not None:
                name = card.card_name or "Unbekannt"
                conf = card.recognition_confidence
                in_gt = name.lower() in [c.lower() for c in report.get("expected_cards", [])]
                tag = "✓" if in_gt else ("?" if not report["expected_cards"] else "✗")
                gallery_items.append((pil_img, f"{tag} {name} ({conf:.0%})"))

        # Comparison table rows
        table_rows = []
        all_names = sorted(set(
            [c.lower() for c in report["expected_cards"]]
            + [c.lower() for c in report["detected_cards"]]
        ))
        for name in all_names:
            in_exp = name in [c.lower() for c in report["expected_cards"]]
            in_det = name in [c.lower() for c in report["detected_cards"]]
            status = "Treffer" if (in_exp and in_det) else ("Fehlend" if in_exp else "Extra")
            table_rows.append([name, "Ja" if in_exp else "Nein", "Ja" if in_det else "Nein", status])

        return summary, gallery_items, table_rows
    except Exception as exc:
        logger.exception("Auswertung fehlgeschlagen")
        return f"Fehler: {exc}", [], []


def _gt_prefill_from_table(evt: gr.SelectData, table_data):
    """When user clicks a row in the labels table, prefill the image path."""
    try:
        if table_data is None or len(table_data) == 0:
            return "", "", ""
        try:
            row = list(table_data.iloc[evt.index[0]])
        except AttributeError:
            row = table_data[evt.index[0]]
        img_path = str(row[0])
        count = str(row[1]) if row[1] != "—" else ""
        cards = str(row[2]) if row[2] != "—" else ""
        return img_path, count, cards
    except Exception:
        return "", "", ""




def _build_evaluation_tab():
    with gr.Tab("Auswertung"):
        gr.Markdown("## Ground-Truth Labeling & Auswertung")

        with gr.Row():
            # --- Left column: labeling form ---
            with gr.Column(scale=1):
                gr.Markdown("### Bild beschriften")
                label_path_input = gr.Textbox(
                    label="Bildpfad (absolut oder relativ)",
                    placeholder="z.B. C:/fotos/karten.jpg",
                )
                label_count_input = gr.Textbox(
                    label="Erwartete Kartenanzahl",
                    placeholder="z.B. 9",
                )
                label_cards_input = gr.Textbox(
                    label="Erwartete Kartennamen (kommagetrennt)",
                    placeholder="Lightning Bolt, Counterspell, ...",
                    lines=4,
                )
                save_label_btn = gr.Button("Label speichern", variant="primary")
                save_label_status = gr.Textbox(label="Status", interactive=False)

            # --- Right column: label table ---
            with gr.Column(scale=2):
                gr.Markdown("### Gespeicherte Labels")
                refresh_labels_btn = gr.Button("Aktualisieren")
                labels_table = gr.Dataframe(
                    headers=["Bildpfad", "Anzahl", "Kartennamen"],
                    datatype=["str", "str", "str"],
                    label="Labels",
                    interactive=False,
                )

        gr.Markdown("---")
        gr.Markdown("### Scan auswerten")

        with gr.Row():
            eval_path_input = gr.Textbox(
                label="Bildpfad zum Auswerten",
                placeholder="Zeile oben anklicken oder Pfad eingeben",
            )
            eval_btn = gr.Button("Scan + Auswertung starten", variant="primary")

        eval_summary = gr.Textbox(label="Auswertungs-Ergebnis", interactive=False, lines=8)

        with gr.Row():
            eval_gallery = gr.Gallery(
                label="Erkannte Karten (✓ Treffer / ✗ Extra / ? keine Erwartung)",
                columns=4,
                object_fit="contain",
            )
            eval_table = gr.Dataframe(
                headers=["Kartenname", "Erwartet", "Erkannt", "Status"],
                datatype=["str", "str", "str", "str"],
                label="Abgleich",
                interactive=False,
            )

        # Wire up events
        save_label_btn.click(
            fn=_gt_save_label,
            inputs=[label_path_input, label_count_input, label_cards_input],
            outputs=[save_label_status, labels_table],
        )
        refresh_labels_btn.click(
            fn=_gt_load_all,
            inputs=[],
            outputs=[labels_table],
        )
        def _prefill_and_sync_eval(evt: gr.SelectData, data):
            path, count, cards = _gt_prefill_from_table(evt, data)
            return path, count, cards, path

        labels_table.select(
            fn=_prefill_and_sync_eval,
            inputs=[labels_table],
            outputs=[label_path_input, label_count_input, label_cards_input, eval_path_input],
        )
        eval_btn.click(
            fn=_gt_run_eval,
            inputs=[eval_path_input],
            outputs=[eval_summary, eval_gallery, eval_table],
        )


# ---------------------------------------------------------------------------
# Tab 7: Archiv
# ---------------------------------------------------------------------------


def _archive_stats_text():
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.image_archive import ImageArchive
        cfg = get_config()
        if not Path(cfg.archive.db_path).exists():
            return "Noch keine Bilder archiviert."
        arch = ImageArchive(db_path=cfg.archive.db_path, index_path=cfg.archive.index_path)
        s = arch.stats()
        arch.close()
        return (
            f"Bilder: {s['total_images']}  |  "
            f"Originalgröße: {s['original_size_mb']:.1f} MB  |  "
            f"Archivgröße: {s['stored_size_mb']:.1f} MB  |  "
            f"Einsparung: {s['space_saved_mb']:.1f} MB ({100 - s['compression_ratio']*100:.0f}%)"
        )
    except Exception as exc:
        return f"Fehler: {exc}"


def _archive_load_list():
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.image_archive import ImageArchive
        cfg = get_config()
        if not Path(cfg.archive.db_path).exists():
            return []
        arch = ImageArchive(db_path=cfg.archive.db_path, index_path=cfg.archive.index_path)
        images = arch.list_images(limit=200)
        arch.close()
        rows = []
        for img in images:
            orig_mb = (img["original_size_bytes"] or 0) / 1_048_576
            stored_mb = (img["stored_size_bytes"] or 0) / 1_048_576
            rows.append([
                img["id"],
                img["captured_at"][:19].replace("T", " ") if img["captured_at"] else "",
                Path(img["original_filename"]).name if img["original_filename"] else "—",
                f"{img['image_width']}x{img['image_height']}" if img["image_width"] else "—",
                f"{orig_mb:.2f} MB",
                f"{stored_mb:.2f} MB",
            ])
        return rows
    except Exception as exc:
        logger.warning("Archiv-Liste fehlgeschlagen: %s", exc)
        return []


def _archive_load_thumbnail(evt: gr.SelectData, table_data):
    try:
        if table_data is None or len(table_data) == 0:
            return None, "Kein Bild ausgewählt."
        try:
            row = list(table_data.iloc[evt.index[0]])
        except AttributeError:
            row = table_data[evt.index[0]]
        archive_id = int(row[0])
        from mtg_scanner.config import get_config
        from mtg_scanner.image_archive import ImageArchive
        from PIL import Image as PILImage
        cfg = get_config()
        arch = ImageArchive(db_path=cfg.archive.db_path, index_path=cfg.archive.index_path)
        thumb_bytes = arch.get_thumbnail(archive_id)
        arch.close()
        if thumb_bytes is None:
            return None, "Kein Thumbnail verfügbar."
        pil_img = PILImage.open(io.BytesIO(thumb_bytes))
        info = f"Archiv-ID: {archive_id}  |  {row[2]}  |  {row[3]}  |  Original: {row[4]}  |  Archiv: {row[5]}"
        return pil_img, info
    except Exception as exc:
        return None, f"Fehler: {exc}"


def _archive_export(dest: str):
    try:
        from mtg_scanner.config import get_config
        from mtg_scanner.image_archive import ImageArchive
        cfg = get_config()
        if not Path(cfg.archive.db_path).exists():
            return "Kein Archiv gefunden."
        arch = ImageArchive(db_path=cfg.archive.db_path, index_path=cfg.archive.index_path)
        dest_path = dest.strip() or "output/archive_export"
        count = arch.export_all(dest_path)
        arch.close()
        return f"{count} Bilder exportiert nach: {dest_path}"
    except Exception as exc:
        return f"Fehler beim Export: {exc}"


def _build_archive_tab():
    with gr.Tab("Archiv"):
        gr.Markdown("## Bild-Archiv")
        with gr.Row():
            stats_box = gr.Textbox(
                value=_archive_stats_text,
                label="Statistiken",
                interactive=False,
            )
            refresh_btn = gr.Button("Aktualisieren")

        archive_table = gr.Dataframe(
            headers=["ID", "Datum", "Dateiname", "Auflösung", "Original", "Archiv"],
            datatype=["number", "str", "str", "str", "str", "str"],
            label="Archivierte Bilder (neueste zuerst)",
            interactive=False,
        )

        with gr.Row():
            thumb_preview = gr.Image(label="Vorschau", type="pil", height=300)
            thumb_info = gr.Textbox(label="Details", interactive=False, lines=3)

        gr.Markdown("### Export")
        with gr.Row():
            export_dest = gr.Textbox(
                label="Zielverzeichnis",
                value="output/archive_export",
                placeholder="output/archive_export",
            )
            export_btn = gr.Button("Alle exportieren")
        export_status = gr.Textbox(label="Export-Status", interactive=False)

        refresh_btn.click(
            fn=lambda: (_archive_stats_text(), _archive_load_list()),
            inputs=[],
            outputs=[stats_box, archive_table],
        )
        archive_table.select(
            fn=_archive_load_thumbnail,
            inputs=[archive_table],
            outputs=[thumb_preview, thumb_info],
        )
        export_btn.click(
            fn=_archive_export,
            inputs=[export_dest],
            outputs=[export_status],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns:
        Configured :class:`gradio.Blocks` instance (not yet launched).
    """
    with gr.Blocks(title="MTG Card Scanner") as app:
        gr.Markdown("# MTG Card Scanner")
        _build_scanner_tab()
        _build_history_tab()
        _build_nachkontrolle_tab()
        _build_hash_db_tab()
        _build_training_tab()
        _build_catalog_tab()
        _build_evaluation_tab()
        _build_archive_tab()
        _build_settings_tab()
    return app


def launch_ui(share: bool = False, port: int = 7860) -> None:
    """Launch the Gradio web interface.

    Args:
        share: When ``True``, create a public Gradio share link.
        port: Local port to listen on.
    """
    app = create_ui()
    app.launch(server_port=port, share=share)
