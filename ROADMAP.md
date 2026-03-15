# MTG Card Scanner — Roadmap & Ideen

Dokumentiert am 2026-03-15. Noch nicht implementiert.

---

## Priorität 1 — Sofort wertvoll

### Sammlungsverwaltung (`collection.db`)
Das fehlende Herzstück. Aktuell gibt es nur einen Scan-Log, aber kein Konzept von "ich besitze diese Karte".
- `collection.db`: pro Karte `scryfall_id`, Anzahl, Zustand (NM/LP/MP/HP), foil, Kaufpreis, Kaufdatum
- Beim Scan-Ergebnis direkt "zur Sammlung hinzufügen"
- Duplikaterkennung: "Du hast bereits 4x Lightning Bolt"
- Gesamtwert der Sammlung
- Alles andere (Preishistorie, Export, Duplikate) baut darauf auf

### Kartenbilder im Zuordnungs-Picker
Scryfall liefert Bild-URLs im Katalog — diese direkt neben dem Patch anzeigen (kein Download, nur URL laden). Verwandelt die Zuordnung von "ich lese die Beschreibung" zu "ich sehe sofort ob es die richtige Karte ist".

### Export in gängige MTG-Tools
- **Moxfield / Archidekt**: CSV mit `Count,Name,Set,CollectorNumber,Foil`
- **TCGplayer**: Eigenes CSV-Format
- **MTG Arena**: `.dek`-Format
- **Cardmarket**: XML/CSV

---

## Priorität 2 — Mittelfristig (wenn Datenmenge wächst)

### Preishistorie
Scryfall-Preise täglich in `price_history`-Tabelle schreiben.
- Wertentwicklung der Sammlung über Zeit
- Karten die stark gestiegen/gefallen sind
- Graphen in der UI

### Hash-DB aufbauen und testen
Die Hash-Erkennungspipeline existiert, aber die DB ist leer. Einmaliger ~50-min-Download.
- Zuverlässigster Erkennungsweg besonders für JA-Karten
- Logischer nächster Schritt sobald Katalog steht

### Offline-Lookups gegen `card_catalog.db`
Aktuell alle Scryfall-Lookups gegen die API. Alternative: direkt gegen den lokalen Katalog.
- Scans ohne Internet möglich
- Rate-Limiting-Problem gelöst
- Preise aus dem Katalog (täglich aktualisiert statt per API)

### Wunschliste / Tauschbörse
- "Ich suche diese Karten" — Want-List
- "Ich habe diese doppelt" — Have-List / Trade-List
- Export beider Listen

---

## Priorität 3 — Langfristig (wenn genug Trainingsdaten)

### YOLO Fine-Tuning mit eigenen Daten
Die bestätigten Patch-Zuordnungen (Karten-Zuordnung-Tab) sind Trainingsmaterial.
- Export der bestätigten Patches mit `scryfall_id`-Labels
- Fine-Tuning eines YOLO-Modells auf echten Fotos
- Robuster als OCR bei schlechten Lichtverhältnissen und Winkeln

### Hash-DB aus eigenen Fotos verbessern
Eigene pHashes aus echten Kamerafotos statt Scryfall-Renderbildern.
- Robuster bei Lichtverhältnissen, Reflexionen, leichtem Verschleiß
- Active Learning: welche Karten haben wenige/keine Patches → gezielt scannen

### REST API
Pipeline als FastAPI-Dienst.
- Mobile-App oder Browser-Extension kann darauf zugreifen
- Andere Tools können die Erkennung nutzen

### Batch-Parallelisierung
Directory-Scans sind sequenziell. Detection + Archivierung könnten parallelisiert werden (Scryfall-Lookups nicht wegen Rate-Limit).

---

## Technische Schulden / Verbesserungen

- **Card-back-Erkennung**: Karten die mit der Rückseite nach oben liegen werden fälschlicherweise erkannt
- **Konfidenz-Feedback**: Korrekturen könnten Fuzzy-Search-Gewichtung verbessern
- **Mobile-optimierte UI**: Gradio funktioniert auf Mobile, könnte aber besser sein
- **Bulk-Assignment**: Beim Scannen eines bekannten Sets (z.B. Commander-Deck) Katalogsuche auf das Set vorfiltern
