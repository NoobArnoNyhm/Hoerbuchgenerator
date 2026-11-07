# Umsetzungsfortschritt

## 2026-11-07

### Erledigt

- Python-Paketgrundgeruest mit FastAPI-Anwendung und separat unter `/ui`
  eingehangener Gradio-Oberflaeche angelegt.
- Versionierter Status-Endpunkt, Initialisierung der persistenten
  Datenverzeichnisse sowie Jobstatus-/Abbruchrouten umgesetzt.
- Globalen atomaren Job-Slot mit Busy-Ausnahme, Statushaltung und
  Slotfreigabe nach Abschluss implementiert.
- Modellunabhaengige Slug-Erzeugung (deutsche Umlaute, Windows-Reservierungen)
  und Textaufteilung mit Satz-/Wortgrenzen implementiert.
- Audio-Hilfsmodul fuer WAV/MP3-Uploadpruefung, FFprobe-Metadatenvalidierung,
  PCM-Mono-24-kHz-Normalisierung und atomare Dateiveroeffentlichung angelegt.
- Referenzfenster-Auswahl auf Whisper-Wortzeitstempeln implementiert; sie
  priorisiert komplette Wortgrenzen in 20-40 Sekunden und nimmt bei kuerzeren
  Aufnahmen den laengsten verfuegbaren vollstaendigen Ausschnitt. Ausschnitt-
  Extraktion und Ergebnisformat werden mit FFmpeg validiert.
- Optionalen faster-whisper-Adapter fuer ausschliesslich lokale, vollstaendige
  Modellverzeichnisse angelegt. Wortzeitstempel/VAD, erkannter Sprachcode und
  Sprachwahrscheinlichkeit werden uebernommen; danach wird das ausgewaehlte
  Referenzsample nochmals transkribiert.
- Optionalen Qwen3-TTS-Adapter fuer die beiden Base-Modelle angelegt; lokale
  Ladepfade, `local_files_only` und temporaerer Hugging-Face-Offline-Modus sind
  erzwungen. Prompttensoren werden atomar und versionsgebunden als NPZ mit
  JSON-Metadaten gespeichert, ohne Pickle, und beim Laden auf das Modellgeraet
  uebertragen.
- Tests fuer API-Status, Gradio-Mount, Verzeichnisse, Job-Konkurrenz, Slugs und
  Textaufteilung angelegt; Audio-Unit- und optionale FFmpeg-Integrationstests
  ergaenzt.
- Isolierte Docker-Testumgebung mit Python 3.12, FFmpeg/FFprobe,
  schreibbarem Projekt-Bind-Mount (`/workspace`) und netzlosem Testlauf
  dokumentiert.
- Offizielle Gradio-Dokumentation geprueft: FastAPI-Mount ueber
  `mount_gradio_app`; direkte Gradio-API-Aufrufe werden mit
  `Blocks.queue(api_open=False)` gesperrt. Monitoring, MCP, Run-History und
  Analytics sind im Prototyp deaktiviert.
- Offizielle Qwen3-TTS-Beispiele bestaetigen `create_voice_clone_prompt` und
  `generate_voice_clone` fuer Base-Modelle.

### Testergebnisse und Blocker

- Docker Compose Konfiguration: bestanden.
- Testimage auf Python 3.12 mit Gradio 5.50.0 gebaut; FFmpeg/FFprobe sind im
  Container vorhanden.
- Nach Whisper- und Qwen-Adapter-Ergaenzung: **40 bestanden** mit Fake-Modell-
  und Prompt-Roundtrip-Tests, Offline-Ladepruefung und FFmpeg-Integration.
  Eine nicht blockierende Starlette-Deprecation-Warnung bleibt. Echte
  Modellgewichte wurden nicht geladen.
- Gradio-Mount wird im Integrationstest geprueft; UI im Browser und
  Audio-Upload/-Wiedergabe mit echten Modellablaeufen sind noch nicht geprueft.
- Qwen-Kompatibilitaet, Prompt-Roundtrip mit echten Tensoren, Whisper-Inferenz
  mit echten Gewichten, CUDA/VRAM, Worker-Abbruch und Modellneustart sind nicht
  auf Zielhardware geprueft. Es gibt noch keinen Modell-Downloadschritt oder
  UI-/API-Integration.

### Naechste Schritte

1. Qwen- und faster-whisper-Adapter mit echten lokalen Modellgewichten und
  Zielhardware verifizieren; Ladezeit, Ressourcen und Prompt-Roundtrip messen.
2. Whisper-Adapter an Upload-/Stimmenablauf anbinden; Sprache,
  Segmentfehler sowie kurze/noisy Referenzen pruefen.
3. Implementieren: Konfiguration, Worker-Prozess, Stimmenpersistenz und echte
  asynchrone Auftragsannahme.