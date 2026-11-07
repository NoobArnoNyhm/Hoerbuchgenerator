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
- Nach Whisper- und Qwen-Adapter-Ergaenzung: **41 bestanden** mit Fake-Modell-
  und Prompt-Roundtrip-Tests, Offline-Ladepruefung und FFmpeg-Integration.
  Eine nicht blockierende Starlette-Deprecation-Warnung bleibt; die Suite
  verwendet weiterhin Fake-Modelle.
- Mit echten lokalen Gewichten und RTX 3080 Laptop GPU: Whisper `base`
  transkribiert ein synthetisches englisches Testaudio offline in 2,6 s
  (Sprachwahrscheinlichkeit 0,956); 108 Wortzeitstempel und die Auswahl,
  Extraktion sowie erneute Transkription eines 30-s-Referenzsamples wurden
  geprueft. Ein einzelner Null-Dauer-Zeitstempel wurde dabei gefunden und wird
  vom Whisper-Adapter verworfen.
- Qwen `Qwen3-TTS-12Hz-0.6B-Base` laedt auf derselben GPU mit BF16/SDPA in
  136 s. Der NPZ-Prompt-Roundtrip mit echten Tensoren dauerte 1,4 s; die
  Synthese erzeugte 5,04 s Audio in 16,6 s. Gemessener CUDA-Spitzenspeicher:
  2,8 GiB. Ein FP16-Lauf scheiterte mit ungueltigen Sampling-Wahrscheinlichkeiten;
  BF16 funktionierte. Die Eingabe war synthetische Sprache, kein menschliches
  Referenzaudio.
- faster-whisper 1.2.1 ist mit PyAV 19 nicht kompatibel
  (`av.open(..., metadata_errors=...)`); die Whisper-Extra-Abhaengigkeit
  begrenzt PyAV deshalb auf Versionen unter 19.
- Gradio-Mount wird im Integrationstest geprueft; UI im Browser und
  Audio-Upload/-Wiedergabe mit echten Modellablaeufen sind noch nicht geprueft.
- Die vollstaendige Anwendung nutzt die getesteten Modelle noch nicht.
  Qwen 1.7B, weitere Whisper-Groessen, reale Nutzerreferenzen, UI/API-Fluss,
  Worker-Abbruch und Modellneustart sind nicht geprueft. Ein automatisierter
  Modell-Downloadschritt fehlt.

### Naechste Schritte

1. Whisper-Adapter an Upload-/Stimmenablauf anbinden; Sprache,
  Segmentfehler sowie kurze/noisy Referenzen pruefen.
2. Qwen 1.7B und weitere Whisper-Groessen auf Zielhardware pruefen; reale
  Referenzaufnahmen sowie Worker-Abbruch und Modellneustart untersuchen.
3. Implementieren: Konfiguration, Worker-Prozess, Stimmenpersistenz und echte
  asynchrone Auftragsannahme.