# Hoerbuchgenerator

Offline-Webanwendung fuer lokale Qwen3-TTS-Stimmklonierung und
Whisper-Transkription. Der Prototyp bietet einen asynchronen API-Ablauf zum
Anlegen von Stimmenprofilen mit Transkript, wiederverwendbarem Qwen-Prompt und
Beispielaudio. Hoerbuchsynthese, Gradio-Ablauf, isolierter Worker und
automatischer Modelldownload sind noch nicht implementiert.

## Voraussetzungen

- Python 3.11 oder neuer
- FFmpeg und FFprobe muessen fuer die Audioverarbeitung im `PATH` liegen

## Lokaler Prototyp

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,whisper,qwen]"
uvicorn hoerbuchgenerator.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Die UI ist unter <http://127.0.0.1:8000/ui> erreichbar. Der versionierte
Status-Endpunkt ist <http://127.0.0.1:8000/api/v1/status>. Datenverzeichnisse
werden beim Start unter `data/` erstellt; `APP_DATA_DIR` kann den Pfad
ueberschreiben. Liegen beide lokalen Modelle in `data/models/`, werden Qwen
`Qwen3-TTS-12Hz-0.6B-Base` und Whisper `base` beim Start geladen. Die Auswahl
kann ueber `QWEN_MODEL` und `WHISPER_MODEL`, das Whisper-Geraet ueber
`WHISPER_DEVICE` und `WHISPER_COMPUTE_TYPE` konfiguriert werden. Qwen verwendet
standardmaessig CUDA, BF16 und SDPA; `QWEN_DEVICE_MAP` kann das Geraet anpassen.

Stimme per REST anlegen:

```powershell
curl.exe -F "name=Meine Stimme" -F "audio=@referenz.wav" http://127.0.0.1:8000/api/v1/voices
```

Die Annahme liefert HTTP 202 mit einer Job-ID. Den Abschluss und das Ergebnis
liefert `GET /api/v1/jobs/{id}`. Profile werden erst nach erfolgreicher
Transkription, Promptserialisierung und Beispielaudioerzeugung atomar unter
`data/voices/<slug>/` sichtbar. Ein Abbruch waehrend einer laufenden
Modellinferenz wird nach Ende dieser Inferenz ausgewertet; sie wird noch nicht
durch einen separaten Worker-Prozess beendet.

## Tests

Empfohlen ist die dokumentierte, isolierte Docker-Testumgebung unter
[Zukunft/Testumgebung.md](Zukunft/Testumgebung.md). Das Repository wird dort
als Volume unter `/workspace` eingebunden:

```powershell
docker compose -f compose.test.yaml build
docker compose -f compose.test.yaml run --rm test
```

Alternativ lokal mit Python:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

Der Stimmenablauf wurde mit lokalen Qwen-0.6B-Base- und
faster-whisper-base-Gewichten auf einer GPU geprueft. Der API-Auftrag ist
implementiert; die Gradio-Oberflaeche verwendet ihn noch nicht. Modellgewichte
sind nicht versioniert, und die Anwendung ist noch nicht fuer den
Offline-Abnahmebetrieb abgenommen. Den Umsetzungsstand und offene Pruefungen
enthaelt [Zukunft/Fortschritt.md](Zukunft/Fortschritt.md).

Die faster-whisper- und Qwen-Adapter sind optionale Extras und laden nur
ausgewaehlte, vollstaendige Modellverzeichnisse aus lokalen Pfaden. Die
Abhaengigkeiten koennen fuer Adapterentwicklung mit
`python -m pip install -e ".[whisper,qwen]"` installiert werden.
Qwen-Prompts werden als NPZ-Datei ohne Pickle gespeichert. Die Pruefung mit
Qwen 1.7B, weiteren Whisper-Modellen und realen Referenzaufnahmen sowie der
Worker-Prozess und die Hoerbuchsynthese stehen noch aus.