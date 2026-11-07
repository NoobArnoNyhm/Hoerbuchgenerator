# Hoerbuchgenerator

Offline-Webanwendung fuer lokale Qwen3-TTS-Stimmklonierung und
Whisper-Transkription. Der aktuelle Stand ist ein technischer Prototyp fuer
FastAPI/Gradio, den API-Status, Auftragsverwaltung und modellunabhaengige
Kernfunktionen. Modelladapter, Arbeitsablaeufe und Containerbetrieb sind noch
nicht fertig; die Anwendung erzeugt derzeit kein Audio.

## Voraussetzungen

- Python 3.11 oder neuer
- FFmpeg und FFprobe muessen fuer die Audio-Hilfsfunktionen im `PATH` liegen;
	diese sind noch nicht mit den UI-/API-Ablaufdiensten verbunden

## Lokaler Prototyp

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn hoerbuchgenerator.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Die UI ist unter <http://127.0.0.1:8000/ui> erreichbar. Der versionierte
Status-Endpunkt ist <http://127.0.0.1:8000/api/v1/status>. Datenverzeichnisse
werden beim Start unter `data/` erstellt; `APP_DATA_DIR` kann den Pfad
ueberschreiben.

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

Die Qwen-/Whisper-Adapter wurden mit lokalen Qwen-0.6B-Base- und
faster-whisper-base-Gewichten in einem separaten GPU-Smoke-Test mit
Offline-Ladeflags geprueft. Die Upload-Vorbereitung fuer Stimmen validiert und
normalisiert Audio und ermittelt mit Whisper Referenzausschnitt, Transkript
und Sprache. Sie ist noch nicht mit Qwen, API-Auftraegen oder UI verbunden;
Modellgewichte sind nicht versioniert und eine automatisierte
Modellinitialisierung fehlt. Der Prototyp ist deshalb weder als fertige
Anwendung noch fuer den Offline-Abnahmebetrieb geeignet. Den Umsetzungsstand
und offene Pruefungen enthaelt [Zukunft/Fortschritt.md](Zukunft/Fortschritt.md).

Die faster-whisper- und Qwen-Adapter sind optionale Extras und laden nur
ausgewaehlte, vollstaendige Modellverzeichnisse aus lokalen Pfaden. Die
Abhaengigkeiten koennen fuer Adapterentwicklung mit
`python -m pip install -e ".[whisper,qwen]"` installiert werden.
Qwen-Prompts werden als NPZ-Datei ohne Pickle gespeichert. Die Pruefung mit
Qwen 1.7B, weiteren Whisper-Modellen und realen Referenzaufnahmen sowie die
Integration in die Anwendungsablaeufe stehen noch aus.