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

Die Qwen-/Whisper-Modelle sind noch nicht integriert oder heruntergeladen.
Der Prototyp ist deshalb weder als fertige Anwendung noch fuer den
Offline-Abnahmebetrieb geeignet. Den Umsetzungstand und offene Pruefungen
enthaelt [Zukunft/Fortschritt.md](Zukunft/Fortschritt.md).

Der faster-whisper-Adapter ist optional und laedt ausschliesslich vollstaendige
Modellverzeichnisse aus einem lokalen Pfad. Die Modellabhaengigkeit kann fuer
Adapterentwicklung mit `python -m pip install -e ".[whisper]"` installiert
werden; Modelldownload und Anwendungsintegration stehen noch aus.