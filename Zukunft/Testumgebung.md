# Docker-Testumgebung

Die Tests laufen in einem Linux-Container auf Python 3.12. Das Repository wird
zur Laufzeit schreibbar nach `/workspace` eingebunden. Quellcode- und
Testaenderungen sind dadurch ohne Image-Neubau sichtbar. Die Image-Abhaengigkeiten
werden aus `pyproject.toml` installiert; FFmpeg/FFprobe kommen aus dem
Python-Basisimage und werden fuer Audiotests mitinstalliert.

## Voraussetzungen

- Docker Desktop mit Linux-Containern unter Windows oder Docker Engine unter
  Linux
- Docker Compose v2
- Netzwerkzugriff beim erstmaligen Image-Build fuer apt- und Python-Pakete

Die Testcontainer laufen mit `network_mode: none`, ohne Host-Ports und ohne
Modellzugriff. Modell-Downloads sind fuer die Tests nicht erforderlich.

## Tests ausfuehren

Im Repository-Verzeichnis:

```powershell
docker compose -f compose.test.yaml build
docker compose -f compose.test.yaml run --rm test
```

Das funktioniert mit denselben Befehlen in PowerShell und einer Linux-Shell.
Nach Aenderungen an `pyproject.toml` das Image neu bauen. Quellcode und Tests
werden durch den Bind-Mount direkt aus dem Arbeitsverzeichnis gelesen.

Einen einzelnen Test ausfuehren:

```powershell
docker compose -f compose.test.yaml run --rm test python -m pytest -q tests/test_audio.py
```

Eine Shell im Testcontainer oeffnen:

```powershell
docker compose -f compose.test.yaml run --rm --entrypoint bash test
```

## Verifizierter Lauf

Am 2026-11-07 wurde das Image gebaut und die komplette Testsuite im Container
ausgefuehrt: **33 bestanden**. Der FFmpeg-/FFprobe-Integrationstest lief dabei
mit. Es gab eine nicht blockierende Starlette-Deprecation-Warnung.

## Aufbau

- `Dockerfile.test`: Python-/FFmpeg-Image und Installation des Projekts samt
  Entwicklungsabhaengigkeiten.
- `compose.test.yaml`: schreibbarer Projekt-Bind-Mount nach `/workspace`,
  temporaerer Datenpfad und vom Netz getrennter Testlauf.
- `.dockerignore`: schliesst Git-Metadaten, lokale virtuelle Umgebungen und
  Nutzdaten aus dem Build-Kontext aus.

Dieses Image ist ausschliesslich fuer Tests gedacht. Es enthaelt keine Qwen-
oder Whisper-Modelle und ersetzt nicht das spaetere Laufzeit-Image.