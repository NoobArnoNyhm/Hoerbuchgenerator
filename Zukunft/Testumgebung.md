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
ausgefuehrt: **53 bestanden**. Der FFmpeg-/FFprobe-Integrationstest lief dabei
mit. Es gab eine nicht blockierende Starlette-Deprecation-Warnung.

## Modell-Smoke-Test auf GPU

Am 2026-11-07 wurden die offiziellen Qwen-0.6B-Base- und
faster-whisper-base-Gewichte lokal gespeichert und in einem separaten
GPU-Container mit Offline-Ladeflags geprueft. Auf einer RTX 3080 Laptop GPU
erzeugte faster-whisper aus synthetischer Sprache ein Transkript mit 108
Wortzeitstempeln; ein 30-Sekunden-Referenzausschnitt wurde extrahiert und
erneut transkribiert. Qwen lud mit BF16/SDPA in 136 Sekunden, speicherte und
lud einen echten Prompt als NPZ und synthetisierte daraus 5,04 Sekunden
Audio. Die gemessene Spitzenbelegung betrug 2,8 GiB CUDA-Speicher. Ein FP16-
Syntheselauf scheiterte an ungueltigen Sampling-Wahrscheinlichkeiten; BF16
funktionierte. Dieser Smoke-Test verwendet kuenstlich erzeugte Sprache und
belegt weder die Qualitaet mit menschlichen Referenzaufnahmen noch den
integrierten UI-/API- oder Worker-Ablauf.

faster-whisper 1.2.1 funktioniert nicht mit PyAV 19; die optionale
Whisper-Abhaengigkeit pinnt PyAV daher auf `<19`.

Der Stimmen-Upload-Vorbereitungspfad wurde mit echten Whisper-Base-Gewichten
und lokal erzeugter Sprache geprueft: Eine kurze Referenz von 11,72 Sekunden
und eine mit leisem rosa Rauschen ueberlagerte Referenz von 11,70 Sekunden
wurden jeweils mit Sprache `en` transkribiert. Eine stille Aufnahme wurde
abgewiesen; temporaere Dateien wurden bereinigt. Es wurden keine menschlichen
Sprachaufnahmen verwendet.

Der API- und Profilablauf wurde anschliessend mit lokalen Whisper-Base- und
Qwen-0.6B-Base-Gewichten vollstaendig ausgefuehrt. Das temporaere Profil wurde
erst nach erfolgreicher NPZ-Promptserialisierung und Beispielgenerierung
atomar unter dem Slug veroeffentlicht. Das Beispielaudio ist 6,08 Sekunden
lang, PCM-Mono mit 24 kHz. Qwen-Ladezeit: 58,4 Sekunden; Profilablauf nach
Modellstart: 15,2 Sekunden. Auch dieser Test verwendete ausschliesslich
synthetisch erzeugte Sprache.

Die asynchrone FastAPI-Route wurde danach mit automatischer lokaler
Modellinitialisierung ausgefuehrt: Status meldete Qwen und Whisper als geladen,
der Multipart-Upload erhielt HTTP 202 und der Job lieferte ein vollstaendiges,
unter dem erwarteten Slug abgelegtes Stimmenprofil. Gesamtlauf inklusive
Modellstart: 170,3 Sekunden. Abbruch waehrend einer laufenden Modellinferenz
bleibt kooperativ und beendet den Inferenzaufruf nicht vorzeitig.

Modellvergleich am 2026-11-07 mit denselben Offline-Bedingungen und
synthetischer englischer Sprache: faster-whisper `medium` lud in 12,2 s und
transkribierte in 14,6 s (Sprache `en`, Wahrscheinlichkeit 0,996, 74 Woerter);
`small` lud in 3,8 s und transkribierte in 5,4 s (Sprache `en`,
Wahrscheinlichkeit 0,981, 74 Woerter). Qwen 1.7B Base lud mit BF16/SDPA auf
der RTX 3080 Laptop GPU in 232,8 s; CUDA-Spitzenwerte lagen bei 3,91 GiB
alloziert und 4,04 GiB reserviert. Ein vollstaendiger Stimmenprofilablauf
einschliesslich Qwen-Beispielaudio mit Whisper small und Qwen 1.7B war
erfolgreich (GPU-Container Exit-Code 0). Damit ist die technische
Modellkompatibilitaet geprueft, nicht die Sprachqualitaet mit menschlichen
Referenzaufnahmen.

## Aufbau

- `Dockerfile.test`: Python-/FFmpeg-Image und Installation des Projekts samt
  Entwicklungsabhaengigkeiten.
- `compose.test.yaml`: schreibbarer Projekt-Bind-Mount nach `/workspace`,
  temporaerer Datenpfad und vom Netz getrennter Testlauf.
- `.dockerignore`: schliesst Git-Metadaten, lokale virtuelle Umgebungen und
  Nutzdaten aus dem Build-Kontext aus.

Dieses Image ist ausschliesslich fuer Tests gedacht. Es enthaelt keine Qwen-
oder Whisper-Modelle und ersetzt nicht das spaetere Laufzeit-Image.