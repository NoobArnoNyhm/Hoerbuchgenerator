# Hoerbuchgenerator

Offline-Webanwendung fuer lokale Qwen3-TTS-Stimmklonierung und
Whisper-Transkription. Der Prototyp bietet einen asynchronen API-Ablauf zum
Anlegen von Stimmenprofilen mit Transkript, wiederverwendbarem Qwen-Prompt und
Beispielaudio. Die Gradio-Oberflaeche und die versionierte API verwenden
denselben Stimmenauftrag und Jobstatus. Qwen und Whisper laufen in einem
separaten Worker-Prozess. Hoerbuchsynthese ist verfuegbar; automatischer
Modelldownload erfolgt ueber den separaten Online-Initialisierungsschritt.

## Verantwortung und Einwilligung

Fremde Stimmen duerfen nur mit ausdruecklicher Einwilligung der betroffenen
Person fuer das Klonen und die beabsichtigte Nutzung verwendet werden.
Eine Nutzung ohne Einwilligung kann illegal sein und Persoenlichkeitsrechte
verletzen; eine Nutzung gegen den Willen der betroffenen Person ist moralisch
hoechst verwerflich. Oeffentlich zugaengliche Aufnahmen sind keine
Einwilligung. Der Tab **Stimmsynthese** zeigt diesen Hinweis unmittelbar vor
**Stimme erstellen** an. Eine tatsaechliche Einwilligung wird durch die
Anwendung nicht technisch geprueft.

## Docker-Laufzeit

Fuer die GPU-Laufzeit werden Docker Compose v2 und eine von Docker erreichbare
NVIDIA-GPU mit passendem NVIDIA Container Toolkit benoetigt. Die Modellgewichte
werden separat initialisiert und nicht in das Image eingebaut.

```powershell
Copy-Item .env.example .env
.\initialize.ps1
```

Unter Linux `.env` mit `cp .env.example .env` anlegen und einmalig
`sh initialize.sh` ausfuehren. Beide Initialisierungsskripte bauen das
Runtime-Image, initialisieren die in `.env` ausgewaehlten Modelle und starten
den App-Container mit Compose. Fuer den Modellabruf ist Netzwerkzugriff auf
Hugging Face erforderlich. Falls Modellzugriff ein Token benoetigt, `HF_TOKEN`
nur in der lokalen `.env` setzen. Der Runtime-Container verwendet die lokalen
Modell-Caches, den Hugging-Face-/Transformers-Offline-Modus und deaktivierte
Telemetrie; automatische Modellabrufe sind im Betrieb damit abgeschaltet.
Docker blockiert ausgehende Netzwerkverbindungen nicht. Der App-Port wird nur
an `127.0.0.1:${APP_PORT}` gebunden; der Healthcheck wird nach dem Modellstart
`healthy`.

Beim App-Start erscheint im Dockerlog zusaetzlich zur Uvicorn-Bind-Adresse
die Browseradresse `http://127.0.0.1:8000/ui`. Bei abweichendem `APP_PORT`
zeigt sie den konfigurierten Host-Port an; der Container lauscht intern
weiterhin auf Port 8000. Die Adresse gilt fuer den Docker-Host, nicht fuer
Remote-Hosts oder Reverse-Proxys, und ist keine Meldung ueber
Modellbereitschaft. Bei direktem Uvicorn-Start mit abweichendem Port muss
`APP_PORT` fuer die Loganzeige passend gesetzt werden; die Variable aendert
nicht den Uvicorn-Listen-Port.

Der weitere Betrieb erfolgt direkt mit Docker Compose, ohne zusaetzliche
Skripte:

```powershell
docker compose ps
docker compose logs hoerbuchgenerator
docker compose down
```

Stimmen, Ausgaben, Modelle und Konfiguration bleiben unter `data/` erhalten;
Backups dieses Verzeichnisses sind die Verantwortung des Betreibers.

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
Nicht unterstuetzte Modellnamen brechen den Start mit einer konkreten Meldung
ab. Fehlen lokale Modell-Caches, startet die Anwendung ohne Modellservice,
protokolliert die erwarteten Cachepfade und meldet den Status als
`not_initialized`; es werden keine Modelle heruntergeladen. Waehrend der
Anwendungslaufzeit erzwingt sie Hugging-Face-/Transformers-Offline-Modus und
deaktiviert Hub-Telemetrie; die vorherigen Prozessumgebungswerte werden beim
Herunterfahren wiederhergestellt. Ein Modellworker startet erst, wenn die
erforderlichen lokalen Konfigurations-, Gewichts- und Tokenizerdateien
vorhanden und nicht leer sind; fehlende, leere oder nicht lesbare Dateien
werden protokolliert. Verwaiste Dateien und Auftragsverzeichnisse in
`data/work/` werden beim Start bereinigt; Stimmenprofile und fertige Ausgaben
bleiben dabei unangetastet.

In der UI koennen Sie einen Stimmenname und eine WAV-/MP3-Referenzaufnahme
angeben. Der Auftrag wird angenommen, der gemeinsame Status alle zwei Sekunden
aktualisiert und nach erfolgreichem Abschluss das synthetisierte Beispielaudio
angezeigt. Fuer die Synthese waehlen Sie ein vorhandenes Stimmenprofil aus der
automatisch aktualisierten Stimmenauswahl; das zugehoerige Beispielaudio wird
in der UI als Vorschau abgespielt. Die API-Aufrufe bleiben weiterhin verfuegbar:

Stimme per REST anlegen:

```powershell
curl.exe -F "name=Meine Stimme" -F "audio=@referenz.wav" http://127.0.0.1:8000/api/v1/voices
```

Die Annahme liefert HTTP 202 mit einer Job-ID. Den Abschluss und das Ergebnis
liefert `GET /api/v1/jobs/{id}`. Profile werden erst nach erfolgreicher
Transkription, Promptserialisierung und Beispielaudioerzeugung atomar unter
`data/voices/<slug>/` sichtbar. Ein Abbruch waehrend einer laufenden
Modellinferenz beendet den Worker-Prozess sofort. Der Worker wird anschliessend
neu gestartet und laedt beide lokalen Modelle erneut; waehrenddessen bleibt der
globale Auftragsslot belegt. Nicht veroeffentlichte Profil- und Arbeitsdateien
des abgebrochenen Auftrags werden entfernt.

Vorhandene Profile samt Sprache, Modell und Beispielaudio-URL liefert
`GET /api/v1/voices`. Das Beispiel ist ueber
`GET /api/v1/voices/{slug}/example` abrufbar; Transkript und Referenzaufnahme
werden nicht in der Stimmenliste ausgegeben.

Kurzen Text als WAV synthetisieren und Ergebnis herunterladen:

```powershell
curl.exe -F "text=Ein kurzer Beispieltext." -F "name=Kapitel Eins" `
  -F "voice=meine_stimme" -F "format=wav" `
  http://127.0.0.1:8000/api/v1/synthesis/text
curl.exe -o kapitel_eins.wav `
  http://127.0.0.1:8000/api/v1/output/kapitel_eins.wav
```

Mehrere UTF-8-Textdateien koennen gesammelt ueber
`POST /api/v1/synthesis/files` mit wiederholten `files=@kapitel.txt`-Feldern
eingereicht werden. `voice` ist der Stimmen-Slug; `language` ist optional und
akzeptiert einen Sprachcode wie `de` oder `en` (standardmaessig die erkannte
Stimmensprache). `format` ist `wav` oder `mp3` und standardmaessig `mp3`.
Chunks werden sequenziell synthetisiert, mit EBU-R128-Lautheitsnormalisierung
auf -20 LUFS (True Peak -2 dBTP) und getesteten Pausen zusammengefuegt und erst
nach Validierung atomar in `data/output/` veroeffentlicht.

## Modulaufteilung

Der ASGI-Einstieg bleibt `hoerbuchgenerator.main:app`; `create_app` bleibt
ebenfalls aus `main` importierbar. Die Implementierung ist nach
Verantwortlichkeiten aufgeteilt:

- [app_factory.py](src/hoerbuchgenerator/app_factory.py) verdrahtet API, UI und
  Auftragsdienste; [app_lifecycle.py](src/hoerbuchgenerator/app_lifecycle.py)
  verwaltet Offline-Umgebung, Verzeichnisse und Worker-Lebenszyklus.
- [voice_jobs.py](src/hoerbuchgenerator/voice_jobs.py) und
  [synthesis_jobs.py](src/hoerbuchgenerator/synthesis_jobs.py) nehmen Auftraege
  entgegen und steuern deren Ausfuehrung ueber denselben `JobManager`.
- [worker.py](src/hoerbuchgenerator/worker.py) steuert den Prozess;
  [worker_runtime.py](src/hoerbuchgenerator/worker_runtime.py) laedt im
  Kindprozess die Modelle und verarbeitet Befehle.
  [worker_synthesizer.py](src/hoerbuchgenerator/worker_synthesizer.py) verbindet
  die synchrone Synthesepipeline mit der asynchronen Worker-Schnittstelle.
- [audio_types.py](src/hoerbuchgenerator/audio_types.py) enthaelt gemeinsame
  Typen und Konstanten, [audio_reference.py](src/hoerbuchgenerator/audio_reference.py)
  die Referenzfensterwahl und [audio_ffmpeg.py](src/hoerbuchgenerator/audio_ffmpeg.py)
  die Audioverarbeitung. Bisherige oeffentliche Imports aus `audio` bleiben
  verfuegbar.
- [ui.py](src/hoerbuchgenerator/ui.py) baut Layout und Event-Verbindungen auf;
  [ui_callbacks.py](src/hoerbuchgenerator/ui_callbacks.py) nimmt Eingaben an und
  [ui_status.py](src/hoerbuchgenerator/ui_status.py) projiziert Status,
  Stimmenauswahl und Vorschau. `create_ui_callbacks` bleibt aus `ui`
  importierbar.

Die Abhaengigkeiten zeigen von der App-Komposition zu UI/API und
Auftragsdiensten, weiter zu Worker bzw. Fachmodulen; die unteren Module
importieren weder `main` noch die App-Factory. Gemeinsam genutzte Audio-Typen
haengen nicht von Audioverarbeitung oder Modellen ab.

## Tests

Empfohlen ist die dokumentierte, isolierte Docker-Testumgebung unter
[Testumgbung/Testumgebung.md](Testumgbung/Testumgebung.md). Das Repository wird dort
als Volume unter `/workspace` eingebunden:

```powershell
docker compose -f Testumgbung/compose.test.yaml build
docker compose -f Testumgbung/compose.test.yaml run --rm test
```

Alternativ lokal mit Python:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

Der Stimmenablauf wurde mit lokalen Qwen-0.6B-Base- und
faster-whisper-base-Gewichten auf einer GPU geprueft. API und Gradio nutzen
denselben Jobslot; UI-Status, Abbruch und Beispielaudio werden aus dem
gemeinsamen Auftrag aktualisiert. Modellgewichte sind nicht versioniert, und
die Anwendung ist noch nicht fuer den Offline-Abnahmebetrieb abgenommen. Den
Umsetzungsstand und offene Pruefungen enthaelt
[Zukunft/Fortschritt.md](Zukunft/Fortschritt.md).

Die faster-whisper- und Qwen-Adapter sind optionale Extras und laden nur
ausgewaehlte, vollstaendige Modellverzeichnisse aus lokalen Pfaden. Die
Abhaengigkeiten koennen fuer Adapterentwicklung mit
`python -m pip install -e ".[whisper,qwen]"` installiert werden.
Qwen-Prompts werden als NPZ-Datei ohne Pickle gespeichert. Die technische
Pruefung mit Qwen 1.7B und weiteren Whisper-Modellen sowie der Abbruch und
Neustart des Workers waehrend eines echten GPU-Auftrags sind erfolgt. WAV- und
MP3-Hoerbuchsynthese wurde ebenfalls mit lokalen Modellen geprueft. Tests mit
menschlichen Referenzaufnahmen und Abnahme der Hoerbuchqualitaet stehen noch aus.