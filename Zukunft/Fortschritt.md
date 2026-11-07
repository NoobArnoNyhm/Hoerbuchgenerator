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
- Upload-Vorbereitung fuer Stimmen implementiert: Datei-Endung, Name und
  Groesse werden geprueft; der Upload wird begrenzt ins Arbeitsverzeichnis
  geschrieben, normalisiert und mit Whisper transkribiert. Referenzausschnitt,
  Transkript, erkannte Sprache und Sprachwahrscheinlichkeit bleiben fuer den
  nachfolgenden Qwen-Schritt im Arbeitsverzeichnis erhalten. Fehler entfernen
  angelegte Arbeitsdateien.
- Mit echten Whisper-Base-Gewichten auf synthetischer Sprache geprueft:
  kurze Aufnahme (11,72 s) und moderat verrauschte Aufnahme (11,70 s) ergaben
  Transkript und Sprache `en` (Wahrscheinlichkeit 0,953 bzw. 0,958).
  Stille wurde mit leerem Transkript abgewiesen und das Arbeitsverzeichnis
  bereinigt.
- Vollstaendige Docker-Testsuite nach der Upload-Ergaenzung: **46 bestanden**;
  die nicht blockierende Starlette-Deprecation-Warnung bleibt.
- Asynchronen `POST /api/v1/voices`-Auftrag und `GET /api/v1/jobs/{id}` fuer
  Stimmenanlage ergaenzt. Uploads werden vor Annahme der Aufgabe begrenzt
  gestreamt, ein aktiver Auftrag blockiert weitere Uploads mit HTTP 403. Qwen-
  und Whisper-Adapter werden beim App-Start aus den konfigurierten lokalen
  Modellverzeichnissen geladen.
- Stimmenprofile werden vor der Veroeffentlichung in einem temporaeren Ordner
  aufgebaut: Referenz-WAV, Transkript, NPZ-Prompt, Metadaten und validiertes
  Beispiel-WAV. Erst nach vollstaendiger Erzeugung wird der Profilordner
  atomar umbenannt; Duplikate, Synthesefehler und Abbruch veroeffentlichen
  kein Profil.
- Echte Whisper-/Qwen-0.6B-Ende-zu-Ende-Pruefung mit synthetischer englischer
  Sprache: Profil samt allen fuenf Dateien erstellt, Transkript und Sprache
  `en` gespeichert sowie ein 6,08-s-PCM-Mono-24-kHz-Beispiel synthetisiert.
  Qwen-Ladezeit in diesem Lauf: 58,4 s; Stimmerstellung: 15,2 s. Die
  Referenz war keine menschliche Aufnahme.
- Zusaetzlich die echte asynchrone REST-Route mit lokaler Modellinitialisierung
  durchlaufen: Status meldete beide geladenen Modelle, `POST /api/v1/voices`
  lieferte 202 und der Job endete erfolgreich mit Profil unter dem erwarteten
  Slug. Gesamtlauf mit Modellstart: 170,3 s.
- Tests fuer API-202/Jobergebnis, Modellservice-Fehler, Busy-403,
  Uploadbereinigung, atomare Profilablage, Duplikat und Fehler/Abbruch:
  **53 bestanden**; eine nicht blockierende Starlette-Deprecation-Warnung
  bleibt.
- Gradio-Oberflaeche an den gemeinsamen Stimmenauftrag angebunden:
  Stimmenname und WAV-/MP3-Upload, laufende Statusanzeige, Abbruch und
  Beispielaudio. UI und REST teilen Uploadannahme und globalen Jobslot.
- UI-Callbacks geprueft: Annahme und Abschluss eines Stimmenauftrags,
  Beispielaudio-Ausgabe sowie Ablehnung bei belegtem Jobslot. Die vollstaendige
  Docker-Testsuite besteht mit **55 Tests**; eine nicht blockierende
  Starlette-Deprecation-Warnung bleibt.
- Modellvergleich auf der RTX 3080 Laptop GPU mit synthetischer englischer
  Sprache und lokalem Offline-Cache: faster-whisper `medium` erkannte Sprache
  `en` (0,996; 74 Woerter; Laden 12,2 s, Transkription 14,6 s), `small`
  erkannte ebenfalls `en` (0,981; 74 Woerter; Laden 3,8 s, Transkription
  5,4 s). Qwen 1.7B Base lud mit BF16/SDPA auf CUDA in 232,8 s; Spitzenwerte
  lagen bei 3,91 GiB alloziertem und 4,04 GiB reserviertem CUDA-Speicher.
  Der vollstaendige Stimmenprofilablauf mit Whisper small und Qwen 1.7B
  einschliesslich Beispielaudio war erfolgreich. Der gezielte GPU-Container
  endete mit Exit-Code 0.
- Separaten, dauerhaft laufenden Modell-Worker implementiert: Qwen und Whisper
  werden beim App-Start im Kindprozess geladen. Ein Abbruch beendet den
  Inferenzprozess, bereinigt auftragsbezogene Arbeits-/unvollstaendige
  Profilordner und startet den Worker samt Modellen neu. Ein gezielter
  Prozess-Test bestaetigte Kill, PID-Wechsel und Folgeauftrag (**9 gezielte
  Worker-/API-/App-Tests bestanden**).
- Echter GPU-Auftrag mit Qwen 0.6B und Whisper base auf synthetischer Sprache:
  den laufenden API-Auftrag abgebrochen, den Qwen-/Whisper-Worker beendet und
  neu geladen und anschliessend im Ersatzprozess erfolgreich ein zweites
  Profil samt Beispielaudio erstellt. Der alte Slug und Arbeitsdateien des
  Abbruchs blieben unveroeffentlicht bzw. wurden entfernt. GPU-Container
  endete mit Exit-Code 0.
- Text- und Mehrdatei-Hoerbuchsynthese fuer API und Gradio ergaenzt: sequenzielle
  Qwen-Chunks, 0,3-s-Satz-/Chunk- und 0,7-s-Absatzpausen, PCM-Mono-24-kHz-
  Normalisierung, EBU-R128-Ziel -20 LUFS/-2 dBTP sowie validierte WAV-/MP3-
  Ausgabe. Ausgaben werden atomar und ohne Ueberschreiben veroeffentlicht;
  Bulk-Ergebnisse, Chunkfortschritt, Abbruch und sichere Downloadroute sind an
  den gemeinsamen Jobslot gebunden.
- Gezielte Tests fuer Chunk-/Pausengrenzen, WAV/MP3, Dateiuploads, Jobstatus,
  Downloads, UI-Konfiguration, Worker und Bulk-Teilfehler bestanden: **17 Tests**.
- Echter GPU-Ende-zu-Ende-Lauf mit Whisper base und Qwen 0.6B: synthetische
  Referenzstimme erstellt, anschliessend Text als WAV (46.124 Bytes) und MP3
  (35.229 Bytes) generiert und ueber die Downloadroute geladen. Der Lauf endete
  erfolgreich; temporaere Arbeitsdateien wurden bereinigt.
- Die zuvor umfangreiche `main.py` aufgeteilt: REST-Routen liegen in `api.py`,
  Gradio-Layout und Callbacks in `ui.py`; `hoerbuchgenerator.main:app` bleibt
  als ASGI-Einstieg kompatibel. Die gezielten App-, Stimmen-API- und
  Synthese-API-Regressionstests bestanden (**11 Tests**).
- Modellkonfiguration beim Start gehaertet: Unbekannte Qwen-/Whisper-Auswahlen
  werden auch ohne vorhandene lokale Caches sofort abgewiesen. Fehlende Caches
  verhindern weiterhin den Start nicht, werden aber mit ihren erwarteten
  Pfaden protokolliert; der API-Status bleibt `not_initialized`. Die gezielten
  App- und API-Regressionstests bestanden mit **13 Tests**.
- Der App-Lifecycle erzwingt waehrend der Laufzeit Hugging-Face- und
  Transformers-Offline-Modus sowie abgeschaltete Hub-Telemetrie und
  `DO_NOT_TRACK`; vorherige Prozessumgebungswerte werden nach Shutdown
  wiederhergestellt. Gradio-Analytics bleiben in der UI deaktiviert. Ein
  Netzwerktest auf einem isolierten Runtime-Container steht weiterhin aus.
- Die lokale Cache-Erkennung an die Qwen-/Whisper-Adapter angeglichen:
  notwendige Gewichts- und Tokenizerdateien werden vor Workerstart geprueft.
  Unvollstaendige Verzeichnisse lassen den API-Dienst mit `not_initialized`
  starten und nennen die fehlenden Dateien im Log; Worker- und Adaptertests
  pruefen die Pflichtdateien.
- Der positive Offline-Initialisierungspfad ist mit vollstaendigen Fake-Caches
  getestet: konfigurierte Qwen-/Whisper-Modelle erreichen den Status `ready`,
  der Worker erhaelt die erwarteten Cachepfade und wird beim Shutdown
  geschlossen. Ein echter GPU-/Netzwerk-Abnahmelauf ist davon nicht abgedeckt.
- Die Mehrabsatz-Synthesepipeline mit zwoelf synthetischen Testabschnitten
  geprueft: Reihenfolge, monotoner Chunkfortschritt, PCM-WAV-Format,
  Absatzpausen und Bereinigung temporaerer Dateien stimmen. Die Audio-
  qualitaet der generierten Sprache und menschliche Referenzaufnahmen sind
  damit nicht bewertet.
- Den Fortschritt eines langen API-Syntheseauftrags ueber `GET /jobs/{id}`
  waehrend eines pausierten Chunks und nach Abschluss geprueft: Phase, Gesamt-
  und Abschlusszaehler sowie Bereinigung des Auftragsverzeichnisses stimmen.
  Synthese- und API-Regressionen bestanden (**10 Tests**).
- Verwaiste Eintraege unter `work/` werden beim App-Start einzeln bereinigt;
  Fehler bei der Bereinigung brechen den Start mit Pfad und Ursache ab.
  Regressionstest bestaetigt, dass persistente Stimmenprofile und fertige
  Ausgaben unberuehrt bleiben.
- Browser-Upload/-Wiedergabe und Qualitaet mit menschlichen Referenzaufnahmen
  sind noch nicht geprueft.
- Reale Nutzerreferenzen, UI-Fluss im Browser und Hoerbuchqualitaet sind noch
  nicht geprueft. Ein automatisierter Modell-Downloadschritt fehlt.

### Naechste Schritte

1. Hoerbuchqualitaet mit laengeren Texten und realen Referenzaufnahmen pruefen.
2. Offline-Betrieb und Modellinitialisierung weiter haerten.