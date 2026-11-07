from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO

from hoerbuchgenerator.audio import (
    AudioError,
    AudioMetadata,
    MAX_UPLOAD_BYTES,
    normalize_audio,
)
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.whisper import (
    FasterWhisperAdapter,
    WhisperError,
    prepare_reference_sample,
)


class VoiceReferenceError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedVoiceReference:
    name: str
    slug: str
    audio: AudioMetadata
    transcript: str
    language: str
    language_probability: float
    source_start: float
    source_end: float
    work_dir: Path


class VoiceReferenceService:
    def __init__(
        self,
        transcriber: FasterWhisperAdapter,
        *,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ):
        if max_upload_bytes < 1:
            raise ValueError("max_upload_bytes muss mindestens 1 sein")
        self._transcriber = transcriber
        self._max_upload_bytes = max_upload_bytes

    def prepare(
        self,
        upload: BinaryIO,
        filename: str,
        name: str,
        work_root: Path,
    ) -> PreparedVoiceReference:
        if not isinstance(name, str) or not name.strip():
            raise VoiceReferenceError("Der Stimmenname darf nicht leer sein")
        try:
            slug = make_slug(name)
        except ValueError as error:
            raise VoiceReferenceError(str(error)) from error

        if not isinstance(filename, str) or not filename:
            raise VoiceReferenceError("Der Dateiname des Stimmen-Uploads fehlt")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".mp3"}:
            raise VoiceReferenceError("Erlaubt sind WAV- und MP3-Dateien")

        work_root = Path(work_root)
        try:
            work_root.mkdir(parents=True, exist_ok=True)
            work_dir = Path(tempfile.mkdtemp(prefix="voice-", dir=work_root))
        except OSError as error:
            raise VoiceReferenceError(
                "Arbeitsverzeichnis fuer den Stimmen-Upload konnte nicht erstellt werden"
            ) from error

        completed = False
        try:
            uploaded_audio = work_dir / f"upload{suffix}"
            self._save_upload(upload, uploaded_audio)
            normalized = normalize_audio(uploaded_audio, work_dir / "normalized.wav")
            sample = prepare_reference_sample(
                normalized.path,
                work_dir / "reference.wav",
                self._transcriber,
            )
            result = PreparedVoiceReference(
                name=name.strip(),
                slug=slug,
                audio=sample.audio,
                transcript=sample.transcript,
                language=sample.language,
                language_probability=sample.language_probability,
                source_start=sample.source_start,
                source_end=sample.source_end,
                work_dir=work_dir,
            )
            completed = True
            return result
        except (AudioError, WhisperError) as error:
            raise VoiceReferenceError(str(error)) from error
        except OSError as error:
            raise VoiceReferenceError("Stimmen-Upload konnte nicht verarbeitet werden") from error
        finally:
            if not completed:
                shutil.rmtree(work_dir)

    def _save_upload(self, upload: BinaryIO, destination: Path) -> None:
        size = 0
        try:
            with destination.open("wb") as target:
                while True:
                    chunk = upload.read(64 * 1024)
                    if chunk == b"":
                        break
                    if not isinstance(chunk, bytes):
                        raise VoiceReferenceError("Der Stimmen-Upload enthaelt ungueltige Daten")
                    size += len(chunk)
                    if size > self._max_upload_bytes:
                        raise VoiceReferenceError("Die Audiodatei ueberschreitet das Uploadlimit")
                    target.write(chunk)
        except OSError as error:
            raise VoiceReferenceError("Stimmen-Upload konnte nicht gespeichert werden") from error

        if size == 0:
            raise VoiceReferenceError("Die Audiodatei ist leer")
