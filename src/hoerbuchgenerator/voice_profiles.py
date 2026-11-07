from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Event
from typing import BinaryIO
import wave

import numpy as np

from hoerbuchgenerator.audio import AudioError, normalize_audio
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QwenError, QwenTTSAdapter
from hoerbuchgenerator.voice_reference import (
    VoiceReferenceError,
    VoiceReferenceService,
)


EXAMPLE_TEXT = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam"
QWEN_LANGUAGES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


class VoiceProfileError(RuntimeError):
    pass


class VoiceCreationCancelled(VoiceProfileError):
    pass


@dataclass(frozen=True)
class CreatedVoice:
    name: str
    slug: str
    language: str
    path: Path


class VoiceProfileService:
    def __init__(
        self,
        references: VoiceReferenceService,
        qwen: QwenTTSAdapter,
        qwen_model_name: str,
    ):
        if qwen_model_name not in QWEN_BASE_MODELS:
            raise ValueError(f"Nicht unterstuetztes Qwen-Base-Modell: {qwen_model_name}")
        self._references = references
        self._qwen = qwen
        self._qwen_model_name = qwen_model_name

    def synthesize(
        self,
        text: str,
        language: str,
        prompt_path: Path,
    ) -> tuple[np.ndarray, int]:
        try:
            prompt = self._qwen.load_voice_prompt(prompt_path)
            return self._qwen.synthesize(text, language, prompt)
        except QwenError as error:
            raise VoiceProfileError(str(error)) from error

    def create(
        self,
        upload: BinaryIO,
        filename: str,
        name: str,
        voices_dir: Path,
        work_dir: Path,
        *,
        cancel_event: Event | None = None,
    ) -> CreatedVoice:
        voices_dir = Path(voices_dir)
        try:
            prepared_name = name.strip() if isinstance(name, str) else name
            slug = make_slug(prepared_name)
        except (AttributeError, TypeError, ValueError) as error:
            raise VoiceProfileError("Ungueltiger Stimmenname") from error

        destination = voices_dir / slug
        if destination.exists():
            raise VoiceProfileError("Eine Stimme mit diesem Namen existiert bereits")

        try:
            voices_dir.mkdir(parents=True, exist_ok=True)
            temporary_profile = Path(
                tempfile.mkdtemp(prefix=f".{slug}-", dir=voices_dir)
            )
        except OSError as error:
            raise VoiceProfileError("Stimmenverzeichnis konnte nicht vorbereitet werden") from error

        prepared = None
        published = False
        try:
            prepared = self._references.prepare(upload, filename, name, work_dir)
            self._check_cancelled(cancel_event)
            language = QWEN_LANGUAGES.get(prepared.language.lower())
            if language is None:
                raise VoiceProfileError(
                    f"Qwen unterstuetzt die erkannte Sprache '{prepared.language}' nicht"
                )

            shutil.copyfile(prepared.audio.path, temporary_profile / "reference.wav")
            (temporary_profile / "transcript.txt").write_text(
                prepared.transcript + "\n",
                encoding="utf-8",
            )
            prompt = self._qwen.create_voice_prompt(
                prepared.audio.path,
                prepared.transcript,
            )
            prompt_path = temporary_profile / "voice-prompt.npz"
            self._qwen.save_voice_prompt(prompt_path, prompt)
            restored_prompt = self._qwen.load_voice_prompt(prompt_path)

            self._check_cancelled(cancel_event)
            audio, sample_rate = self._qwen.synthesize(
                EXAMPLE_TEXT,
                language,
                restored_prompt,
            )
            generated_audio = temporary_profile / "generated.wav"
            self._write_audio(generated_audio, audio, sample_rate)
            normalize_audio(generated_audio, temporary_profile / "example.wav")
            generated_audio.unlink()

            self._check_cancelled(cancel_event)
            metadata = {
                "schema_version": 1,
                "name": prepared.name,
                "slug": prepared.slug,
                "language": prepared.language,
                "language_probability": prepared.language_probability,
                "transcript": prepared.transcript,
                "qwen_model": self._qwen_model_name,
                "example_text": EXAMPLE_TEXT,
                "reference_source_start": prepared.source_start,
                "reference_source_end": prepared.source_end,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path = temporary_profile / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with metadata_path.open("rb") as metadata_file:
                os.fsync(metadata_file.fileno())

            self._check_cancelled(cancel_event)
            if destination.exists():
                raise VoiceProfileError("Eine Stimme mit diesem Namen existiert bereits")
            os.rename(temporary_profile, destination)
            published = True
            return CreatedVoice(
                name=prepared.name,
                slug=prepared.slug,
                language=prepared.language,
                path=destination,
            )
        except VoiceCreationCancelled:
            raise
        except VoiceProfileError:
            raise
        except (AudioError, QwenError, VoiceReferenceError) as error:
            raise VoiceProfileError(str(error)) from error
        except OSError as error:
            raise VoiceProfileError("Stimmenprofil konnte nicht gespeichert werden") from error
        finally:
            if prepared is not None:
                shutil.rmtree(prepared.work_dir, ignore_errors=True)
            if not published:
                shutil.rmtree(temporary_profile, ignore_errors=True)

    @staticmethod
    def _check_cancelled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise VoiceCreationCancelled("Stimmerstellung wurde abgebrochen")

    @staticmethod
    def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise VoiceProfileError("Qwen lieferte eine ungueltige Samplerate")
        samples = np.asarray(audio)
        if (
            samples.ndim != 1
            or samples.size == 0
            or not np.issubdtype(samples.dtype, np.number)
            or not np.isfinite(samples).all()
        ):
            raise VoiceProfileError("Qwen lieferte ungueltige Audiodaten")

        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
