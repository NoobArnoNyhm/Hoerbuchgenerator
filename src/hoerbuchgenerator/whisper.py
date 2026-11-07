from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol

from hoerbuchgenerator.audio import (
    AudioError,
    AudioMetadata,
    WordInterval,
    extract_audio_segment,
    select_reference_window,
)
from hoerbuchgenerator.model_cache import incomplete_model_files


WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
WHISPER_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")


class WhisperError(RuntimeError):
    pass


class WhisperModelProtocol(Protocol):
    def transcribe(self, audio: str, **kwargs): ...


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    language_probability: float
    words: tuple[WordInterval, ...]


@dataclass(frozen=True)
class ReferenceSample:
    audio: AudioMetadata
    transcript: str
    language: str
    language_probability: float
    source_start: float
    source_end: float


class FasterWhisperAdapter:
    def __init__(self, model: WhisperModelProtocol):
        self._model = model

    @classmethod
    def load_local(
        cls,
        model_name: str,
        models_dir: Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> "FasterWhisperAdapter":
        if model_name not in WHISPER_MODELS:
            raise WhisperError(f"Nicht unterstuetztes Whisper-Modell: {model_name}")

        model_path = Path(models_dir) / model_name
        incomplete = incomplete_model_files(model_path, WHISPER_REQUIRED_FILES)
        if incomplete:
            raise WhisperError(
                f"Whisper-Modell '{model_name}' fehlt oder ist unvollstaendig: "
                f"{', '.join(incomplete)} in {model_path}"
            )

        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise WhisperError(
                "faster-whisper fehlt; Installation mit pip install -e '.[whisper]'"
            ) from error

        try:
            model = WhisperModel(
                str(model_path),
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
        except Exception as error:
            raise WhisperError(f"Whisper-Modell '{model_name}' konnte nicht geladen werden") from error
        return cls(model)

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> Transcription:
        try:
            segments, info = self._model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                word_timestamps=True,
                vad_filter=True,
                log_progress=False,
            )
            segments = list(segments)
        except Exception as error:
            raise WhisperError("Whisper-Transkription fehlgeschlagen") from error

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        detected_language = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)
        if not text:
            raise WhisperError("Whisper hat keine Sprache erkannt oder ein leeres Transkript geliefert")
        if not isinstance(detected_language, str) or not detected_language.strip():
            raise WhisperError("Whisper hat keinen Sprachcode geliefert")
        if not isinstance(probability, (int, float)) or not math.isfinite(probability):
            raise WhisperError("Whisper hat keine gueltige Sprachwahrscheinlichkeit geliefert")

        words: list[WordInterval] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or ():
                try:
                    start, end = float(word.start), float(word.end)
                except (AttributeError, TypeError, ValueError) as error:
                    raise WhisperError("Whisper hat ungueltige Wortzeitstempel geliefert") from error
                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end < start
                ):
                    raise WhisperError("Whisper hat ungueltige Wortzeitstempel geliefert")
                if end > start:
                    words.append(WordInterval(start, end))

        return Transcription(
            text=text,
            language=detected_language.strip(),
            language_probability=float(probability),
            words=tuple(words),
        )


def prepare_reference_sample(
    audio_path: Path,
    sample_path: Path,
    transcriber: FasterWhisperAdapter,
    *,
    min_seconds: float = 20,
    target_seconds: float = 30,
    max_seconds: float = 40,
) -> ReferenceSample:
    full_transcription = transcriber.transcribe(audio_path)
    try:
        start, end = select_reference_window(
            list(full_transcription.words),
            min_seconds=min_seconds,
            target_seconds=target_seconds,
            max_seconds=max_seconds,
        )
    except AudioError as error:
        raise WhisperError(str(error)) from error

    try:
        sample_audio = extract_audio_segment(audio_path, sample_path, start, end)
    except AudioError as error:
        raise WhisperError(f"Whisper-Sample konnte nicht ausgeschnitten werden: {error}") from error

    sample_transcription = transcriber.transcribe(sample_path)
    return ReferenceSample(
        audio=sample_audio,
        transcript=sample_transcription.text,
        language=sample_transcription.language,
        language_probability=sample_transcription.language_probability,
        source_start=start,
        source_end=end,
    )