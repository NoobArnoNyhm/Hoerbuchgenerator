import json
import os
from pathlib import Path
import re
import subprocess
import wave
from typing import Callable
from uuid import uuid4

import numpy as np

from hoerbuchgenerator.audio import AudioError, TARGET_SAMPLE_RATE, normalize_audio
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.text import chunk_text
from hoerbuchgenerator.voice_profiles import QWEN_LANGUAGES


PAUSE_BETWEEN_CHUNKS_SECONDS = 0.3
PAUSE_BETWEEN_PARAGRAPHS_SECONDS = 0.7
MAX_TEXT_BYTES = 10 * 1024 * 1024


class SynthesisError(RuntimeError):
    pass


def _chunks_with_pauses(text: str, max_chars: int = 600) -> list[tuple[str, float]]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        raise SynthesisError("Der Text darf nicht leer sein")

    tagged_chunks: list[tuple[str, int]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            tagged_chunks.extend(
                (chunk, paragraph_index)
                for chunk in chunk_text(sentence, max_chars=max_chars)
            )

    index = 0
    while index < len(tagged_chunks) - 1:
        current, current_paragraph = tagged_chunks[index]
        following, following_paragraph = tagged_chunks[index + 1]
        if (
            len(current) < 5
            and len(current) + 1 + len(following) <= max_chars
        ):
            tagged_chunks[index:index + 2] = [
                (f"{current} {following}", following_paragraph)
            ]
        else:
            index += 1

    result: list[tuple[str, float]] = []
    for index, (chunk, paragraph_index) in enumerate(tagged_chunks):
        if index == len(tagged_chunks) - 1:
            pause = 0.0
        elif tagged_chunks[index + 1][1] != paragraph_index:
            pause = PAUSE_BETWEEN_PARAGRAPHS_SECONDS
        else:
            pause = PAUSE_BETWEEN_CHUNKS_SECONDS
        result.append((chunk, pause))
    return result


def _write_pcm(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    if (
        not isinstance(sample_rate, int)
        or sample_rate < 1
        or samples.ndim != 1
        or samples.size == 0
        or not np.issubdtype(samples.dtype, np.number)
        or not np.isfinite(samples).all()
    ):
        raise SynthesisError("Qwen lieferte ungueltige Audiodaten")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.tobytes())


def _convert_mp3(source: Path, destination: Path) -> None:
    try:
        converted = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-map", "0:a:0", "-vn",
                "-codec:a", "libmp3lame", "-q:a", "3", str(destination),
            ],
            check=False,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SynthesisError("MP3-Konvertierung fehlgeschlagen") from error
    if converted.returncode != 0:
        raise SynthesisError("MP3-Konvertierung fehlgeschlagen")


def _normalize_loudness(path: Path) -> None:
    normalized = path.with_name(f".{path.stem}.{uuid4().hex}.loudness.wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path),
                "-af", "loudnorm=I=-20:TP=-2:LRA=11",
                "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
                "-codec:a", "pcm_s16le", str(normalized),
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise SynthesisError("Lautheitsnormalisierung eines Audio-Chunks fehlgeschlagen")
        with wave.open(str(normalized), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != TARGET_SAMPLE_RATE
                or audio.getnframes() == 0
            ):
                raise SynthesisError("Lautheitsnormalisierung lieferte ein ungueltiges WAV")
        normalized.replace(path)
    except (OSError, wave.Error, subprocess.TimeoutExpired) as error:
        raise SynthesisError("Lautheitsnormalisierung eines Audio-Chunks fehlgeschlagen") from error
    finally:
        normalized.unlink(missing_ok=True)


def synthesize_file(
    qwen,
    voice_dir: Path,
    text: str,
    output_name: str,
    output_dir: Path,
    output_format: str,
    work_dir: Path,
    *,
    language: str | None = None,
    cancel_check: Callable[[], bool] = lambda: False,
    progress: Callable[[int], None] = lambda completed: None,
    max_chars: int = 600,
) -> dict[str, str | int]:
    if output_format not in {"wav", "mp3"}:
        raise SynthesisError("Ausgabeformat muss WAV oder MP3 sein")
    try:
        output_slug = make_slug(output_name)
    except ValueError as error:
        raise SynthesisError(str(error)) from error

    voice_dir = Path(voice_dir)
    output_dir = Path(output_dir)
    work_dir = Path(work_dir)
    metadata_path = voice_dir / "metadata.json"
    prompt_path = voice_dir / "voice-prompt.npz"
    if not voice_dir.is_dir() or not metadata_path.is_file() or not prompt_path.is_file():
        raise SynthesisError("Das Stimmenprofil ist unvollstaendig")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        profile_language = metadata["language"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SynthesisError("Stimmenprofil oder Sprachmetadaten sind unlesbar") from error
    if not isinstance(profile_language, str):
        raise SynthesisError("Stimmenprofil enthaelt keine gueltige Sprache")
    try:
        language_name = QWEN_LANGUAGES[(language or profile_language).lower()]
    except (AttributeError, KeyError) as error:
        raise SynthesisError("Die Sprache wird von Qwen nicht unterstuetzt") from error
    if not prompt_path.is_file():
        raise SynthesisError("Stimmenprofil enthaelt kein Qwen-Prompt")

    chunks = _chunks_with_pauses(text, max_chars)
    extension = "wav" if output_format == "wav" else "mp3"
    output_path = output_dir / f"{output_slug}.{extension}"
    if output_path.exists():
        raise SynthesisError("Eine Ausgabedatei mit diesem Namen existiert bereits")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output_dir / f".{output_slug}.{uuid4().hex}.tmp.wav"
    temporary_mp3 = output_dir / f".{output_slug}.{uuid4().hex}.tmp.mp3"
    chunk_paths: list[Path] = []
    try:
        with wave.open(str(temporary_output), "wb") as combined:
            combined.setnchannels(1)
            combined.setsampwidth(2)
            combined.setframerate(TARGET_SAMPLE_RATE)
            for index, (chunk, pause) in enumerate(chunks, start=1):
                if cancel_check():
                    raise SynthesisError("Synthese wurde abgebrochen")
                chunk_path = work_dir / f"chunk-{index:05d}.wav"
                qwen.synthesize_chunk(
                    voice_dir=voice_dir,
                    text=chunk,
                    language=language_name,
                    destination=chunk_path,
                )
                normalized = work_dir / f"normalized-{index:05d}.wav"
                try:
                    normalize_audio(chunk_path, normalized)
                    _normalize_loudness(normalized)
                except AudioError as error:
                    raise SynthesisError(f"Audio-Chunk {index} ist ungueltig: {error}") from error
                chunk_paths.extend((chunk_path, normalized))
                try:
                    with wave.open(str(normalized), "rb") as source:
                        if (
                            source.getnchannels() != 1
                            or source.getsampwidth() != 2
                            or source.getframerate() != TARGET_SAMPLE_RATE
                        ):
                            raise SynthesisError("Normalisierter Audio-Chunk hat ein ungueltiges Format")
                        combined.writeframes(source.readframes(source.getnframes()))
                except wave.Error as error:
                    raise SynthesisError(f"Audio-Chunk {index} ist unlesbar") from error
                if pause:
                    combined.writeframes(
                        bytes(round(TARGET_SAMPLE_RATE * pause) * 2)
                    )
                progress(index)

        with wave.open(str(temporary_output), "rb") as result:
            if result.getnframes() == 0 or result.getframerate() != TARGET_SAMPLE_RATE:
                raise SynthesisError("Zusammengefuehrte WAV-Datei ist ungueltig")

        if output_format == "mp3":
            _convert_mp3(temporary_output, temporary_mp3)
            try:
                normalize_audio(temporary_mp3, work_dir / "verify-mp3.wav")
            except AudioError as error:
                raise SynthesisError("Erzeugte MP3-Datei ist nicht dekodierbar") from error
            os.link(temporary_mp3, output_path)
        else:
            os.link(temporary_output, output_path)

        return {
            "name": output_path.name,
            "format": output_format,
            "chunks": len(chunks),
        }
    except FileExistsError as error:
        raise SynthesisError("Eine Ausgabedatei mit diesem Namen existiert bereits") from error
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_mp3.unlink(missing_ok=True)
        for chunk_path in chunk_paths:
            chunk_path.unlink(missing_ok=True)
        (work_dir / "verify-mp3.wav").unlink(missing_ok=True)
