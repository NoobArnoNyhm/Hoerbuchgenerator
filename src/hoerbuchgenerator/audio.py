from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import wave
from uuid import uuid4


TARGET_SAMPLE_RATE = 24_000
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_AUDIO_SECONDS = 4 * 60 * 60


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class WordInterval:
    start: float
    end: float


@dataclass(frozen=True)
class AudioMetadata:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    sample_width: int


def select_reference_window(
    words: list[WordInterval],
    min_seconds: float = 20,
    target_seconds: float = 30,
    max_seconds: float = 40,
) -> tuple[float, float]:
    if not 0 < min_seconds <= target_seconds <= max_seconds:
        raise ValueError("Ungueltige Dauergrenzen fuer das Referenzsample")
    if not words:
        raise AudioError("Whisper hat keine Wortzeitstempel geliefert")

    previous_start = -1.0
    previous_end = -1.0
    for word in words:
        if (
            not math.isfinite(word.start)
            or not math.isfinite(word.end)
            or word.start < 0
            or word.end <= word.start
            or word.start < previous_start
            or word.end < previous_end
        ):
            raise AudioError("Whisper-Wortzeitstempel sind ungueltig oder unsortiert")
        previous_start, previous_end = word.start, word.end

    ends = [word.end for word in words]
    best_window: tuple[float, float] | None = None
    best_score: tuple[float, float] | None = None
    shortest_fallback: tuple[float, float] | None = None
    longest_fallback_duration = -1.0

    for start_index, word in enumerate(words):
        start = word.start
        last_index = bisect_right(ends, start + max_seconds, lo=start_index)
        if last_index == start_index:
            continue
        last_index -= 1

        minimum_index = bisect_left(
            ends,
            start + min_seconds,
            lo=start_index,
            hi=last_index + 1,
        )
        if minimum_index <= last_index:
            target_index = bisect_left(
                ends,
                start + target_seconds,
                lo=minimum_index,
                hi=last_index + 1,
            )
            for end_index in {minimum_index, last_index, target_index - 1, target_index}:
                if minimum_index <= end_index <= last_index:
                    end = ends[end_index]
                    score = (abs(end - start - target_seconds), -(end - start))
                    if best_score is None or score < best_score:
                        best_window = (start, end)
                        best_score = score

        fallback_end = ends[last_index]
        fallback_duration = fallback_end - start
        if fallback_duration > longest_fallback_duration:
            shortest_fallback = (start, fallback_end)
            longest_fallback_duration = fallback_duration

    if best_window is not None:
        return best_window
    if shortest_fallback is not None:
        return shortest_fallback
    raise AudioError("Kein vollstaendiges Wort passt in das Referenzsample")


def _run(command: list[str], purpose: str, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise AudioError(f"{purpose} nicht verfuegbar; FFmpeg/FFprobe fehlt") from error
    except subprocess.TimeoutExpired as error:
        raise AudioError(f"Zeitlimit bei {purpose} ueberschritten") from error

    if result.returncode != 0:
        raise AudioError(f"{purpose} fehlgeschlagen (Exit-Code {result.returncode})")
    return result


def _probe_audio(path: Path, ffprobe: str, timeout: int) -> float:
    result = _run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(path),
        ],
        "Audio-Pruefung",
        timeout,
    )
    try:
        probe = json.loads(result.stdout)
        has_audio = any(stream.get("codec_type") == "audio" for stream in probe["streams"])
        duration = float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioError("Die Datei enthaelt keinen lesbaren Audiostream") from error

    if not has_audio or not math.isfinite(duration) or duration <= 0:
        raise AudioError("Die Datei enthaelt keinen lesbaren Audiostream")
    if duration > MAX_AUDIO_SECONDS:
        raise AudioError("Die Audiodauer ueberschreitet das Limit von vier Stunden")
    return duration


def normalize_audio(
    source: Path,
    destination: Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    timeout: int = 120,
) -> AudioMetadata:
    source = Path(source)
    destination = Path(destination)
    if source.suffix.lower() not in {".wav", ".mp3"}:
        raise AudioError("Erlaubt sind WAV- und MP3-Dateien")
    try:
        source_size = source.stat().st_size
    except OSError as error:
        raise AudioError("Die Audiodatei ist nicht lesbar") from error
    if source_size == 0:
        raise AudioError("Die Audiodatei ist leer")
    if source_size > max_upload_bytes:
        raise AudioError("Die Audiodatei ueberschreitet das Uploadlimit")

    _probe_audio(source, ffprobe, timeout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.wav")
    try:
        _run(
            [
                ffmpeg,
                "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source),
                "-map", "0:a:0", "-vn",
                "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                str(temporary),
            ],
            "Audio-Decodierung",
            timeout,
        )
        with wave.open(str(temporary), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
        if channels != 1 or sample_rate != TARGET_SAMPLE_RATE or sample_width != 2 or frames < 1:
            raise AudioError("FFmpeg-Ausgabe hat kein gueltiges PCM-Mono-WAV-Format")

        temporary.replace(destination)
        return AudioMetadata(
            path=destination,
            duration=frames / sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
    except (OSError, wave.Error) as error:
        raise AudioError("Audio konnte nicht als PCM-WAV validiert werden") from error
    finally:
        temporary.unlink(missing_ok=True)


def extract_audio_segment(
    source: Path,
    destination: Path,
    start: float,
    end: float,
    *,
    ffmpeg: str = "ffmpeg",
    timeout: int = 120,
) -> AudioMetadata:
    source = Path(source)
    destination = Path(destination)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise AudioError("Ungueltige Zeitgrenzen fuer den Audioausschnitt")
    if end - start > MAX_AUDIO_SECONDS:
        raise AudioError("Der Audioausschnitt ueberschreitet das Zeitlimit")

    try:
        with wave.open(str(source), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
    except (OSError, wave.Error) as error:
        raise AudioError("Die Quelldatei ist kein lesbares WAV") from error

    duration = frames / sample_rate if sample_rate else 0
    if channels != 1 or sample_rate != TARGET_SAMPLE_RATE or sample_width != 2 or frames < 1:
        raise AudioError("Die Quelle muss ein normalisiertes PCM-Mono-WAV sein")
    if end > duration:
        raise AudioError("Der Audioausschnitt liegt ausserhalb der Quelldatei")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.wav")
    try:
        _run(
            [
                ffmpeg,
                "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.6f}", "-i", str(source),
                "-t", f"{end - start:.6f}",
                "-map", "0:a:0", "-vn",
                "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                str(temporary),
            ],
            "Audioausschnitt",
            timeout,
        )
        with wave.open(str(temporary), "rb") as wav_file:
            result_channels = wav_file.getnchannels()
            result_rate = wav_file.getframerate()
            result_width = wav_file.getsampwidth()
            result_frames = wav_file.getnframes()
        actual_duration = result_frames / result_rate if result_rate else 0
        if (
            result_channels != 1
            or result_rate != TARGET_SAMPLE_RATE
            or result_width != 2
            or result_frames < 1
            or abs(actual_duration - (end - start)) > 0.02
        ):
            raise AudioError("Der erzeugte Audioausschnitt ist unvollstaendig oder ungueltig")

        temporary.replace(destination)
        return AudioMetadata(
            path=destination,
            duration=actual_duration,
            sample_rate=result_rate,
            channels=result_channels,
            sample_width=result_width,
        )
    except (OSError, wave.Error) as error:
        raise AudioError("Audioausschnitt konnte nicht als PCM-WAV validiert werden") from error
    finally:
        temporary.unlink(missing_ok=True)