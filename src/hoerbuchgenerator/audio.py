"""Compatible public imports for audio processing and reference selection."""
from hoerbuchgenerator.audio_ffmpeg import extract_audio_segment, normalize_audio
from hoerbuchgenerator.audio_reference import select_reference_window
from hoerbuchgenerator.audio_types import (
    AudioError,
    AudioMetadata,
    MAX_AUDIO_SECONDS,
    MAX_UPLOAD_BYTES,
    TARGET_SAMPLE_RATE,
    WordInterval,
)

__all__ = [
    "AudioError",
    "AudioMetadata",
    "MAX_AUDIO_SECONDS",
    "MAX_UPLOAD_BYTES",
    "TARGET_SAMPLE_RATE",
    "WordInterval",
    "extract_audio_segment",
    "normalize_audio",
    "select_reference_window",
]
