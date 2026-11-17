from dataclasses import dataclass
from pathlib import Path


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
