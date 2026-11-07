import json
from io import BytesIO
import wave
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from hoerbuchgenerator.audio import AudioMetadata
from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QwenError
from hoerbuchgenerator.voice_profiles import (
    EXAMPLE_TEXT,
    VoiceCreationCancelled,
    VoiceProfileError,
    VoiceProfileService,
)


class FakeQwen:
    def __init__(self, *, cancel_event=None, fail_synthesis=False):
        self.cancel_event = cancel_event
        self.fail_synthesis = fail_synthesis
        self.prompt_calls = []
        self.synthesis_calls = []

    def create_voice_prompt(self, audio, transcript):
        self.prompt_calls.append((audio, transcript))
        return "prompt"

    def save_voice_prompt(self, path, prompt):
        path.write_text(prompt)

    def load_voice_prompt(self, path):
        return path.read_text()

    def synthesize(self, text, language, prompt):
        self.synthesis_calls.append((text, language, prompt))
        if self.cancel_event is not None:
            self.cancel_event.set()
        if self.fail_synthesis:
            raise QwenError("synthesis failed")
        return np.array([0.0, 0.2, -0.2, 0.1], dtype=np.float32), 24_000


class FakeReferences:
    def prepare(self, upload, filename, name, work_root):
        work_dir = Path(work_root) / "prepared"
        work_dir.mkdir(parents=True)
        reference = work_dir / "reference.wav"
        reference.write_bytes(b"reference")
        return SimpleNamespace(
            name=name.strip(),
            slug="aenne_oeztuerk",
            audio=AudioMetadata(reference, 1, 24_000, 1, 2),
            transcript="Guten Morgen.",
            language="de",
            language_probability=0.97,
            source_start=1.5,
            source_end=2.5,
            work_dir=work_dir,
        )


def create_service(qwen):
    return VoiceProfileService(
        FakeReferences(),
        qwen,
        QWEN_BASE_MODELS[0],
    )


def test_voice_profile_publishes_all_validated_files_atomically(tmp_path):
    qwen = FakeQwen()
    service = create_service(qwen)
    voices_dir = tmp_path / "voices"
    work_dir = tmp_path / "work"

    result = service.create(
        BytesIO(b"upload"),
        "voice.wav",
        "Änne Öztürk",
        voices_dir,
        work_dir,
    )

    assert result.slug == "aenne_oeztuerk"
    assert result.path == voices_dir / result.slug
    assert sorted(path.name for path in result.path.iterdir()) == [
        "example.wav",
        "metadata.json",
        "reference.wav",
        "transcript.txt",
        "voice-prompt.npz",
    ]
    assert (result.path / "reference.wav").read_bytes() == b"reference"
    assert (result.path / "transcript.txt").read_text(encoding="utf-8") == "Guten Morgen.\n"
    metadata = json.loads((result.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "Änne Öztürk"
    assert metadata["language"] == "de"
    assert metadata["qwen_model"] == QWEN_BASE_MODELS[0]
    assert metadata["reference_source_start"] == 1.5
    assert qwen.prompt_calls[0][1] == "Guten Morgen."
    assert qwen.synthesis_calls == [(EXAMPLE_TEXT, "German", "prompt")]
    with wave.open(str(result.path / "example.wav"), "rb") as example:
        assert example.getnchannels() == 1
        assert example.getframerate() == 24_000
    assert not list(work_dir.iterdir())
    assert not list(voices_dir.glob(".aenne_oeztuerk-*"))


def test_profile_is_not_published_if_example_synthesis_fails(tmp_path):
    service = create_service(FakeQwen(fail_synthesis=True))
    voices_dir = tmp_path / "voices"
    work_dir = tmp_path / "work"

    with pytest.raises(VoiceProfileError, match="synthesis failed"):
        service.create(
            BytesIO(b"upload"),
            "voice.wav",
            "Änne Öztürk",
            voices_dir,
            work_dir,
        )

    assert not (voices_dir / "aenne_oeztuerk").exists()
    assert not list(voices_dir.iterdir())
    assert not list(work_dir.iterdir())


def test_cancel_before_publication_removes_staging_and_keeps_profile_invisible(tmp_path):
    cancel_event = Event()
    service = create_service(FakeQwen(cancel_event=cancel_event))
    voices_dir = tmp_path / "voices"
    work_dir = tmp_path / "work"

    with pytest.raises(VoiceCreationCancelled):
        service.create(
            BytesIO(b"upload"),
            "voice.wav",
            "Änne Öztürk",
            voices_dir,
            work_dir,
            cancel_event=cancel_event,
        )

    assert not (voices_dir / "aenne_oeztuerk").exists()
    assert not list(voices_dir.iterdir())
    assert not list(work_dir.iterdir())


def test_duplicate_voice_name_does_not_run_preparation(tmp_path):
    voices_dir = tmp_path / "voices"
    existing_voice = voices_dir / "aenne_oeztuerk"
    existing_voice.mkdir(parents=True)
    service = create_service(FakeQwen())

    with pytest.raises(VoiceProfileError, match="existiert bereits"):
        service.create(
            BytesIO(b"upload"),
            "voice.wav",
            "Änne Öztürk",
            voices_dir,
            tmp_path / "work",
        )

    assert list(voices_dir.iterdir()) == [existing_voice]
    assert not (tmp_path / "work").exists()
