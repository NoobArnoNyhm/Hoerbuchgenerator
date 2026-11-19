from io import BytesIO
from types import SimpleNamespace

import pytest

from hoerbuchgenerator.audio import AudioError, AudioMetadata, WordInterval
from hoerbuchgenerator.voice_reference import (
    VoiceReferenceError,
    VoiceReferenceService,
    VoiceTranscriptionError,
)
from hoerbuchgenerator.whisper import FasterWhisperAdapter, WhisperError


class FakeWhisperModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return next(self.responses)


def make_response(text, words, language="de"):
    segment = SimpleNamespace(
        text=text,
        words=[SimpleNamespace(start=word.start, end=word.end) for word in words],
    )
    info = SimpleNamespace(language=language, language_probability=0.97)
    return iter([segment]), info


def test_prepare_uploaded_voice_normalizes_and_returns_transcribed_sample(monkeypatch, tmp_path):
    transcriber = FasterWhisperAdapter(FakeWhisperModel([
        make_response("Gesamtes Transkript", [WordInterval(0, 0.5), WordInterval(0.5, 1)]),
        make_response("Guten Morgen.", [WordInterval(0, 0.5)]),
    ]))

    def fake_normalize(source, destination):
        assert source.suffix == ".mp3"
        assert source.read_bytes() == b"upload-data"
        destination.write_bytes(b"normalized")
        return AudioMetadata(destination, 1, 24_000, 1, 2)

    def fake_prepare(source, destination, model):
        assert source.name == "normalized.wav"
        assert destination.name == "reference.wav"
        assert model is transcriber
        destination.write_bytes(b"reference")
        return SimpleNamespace(
            audio=AudioMetadata(destination, 1, 24_000, 1, 2),
            transcript="Guten Morgen.",
            language="de",
            language_probability=0.97,
            source_start=0,
            source_end=1,
        )

    monkeypatch.setattr("hoerbuchgenerator.voice_reference.normalize_audio", fake_normalize)
    monkeypatch.setattr("hoerbuchgenerator.voice_reference.prepare_reference_sample", fake_prepare)
    service = VoiceReferenceService(transcriber)

    result = service.prepare(BytesIO(b"upload-data"), "voice.mp3", "Änne Öztürk", tmp_path / "work")

    assert result.name == "Änne Öztürk"
    assert result.slug == "aenne_oeztuerk"
    assert result.transcript == "Guten Morgen."
    assert result.language == "de"
    assert result.language_probability == 0.97
    assert result.source_start == 0
    assert result.source_end == 1
    assert result.audio.path.read_bytes() == b"reference"
    assert result.work_dir.is_dir()


def test_transcribe_audio_normalizes_and_transcribes_the_full_file(monkeypatch, tmp_path):
    model = FakeWhisperModel([make_response("Der gesamte erkannte Text.", [])])
    transcriber = FasterWhisperAdapter(model)
    source = tmp_path / "recording.MP3"
    source.write_bytes(b"uploaded-audio")
    work_dir = tmp_path / "work"

    def fake_normalize(source_path, destination, **kwargs):
        assert source_path == source
        assert source_path.read_bytes() == b"uploaded-audio"
        assert kwargs["max_upload_bytes"] > len(b"uploaded-audio")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"normalized-full-recording")
        return AudioMetadata(destination, 60, 24_000, 1, 2)

    monkeypatch.setattr(
        "hoerbuchgenerator.voice_reference.normalize_audio",
        fake_normalize,
    )
    service = VoiceReferenceService(transcriber)

    result = service.transcribe_audio(source, work_dir)

    assert result.text == "Der gesamte erkannte Text."
    assert result.language == "de"
    assert len(model.calls) == 1
    assert model.calls[0][0] == str(work_dir / "normalized.wav")
    assert model.calls[0][1]["task"] == "transcribe"
    assert (work_dir / "normalized.wav").read_bytes() == b"normalized-full-recording"


def test_transcribe_audio_rejects_invalid_audio_content(monkeypatch, tmp_path):
    source = tmp_path / "not-really-audio.wav"
    source.write_bytes(b"invalid audio")

    def reject_audio(*args, **kwargs):
        raise AudioError("Die Datei enthaelt keinen lesbaren Audiostream")

    monkeypatch.setattr(
        "hoerbuchgenerator.voice_reference.normalize_audio",
        reject_audio,
    )
    service = VoiceReferenceService(
        FasterWhisperAdapter(FakeWhisperModel([]))
    )

    with pytest.raises(VoiceTranscriptionError, match="lesbaren Audiostream"):
        service.transcribe_audio(source, tmp_path / "work")


@pytest.mark.parametrize(
    ("filename", "payload", "max_upload_bytes", "message"),
    [
        ("voice.txt", b"audio", 10, "WAV- und MP3"),
        ("voice.wav", b"", 10, "leer"),
        ("voice.wav", b"too large", 4, "Uploadlimit"),
    ],
)
def test_prepare_rejects_invalid_uploads(filename, payload, max_upload_bytes, message, tmp_path):
    service = VoiceReferenceService(
        FasterWhisperAdapter(FakeWhisperModel([])),
        max_upload_bytes=max_upload_bytes,
    )

    with pytest.raises(VoiceReferenceError, match=message):
        service.prepare(BytesIO(payload), filename, "Stimme", tmp_path / "work")

    work_root = tmp_path / "work"
    assert not work_root.exists() or list(work_root.iterdir()) == []


def test_prepare_removes_partial_work_after_transcription_failure(monkeypatch, tmp_path):
    def fail_transcription(*args, **kwargs):
        raise WhisperError("kein verwertbarer Sprachabschnitt")

    monkeypatch.setattr(
        "hoerbuchgenerator.voice_reference.normalize_audio",
        lambda source, destination: AudioMetadata(destination, 1, 24_000, 1, 2),
    )
    monkeypatch.setattr("hoerbuchgenerator.voice_reference.prepare_reference_sample", fail_transcription)
    service = VoiceReferenceService(FasterWhisperAdapter(FakeWhisperModel([])))

    with pytest.raises(VoiceReferenceError, match="kein verwertbarer Sprachabschnitt"):
        service.prepare(BytesIO(b"audio"), "voice.wav", "Stimme", tmp_path / "work")

    assert list((tmp_path / "work").iterdir()) == []
