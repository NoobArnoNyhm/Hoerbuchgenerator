from pathlib import Path
from types import SimpleNamespace

import pytest

from hoerbuchgenerator.audio import AudioError, AudioMetadata, WordInterval
from hoerbuchgenerator.whisper import (
    FasterWhisperAdapter,
    WhisperError,
    prepare_reference_sample,
)


class FakeWhisperModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return next(self.responses)


def make_segments(text, words=()):
    segment = SimpleNamespace(text=text, words=[SimpleNamespace(start=w.start, end=w.end) for w in words])
    info = SimpleNamespace(language="de", language_probability=0.98)
    return iter([segment]), info


def test_transcribe_requests_vad_word_times_and_preserves_language_metadata():
    model = FakeWhisperModel([make_segments(" Guten Tag ", [WordInterval(0, 0.4)])])
    adapter = FasterWhisperAdapter(model)

    result = adapter.transcribe(Path("sample.wav"))

    assert result.text == "Guten Tag"
    assert result.language == "de"
    assert result.language_probability == 0.98
    assert result.words == (WordInterval(0, 0.4),)
    assert model.calls[0][1]["word_timestamps"] is True
    assert model.calls[0][1]["vad_filter"] is True


def test_transcribe_skips_zero_duration_word_timestamps():
    model = FakeWhisperModel([
        make_segments("Guten Tag", [WordInterval(0, 0), WordInterval(0.1, 0.5)])
    ])
    adapter = FasterWhisperAdapter(model)

    result = adapter.transcribe(Path("sample.wav"))

    assert result.words == (WordInterval(0.1, 0.5),)


def test_prepare_reference_uses_complete_word_window_then_transcribes_sample(monkeypatch, tmp_path):
    words = tuple(WordInterval(index, index + 0.5) for index in range(0, 60))
    model = FakeWhisperModel([
        make_segments("Gesamter Text", words),
        make_segments("Der Sampletext.", [WordInterval(0, 0.5)]),
    ])
    adapter = FasterWhisperAdapter(model)
    extracted = {}

    def fake_extract(source, destination, start, end):
        extracted.update(source=source, destination=destination, start=start, end=end)
        return AudioMetadata(Path(destination), end - start, 24_000, 1, 2)

    monkeypatch.setattr("hoerbuchgenerator.whisper.extract_audio_segment", fake_extract)
    result = prepare_reference_sample(
        tmp_path / "input.wav",
        tmp_path / "reference.wav",
        adapter,
        min_seconds=20,
        target_seconds=30,
        max_seconds=40,
    )

    assert 20 <= result.source_end - result.source_start <= 40
    assert extracted["source"] == tmp_path / "input.wav"
    assert extracted["destination"] == tmp_path / "reference.wav"
    assert result.transcript == "Der Sampletext."
    assert result.language == "de"
    assert len(model.calls) == 2
    assert model.calls[1][0] == str(tmp_path / "reference.wav")


def test_prepare_reference_rejects_audio_without_word_timestamps():
    model = FakeWhisperModel([
        make_segments("Text ohne Zeiten", ()),
    ])
    adapter = FasterWhisperAdapter(model)

    with pytest.raises(WhisperError, match="keine Wortzeitstempel"):
        prepare_reference_sample(Path("input.wav"), Path("sample.wav"), adapter)


def test_transcribe_rejects_empty_speech_result():
    model = FakeWhisperModel([make_segments("   ")])
    adapter = FasterWhisperAdapter(model)

    with pytest.raises(WhisperError, match="leeres Transkript"):
        adapter.transcribe(Path("sample.wav"))


def test_load_local_rejects_unlisted_models_before_loading(tmp_path):
    with pytest.raises(WhisperError, match="Nicht unterstuetztes Whisper-Modell"):
        FasterWhisperAdapter.load_local("large-v3-turbo", tmp_path)


def test_load_local_rejects_incomplete_cache_before_loading(tmp_path):
    model_path = tmp_path / "base"
    model_path.mkdir()
    (model_path / "config.json").touch()

    with pytest.raises(WhisperError, match="unvollstaendig"):
        FasterWhisperAdapter.load_local("base", tmp_path)


def test_load_local_rejects_empty_model_weights(tmp_path):
    model_path = tmp_path / "base"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"valid")
    (model_path / "model.bin").write_bytes(b"")
    (model_path / "tokenizer.json").write_bytes(b"valid")

    with pytest.raises(WhisperError, match="model.bin \\(empty\\)"):
        FasterWhisperAdapter.load_local("base", tmp_path)