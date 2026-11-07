import shutil
import wave

import pytest

from hoerbuchgenerator.audio import (
    AudioError,
    WordInterval,
    extract_audio_segment,
    normalize_audio,
    select_reference_window,
)


def make_words(duration: float, word_duration: float = 0.5) -> list[WordInterval]:
    words = []
    start = 0.0
    while start + word_duration <= duration:
        words.append(WordInterval(start, start + word_duration))
        start += word_duration
    return words


def test_reference_window_uses_complete_word_boundaries():
    start, end = select_reference_window(make_words(90))

    assert 20 <= end - start <= 40
    assert start % 0.5 == 0
    assert end % 0.5 == 0


def test_short_recording_uses_available_complete_speech():
    start, end = select_reference_window(make_words(12))

    assert start == 0
    assert end == 12


@pytest.mark.parametrize("words", [[], [WordInterval(0, 0)]])
def test_reference_window_rejects_missing_or_invalid_word_timestamps(words):
    with pytest.raises(AudioError):
        select_reference_window(words)


def test_reference_window_rejects_unsorted_words():
    words = [WordInterval(2, 2.5), WordInterval(1, 1.5)]

    with pytest.raises(AudioError):
        select_reference_window(words)


def test_normalize_audio_rejects_unsupported_file_types_before_ffmpeg(tmp_path):
    source = tmp_path / "reference.txt"
    source.write_text("not audio")

    with pytest.raises(AudioError, match="WAV- und MP3"):
        normalize_audio(source, tmp_path / "normalized.wav")


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (float("nan"), 2)])
def test_extract_audio_segment_rejects_invalid_time_ranges(tmp_path, start, end):
    with pytest.raises(AudioError, match="Zeitgrenzen"):
        extract_audio_segment(tmp_path / "source.wav", tmp_path / "clip.wav", start, end)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and FFprobe are required for audio integration tests",
)
def test_normalize_and_extract_produce_valid_pcm_wav(tmp_path):
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48_000)
        wav_file.writeframes(bytes(4_800 * 2 * 2))

    normalized = normalize_audio(source, tmp_path / "normalized.wav")
    sample = extract_audio_segment(normalized.path, tmp_path / "sample.wav", 0.02, 0.08)

    assert normalized.sample_rate == 24_000
    assert normalized.channels == 1
    assert normalized.sample_width == 2
    assert sample.duration == pytest.approx(0.06, abs=0.001)