import json
from pathlib import Path
import wave

import numpy as np
import pytest

from hoerbuchgenerator.synthesis import (
    PAUSE_BETWEEN_CHUNKS_SECONDS,
    PAUSE_BETWEEN_PARAGRAPHS_SECONDS,
    SynthesisError,
    _chunks_with_pauses,
    synthesize_file,
)


class FakeSynthesizer:
    def __init__(self):
        self.chunks = []

    def synthesize_chunk(self, *, voice_dir, text, language, destination):
        self.chunks.append((text, language))
        samples = (np.sin(np.arange(2400) / 10) * 0.1 * 32767).astype("<i2")
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24_000)
            output.writeframes(samples.tobytes())


def create_voice(tmp_path):
    voice_dir = tmp_path / "voices" / "test_voice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "metadata.json").write_text(
        json.dumps({"language": "de"}),
        encoding="utf-8",
    )
    (voice_dir / "voice-prompt.npz").write_bytes(b"prompt")
    return voice_dir


def test_chunks_preserve_sentence_and_paragraph_pause_boundaries():
    chunks = _chunks_with_pauses("Erster Satz. Zweiter Satz.\n\nNeuer Absatz.")

    assert [text for text, _ in chunks] == [
        "Erster Satz.",
        "Zweiter Satz.",
        "Neuer Absatz.",
    ]
    assert chunks[0][1] == PAUSE_BETWEEN_CHUNKS_SECONDS
    assert chunks[1][1] == PAUSE_BETWEEN_PARAGRAPHS_SECONDS
    assert chunks[2][1] == 0
    assert all("\r" not in text and "\n" not in text for text, _ in chunks)
    assert PAUSE_BETWEEN_CHUNKS_SECONDS < PAUSE_BETWEEN_PARAGRAPHS_SECONDS


@pytest.mark.parametrize("line_break", ["\n", "\r\n", "\r"])
def test_chunks_ignore_wrapped_lines_and_remove_line_breaks(line_break):
    wrapped = f"Erster Satz{line_break}wird fortgesetzt. Zweiter Satz."
    expected = _chunks_with_pauses(
        "Erster Satz wird fortgesetzt. Zweiter Satz."
    )

    chunks = _chunks_with_pauses(wrapped)

    assert chunks == expected
    assert all("\r" not in text and "\n" not in text for text, _ in chunks)
    assert chunks[0][1] == PAUSE_BETWEEN_CHUNKS_SECONDS


def test_synthesis_publishes_valid_wav_and_cleans_chunks(tmp_path):
    qwen = FakeSynthesizer()
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"

    result = synthesize_file(
        qwen,
        create_voice(tmp_path),
        "Ein kurzer Text fuer die Synthese.",
        "Kapitel Eins",
        output_dir,
        "wav",
        work_dir,
    )

    assert result == {"name": "kapitel_eins.wav", "format": "wav", "chunks": 1}
    with wave.open(str(output_dir / "kapitel_eins.wav"), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 24_000
    assert qwen.chunks == [("Ein kurzer Text fuer die Synthese.", "German")]
    assert not list(work_dir.iterdir())
    assert list(output_dir.iterdir()) == [output_dir / "kapitel_eins.wav"]


def test_synthesis_sends_linebreak_free_chunks_to_qwen(tmp_path):
    qwen = FakeSynthesizer()

    synthesize_file(
        qwen,
        create_voice(tmp_path),
        "Ein mehrzeiliger\nText bleibt\r\nzusammen.",
        "Mehrzeiliger Text",
        tmp_path / "output",
        "wav",
        tmp_path / "work",
    )

    assert qwen.chunks == [("Ein mehrzeiliger Text bleibt zusammen.", "German")]
    assert all("\r" not in text and "\n" not in text for text, _ in qwen.chunks)


def test_long_synthetic_book_reports_chunk_progress_and_preserves_paragraph_pauses(
    tmp_path,
):
    qwen = FakeSynthesizer()
    paragraphs = [
        f"Absatz {index} enthaelt einen eigenen synthetischen Testsatz."
        for index in range(12)
    ]
    text = "\n\n".join(paragraphs)
    progress = []
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"

    result = synthesize_file(
        qwen,
        create_voice(tmp_path),
        text,
        "Langes Testbuch",
        output_dir,
        "wav",
        work_dir,
        progress=progress.append,
    )

    assert result == {
        "name": "langes_testbuch.wav",
        "format": "wav",
        "chunks": len(paragraphs),
    }
    assert [chunk[0] for chunk in qwen.chunks] == paragraphs
    assert progress == list(range(1, len(paragraphs) + 1))
    with wave.open(str(output_dir / "langes_testbuch.wav"), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 24_000
        assert audio.getnframes() >= round(
            24_000
            * (
                len(paragraphs) * 0.1
                + (len(paragraphs) - 1) * 0.7
            )
        )
    assert not list(work_dir.iterdir())
    assert list(output_dir.iterdir()) == [output_dir / "langes_testbuch.wav"]


def test_synthesis_failure_does_not_publish_partial_output(tmp_path):
    class FailedSynthesizer:
        def synthesize_chunk(self, **kwargs):
            raise RuntimeError("worker failed")

    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    with pytest.raises(RuntimeError, match="worker failed"):
        synthesize_file(
            FailedSynthesizer(),
            create_voice(tmp_path),
            "Ein kurzer Text.",
            "Kapitel",
            output_dir,
            "wav",
            work_dir,
        )

    assert not output_dir.exists() or not list(output_dir.iterdir())
    assert not list(work_dir.iterdir())


def test_synthesis_creates_decodable_mp3(tmp_path):
    qwen = FakeSynthesizer()
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"

    result = synthesize_file(
        qwen,
        create_voice(tmp_path),
        "Ein kurzer MP3-Test.",
        "Kapitel-Zwei",
        output_dir,
        "mp3",
        work_dir,
    )

    assert result["name"] == "kapitel-zwei.mp3"
    assert (output_dir / "kapitel-zwei.mp3").stat().st_size > 0
    assert not list(work_dir.glob("*.wav"))


def test_synthesis_rejects_duplicate_destination(tmp_path):
    voice_dir = create_voice(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "kapitel.wav").write_bytes(b"existing")

    with pytest.raises(SynthesisError, match="existiert bereits"):
        synthesize_file(
            FakeSynthesizer(),
            voice_dir,
            "Ein kurzer Text.",
            "Kapitel",
            output_dir,
            "wav",
            tmp_path / "work",
        )

    assert (output_dir / "kapitel.wav").read_bytes() == b"existing"
