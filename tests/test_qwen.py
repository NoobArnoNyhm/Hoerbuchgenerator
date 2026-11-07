import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QwenError, QwenTTSAdapter
from hoerbuchgenerator.voice_prompt import VoicePrompt


class FakeQwenModel:
    device = "cpu"

    def __init__(self):
        self.prompt_calls = []
        self.synthesis_calls = []

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return [SimpleNamespace(
            ref_code=np.arange(6, dtype=np.int16).reshape(3, 2),
            ref_spk_embedding=np.array([0.2, 0.8], dtype=np.float32),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=kwargs["ref_text"],
        )]

    def generate_voice_clone(self, **kwargs):
        self.synthesis_calls.append(kwargs)
        return [np.ones(100, dtype=np.float32)], 24_000


def test_local_loader_allows_only_base_models_and_requires_local_config(tmp_path):
    with pytest.raises(QwenError, match="Nicht unterstuetztes"):
        QwenTTSAdapter.load_local("Qwen3-TTS-12Hz-1.7B-VoiceDesign", tmp_path)

    with pytest.raises(QwenError, match="fehlt oder ist unvollstaendig"):
        QwenTTSAdapter.load_local(QWEN_BASE_MODELS[0], tmp_path)


def test_loader_passes_local_only_flag_to_model_loader(tmp_path, monkeypatch):
    model_path = tmp_path / QWEN_BASE_MODELS[0]
    model_path.mkdir()
    (model_path / "config.json").touch()
    calls = []
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    def fake_loader(path, **kwargs):
        calls.append((path, kwargs, os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]))
        return FakeQwenModel()

    adapter = QwenTTSAdapter.load_local(
        QWEN_BASE_MODELS[0],
        tmp_path,
        loader=fake_loader,
    )

    assert isinstance(adapter, QwenTTSAdapter)
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][2:] == ("1", "1")
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_qwen_prompt_is_created_persisted_reloaded_and_reused(tmp_path):
    model = FakeQwenModel()
    adapter = QwenTTSAdapter(model, tensor_factory=lambda array, device: array)
    prompt = adapter.create_voice_prompt(Path("reference.wav"), "Guten Morgen.")
    prompt_path = tmp_path / "voice-prompt.npz"
    adapter.save_voice_prompt(prompt_path, prompt)
    restored = adapter.load_voice_prompt(prompt_path)

    audio, sample_rate = adapter.synthesize("Hallo Welt.", "German", restored)

    assert model.prompt_calls[0]["x_vector_only_mode"] is False
    assert model.synthesis_calls[0]["voice_clone_prompt"][0].ref_text == "Guten Morgen."
    np.testing.assert_array_equal(audio, np.ones(100, dtype=np.float32))
    assert sample_rate == 24_000