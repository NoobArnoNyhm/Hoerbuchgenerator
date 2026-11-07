import json

import numpy as np
import pytest

from hoerbuchgenerator.voice_prompt import VoicePrompt, VoicePromptError, VoicePromptStore


def test_prompt_store_round_trips_numeric_arrays_and_metadata(tmp_path):
    path = tmp_path / "voice-prompt.npz"
    prompt = VoicePrompt(
        ref_code=np.arange(12, dtype=np.int16).reshape(4, 3),
        ref_spk_embedding=np.array([0.25, -0.5], dtype=np.float32),
        x_vector_only_mode=False,
        icl_mode=True,
        ref_text="Guten Morgen.",
    )

    VoicePromptStore.save(path, prompt)
    restored = VoicePromptStore.load(path)

    np.testing.assert_array_equal(restored.ref_code, prompt.ref_code)
    np.testing.assert_array_equal(restored.ref_spk_embedding, prompt.ref_spk_embedding)
    assert restored.icl_mode is True
    assert restored.x_vector_only_mode is False
    assert restored.ref_text == "Guten Morgen."


def test_prompt_store_supports_x_vector_only_prompt_without_ref_code(tmp_path):
    path = tmp_path / "voice-prompt.npz"
    prompt = VoicePrompt(
        ref_code=None,
        ref_spk_embedding=np.array([1.0, 2.0], dtype=np.float32),
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )

    VoicePromptStore.save(path, prompt)
    restored = VoicePromptStore.load(path)

    assert restored.ref_code is None
    assert restored.x_vector_only_mode is True
    assert restored.ref_text is None


def test_prompt_store_rejects_object_arrays_without_unpickling(tmp_path):
    path = tmp_path / "unsafe.npz"
    metadata = {
        "schema_version": VoicePromptStore.SCHEMA_VERSION,
        "has_ref_code": False,
        "x_vector_only_mode": True,
        "icl_mode": False,
        "ref_text": None,
    }
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata)),
        ref_spk_embedding=np.array([object()]),
    )

    with pytest.raises(VoicePromptError, match="ungueltig"):
        VoicePromptStore.load(path)


def test_prompt_store_rejects_invalid_tensor_values(tmp_path):
    prompt = VoicePrompt(
        ref_code=None,
        ref_spk_embedding=np.array([float("nan")]),
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )

    with pytest.raises(VoicePromptError, match="nicht-endliche"):
        VoicePromptStore.save(tmp_path / "bad.npz", prompt)