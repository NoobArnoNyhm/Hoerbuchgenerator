import os

import pytest
from fastapi.testclient import TestClient

from hoerbuchgenerator.main import create_app
from hoerbuchgenerator.qwen import QWEN_REQUIRED_FILES
from hoerbuchgenerator.whisper import WHISPER_REQUIRED_FILES


def test_status_is_versioned_and_does_not_claim_models_are_loaded(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/status")

    assert response.status_code == 200
    assert response.json() == {
        "state": "not_initialized",
        "active_job": None,
        "models": {"qwen": None, "whisper": None},
    }


def test_gradio_ui_is_mounted_separately_and_hidden_from_api_schema():
    app = create_app()
    mounted_ui = next(route.app for route in app.routes if route.path == "/ui")
    blocks = mounted_ui.get_blocks()
    labels = {
        component.get("props", {}).get("label")
        for component in blocks.config["components"]
    }

    assert blocks.api_open is False
    assert "Text" in labels
    assert "TXT-Dateien (optional)" in labels
    assert "Stimme" in labels
    assert any(
        function.fn.__name__ == "submit_synthesis_ui"
        for function in blocks.fns.values()
    )
    assert any(
        function.fn.__name__ == "load_voice_example_ui"
        for function in blocks.fns.values()
    )
    assert "/api/v1/status" in app.openapi()["paths"]
    assert "/api/v1/voices" in app.openapi()["paths"]
    assert "/api/v1/voices/{slug}/example" in app.openapi()["paths"]
    assert "/api/v1/synthesis/text" in app.openapi()["paths"]
    assert "/api/v1/synthesis/files" in app.openapi()["paths"]
    assert "/api/v1/output/{name}" in app.openapi()["paths"]
    assert not any(path.startswith("/ui") for path in app.openapi()["paths"])


def test_startup_creates_persistent_data_directories(tmp_path):
    with TestClient(create_app(tmp_path)):
        pass

    assert (tmp_path / "models/huggingface").is_dir()
    assert (tmp_path / "models/whisper").is_dir()
    assert (tmp_path / "voices").is_dir()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / "work").is_dir()
    assert (tmp_path / "config").is_dir()


def test_startup_cleans_orphaned_work_without_touching_persistent_data(tmp_path):
    orphaned_job = tmp_path / "work" / "abandoned-job"
    orphaned_job.mkdir(parents=True)
    (orphaned_job / "input.txt").write_text("temporary", encoding="utf-8")
    (tmp_path / "work" / "stray.tmp").write_text("temporary", encoding="utf-8")
    voice_profile = tmp_path / "voices" / "keep_voice"
    voice_profile.mkdir(parents=True)
    (voice_profile / "metadata.json").write_text("{}", encoding="utf-8")
    output_file = tmp_path / "output" / "keep.wav"
    output_file.parent.mkdir()
    output_file.write_bytes(b"finished")

    with TestClient(create_app(tmp_path)):
        pass

    assert list((tmp_path / "work").iterdir()) == []
    assert (voice_profile / "metadata.json").is_file()
    assert output_file.read_bytes() == b"finished"


def test_startup_skips_incomplete_local_model_caches(tmp_path, caplog):
    qwen_path = (
        tmp_path
        / "models"
        / "huggingface"
        / "Qwen3-TTS-12Hz-0.6B-Base"
    )
    whisper_path = tmp_path / "models" / "whisper" / "base"
    qwen_path.mkdir(parents=True)
    whisper_path.mkdir(parents=True)
    for filename in QWEN_REQUIRED_FILES:
        (qwen_path / filename).write_bytes(b"valid")
    (whisper_path / "config.json").write_bytes(b"valid")
    (whisper_path / "tokenizer.json").write_bytes(b"valid")

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/status")

    assert response.json()["state"] == "not_initialized"
    assert response.json()["models"] == {"qwen": None, "whisper": None}
    assert "model.bin (missing)" in caplog.text


def test_startup_skips_empty_local_model_files(tmp_path, caplog):
    qwen_path = tmp_path / "models" / "huggingface" / "Qwen3-TTS-12Hz-0.6B-Base"
    whisper_path = tmp_path / "models" / "whisper" / "base"
    qwen_path.mkdir(parents=True)
    whisper_path.mkdir(parents=True)
    for filename in QWEN_REQUIRED_FILES:
        (qwen_path / filename).write_bytes(b"valid")
    for filename in WHISPER_REQUIRED_FILES:
        (whisper_path / filename).write_bytes(
            b"" if filename == "model.bin" else b"valid"
        )

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/status")

    assert response.json()["state"] == "not_initialized"
    assert response.json()["models"] == {"qwen": None, "whisper": None}
    assert "model.bin (empty)" in caplog.text


def test_startup_loads_worker_from_complete_local_model_caches(
    tmp_path,
    monkeypatch,
):
    qwen_name = "Qwen3-TTS-12Hz-1.7B-Base"
    whisper_name = "medium"
    qwen_path = tmp_path / "models" / "huggingface" / qwen_name
    whisper_path = tmp_path / "models" / "whisper" / whisper_name
    qwen_path.mkdir(parents=True)
    whisper_path.mkdir(parents=True)
    for filename in QWEN_REQUIRED_FILES:
        (qwen_path / filename).write_bytes(b"valid")
    for filename in WHISPER_REQUIRED_FILES:
        (whisper_path / filename).write_bytes(b"valid")
    monkeypatch.setenv("QWEN_MODEL", qwen_name)
    monkeypatch.setenv("WHISPER_MODEL", whisper_name)

    workers = []

    class FakeWorker:
        def __init__(self, settings):
            self.settings = settings
            self.ready = False
            self.closed = False
            self.runtime_environment = {}
            workers.append(self)

        async def start(self):
            self.runtime_environment = {
                name: os.environ[name]
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_HUB_DISABLE_TELEMETRY",
                    "DO_NOT_TRACK",
                )
            }
            self.ready = True

        async def close(self):
            self.closed = True
            self.ready = False

    monkeypatch.setattr("hoerbuchgenerator.main.VoiceWorker", FakeWorker)

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/status")
        assert response.json()["state"] == "ready"
        assert response.json()["models"] == {
            "qwen": qwen_name,
            "whisper": whisper_name,
        }

    assert len(workers) == 1
    assert workers[0].settings["qwen_model"] == qwen_name
    assert workers[0].settings["whisper_model"] == whisper_name
    assert workers[0].settings["qwen_models_dir"] == str(
        tmp_path / "models" / "huggingface"
    )
    assert workers[0].settings["whisper_models_dir"] == str(
        tmp_path / "models" / "whisper"
    )
    assert workers[0].runtime_environment == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    assert workers[0].closed


def test_runtime_enforces_offline_model_loading_and_restores_environment(
    tmp_path,
    monkeypatch,
):
    variables = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    for variable in variables:
        monkeypatch.setenv(variable, "0")

    with TestClient(create_app(tmp_path)):
        assert {
            variable: os.environ[variable]
            for variable in variables
        } == variables

    assert {
        variable: os.environ[variable]
        for variable in variables
    } == {
        variable: "0" for variable in variables
    }


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        (
            "QWEN_MODEL",
            "Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            "Nicht unterstuetztes Qwen",
        ),
        (
            "WHISPER_MODEL",
            "large-v3-turbo",
            "Nicht unterstuetztes Whisper",
        ),
    ],
)
def test_startup_rejects_unsupported_model_selection(
    tmp_path,
    monkeypatch,
    variable,
    value,
    message,
):
    monkeypatch.setenv(variable, value)

    with pytest.raises(RuntimeError, match=message):
        with TestClient(create_app(tmp_path)):
            pass