from fastapi.testclient import TestClient

from hoerbuchgenerator.main import create_app


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
    assert "Stimmen-Slug" in labels
    assert any(
        function.fn.__name__ == "submit_synthesis_ui"
        for function in blocks.fns.values()
    )
    assert "/api/v1/status" in app.openapi()["paths"]
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