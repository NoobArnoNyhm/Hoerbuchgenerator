import asyncio
import json
import time
from threading import Event
from types import SimpleNamespace
import wave

import pytest
from fastapi.testclient import TestClient

from hoerbuchgenerator.main import create_app


class FakeVoiceProfiles:
    def __init__(self, *, entered=None, release=None):
        self.entered = entered
        self.release = release

    def create(self, upload, filename, name, voices_dir, work_dir, *, cancel_event):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        assert upload.read() == b"audio"
        example_dir = voices_dir / "test-voice"
        example_dir.mkdir(parents=True, exist_ok=True)
        with wave.open(str(example_dir / "example.wav"), "wb") as example:
            example.setnchannels(1)
            example.setsampwidth(2)
            example.setframerate(24000)
            example.writeframes(b"\0\0" * 10)
        return SimpleNamespace(name=name, slug="test-voice", language="de")


def create_upload(client, name="Teststimme"):
    return client.post(
        "/api/v1/voices",
        data={"name": name},
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )


def create_voice_profile(voices_dir, slug, *, name, language="de", example=True):
    profile_dir = voices_dir / slug
    profile_dir.mkdir(parents=True)
    (profile_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "slug": slug,
                "language": language,
                "language_probability": 0.99,
                "transcript": "private transcript",
                "qwen_model": "Qwen3-TTS-12Hz-0.6B-Base",
                "created_at": "2026-11-07T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    if example:
        with wave.open(str(profile_dir / "example.wav"), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24_000)
            audio.writeframes(b"\0\0" * 10)
    return profile_dir


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        if response.json()["state"] in {"completed", "failed", "cancelled"}:
            return response.json()
        time.sleep(0.01)
    raise AssertionError("Voice job did not finish")


def test_voice_list_returns_sorted_public_metadata_and_example_urls(tmp_path):
    voices_dir = tmp_path / "voices"
    create_voice_profile(voices_dir, "zweite_stimme", name="Zweite Stimme")
    create_voice_profile(
        voices_dir,
        "erste_stimme",
        name="Erste Stimme",
        language="en",
        example=False,
    )
    staging_dir = voices_dir / ".temporary-profile"
    staging_dir.mkdir()
    (staging_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/voices")

    assert response.status_code == 200
    assert response.json() == {
        "voices": [
            {
                "slug": "erste_stimme",
                "name": "Erste Stimme",
                "language": "en",
                "qwen_model": "Qwen3-TTS-12Hz-0.6B-Base",
                "created_at": "2026-11-07T12:00:00+00:00",
                "example_audio_url": None,
            },
            {
                "slug": "zweite_stimme",
                "name": "Zweite Stimme",
                "language": "de",
                "qwen_model": "Qwen3-TTS-12Hz-0.6B-Base",
                "created_at": "2026-11-07T12:00:00+00:00",
                "example_audio_url": "/api/v1/voices/zweite_stimme/example",
            },
        ]
    }
    assert "transcript" not in response.text


def test_voice_example_endpoint_downloads_only_profile_example(tmp_path):
    create_voice_profile(tmp_path / "voices", "test_voice", name="Teststimme")

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/voices/test_voice/example")
        missing = client.get("/api/v1/voices/unknown/example")
        traversal = client.get("/api/v1/voices/%2e%2e/example")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert missing.status_code == 404
    assert traversal.status_code == 404


@pytest.mark.parametrize(
    "metadata",
    ["not-json", json.dumps({"name": "Broken", "slug": "bad_voice"})],
)
def test_voice_list_reports_invalid_profile_metadata(tmp_path, metadata):
    profile_dir = tmp_path / "voices" / "bad_voice"
    profile_dir.mkdir(parents=True)
    (profile_dir / "metadata.json").write_text(metadata, encoding="utf-8")

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/v1/voices")

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "voice_profile_invalid"


def test_voice_endpoint_accepts_async_job_and_exposes_result(tmp_path):
    with TestClient(create_app(tmp_path, FakeVoiceProfiles())) as client:
        accepted = create_upload(client)

        assert accepted.status_code == 202
        job_id = accepted.json()["id"]
        job = wait_for_job(client, job_id)

    assert job["state"] == "completed"
    assert job["result"] == {
        "name": "Teststimme",
        "slug": "test-voice",
        "language": "de",
    }


def test_voice_endpoint_rejects_busy_request_and_cleans_staged_upload(tmp_path):
    entered = Event()
    release = Event()
    service = FakeVoiceProfiles(entered=entered, release=release)
    with TestClient(create_app(tmp_path, service)) as client:
        accepted = create_upload(client)
        assert accepted.status_code == 202
        assert entered.wait(timeout=5)

        rejected = create_upload(client, "Zweite Stimme")

        assert rejected.status_code == 403
        assert rejected.json()["detail"]["code"] == "busy"
        assert len(list((tmp_path / "work").iterdir())) == 1
        release.set()
        completed = wait_for_job(client, accepted.json()["id"])

    assert completed["state"] == "completed"
    assert list((tmp_path / "work").iterdir()) == []


def test_voice_endpoint_requires_configured_model_service(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = create_upload(client)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "voice_service_not_initialized"


def test_gradio_submission_uses_shared_job_and_displays_created_example(tmp_path):
    app = create_app(tmp_path, FakeVoiceProfiles())
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"audio")

    async def submit_and_refresh():
        submitted = await ui.fns[0].fn(str(audio_path), "Teststimme")
        await asyncio.gather(*tuple(app.state.background_tasks))
        refreshed = await ui.fns[2].fn(submitted[0], submitted[3])
        return submitted, refreshed

    submitted, refreshed = asyncio.run(submit_and_refresh())

    assert submitted[1] == "Stimmenauftrag angenommen."
    assert submitted[2] is False
    assert refreshed[0] == submitted[0]
    assert "Stimme **Teststimme** wurde erstellt." in refreshed[1]
    assert refreshed[2][0] == 24000
    assert len(refreshed[2][1]) == 10
    assert refreshed[3]["interactive"] is True
    assert refreshed[4]["interactive"] is False


def test_gradio_submission_respects_shared_busy_job_slot(tmp_path):
    app = create_app(tmp_path, FakeVoiceProfiles())
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"audio")

    async def submit_while_busy():
        await app.state.jobs.reserve("voice")
        return await ui.fns[0].fn(str(audio_path), "Teststimme")

    submitted = asyncio.run(submit_while_busy())

    assert submitted[0] == ""
    assert submitted[1] == "Stimme konnte nicht gestartet werden: busy"
    assert submitted[2] is True
