import time
from threading import Event
from types import SimpleNamespace

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
        return SimpleNamespace(name=name, slug="test-voice", language="de")


def create_upload(client, name="Teststimme"):
    return client.post(
        "/api/v1/voices",
        data={"name": name},
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        if response.json()["state"] in {"completed", "failed", "cancelled"}:
            return response.json()
        time.sleep(0.01)
    raise AssertionError("Voice job did not finish")


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
