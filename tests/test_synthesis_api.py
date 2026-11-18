import asyncio
import json
import threading
import time
import wave

from fastapi.testclient import TestClient

from hoerbuchgenerator import synthesis_jobs
from hoerbuchgenerator.main import create_app
from hoerbuchgenerator.synthesis_jobs import _estimate_remaining_seconds
from hoerbuchgenerator.worker import VoiceWorkerCancelled, VoiceWorkerCommandError


class FakeSynthesisWorker:
    def __init__(self):
        self.ready = True

    async def execute(self, command, cancel_event):
        if command["text"].startswith("FAIL"):
            raise VoiceWorkerCommandError("synthetic chunk failure")
        with wave.open(command["destination"], "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24_000)
            audio.writeframes(b"\x10\x00" * 100)
        return {"sample_rate": 24_000}

    async def close(self):
        self.ready = False


def create_client(tmp_path):
    app = create_app(tmp_path, voice_profiles=object())
    app.state.voice_worker = FakeSynthesisWorker()
    voice = tmp_path / "voices" / "test_voice"
    voice.mkdir(parents=True)
    (voice / "metadata.json").write_text(
        json.dumps({"language": "en"}),
        encoding="utf-8",
    )
    (voice / "voice-prompt.npz").write_bytes(b"prompt")
    return TestClient(app)


def test_estimated_remaining_seconds_uses_average_completed_chunk_time():
    assert _estimate_remaining_seconds(10.0, 1, 5) == 50.0
    assert _estimate_remaining_seconds(20.0, 2, 3) == 30.0
    assert _estimate_remaining_seconds(10.0, 0, 3) is None
    assert _estimate_remaining_seconds(10.0, 3, 0) is None


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["state"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Synthesis job did not finish")


def test_text_api_synthesizes_and_downloads_result(tmp_path):
    with create_client(tmp_path) as client:
        accepted = client.post(
            "/api/v1/synthesis/text",
            data={
                "text": "A short synthetic test sentence.",
                "name": "Chapter One",
                "voice": "test_voice",
                "language": "en",
                "format": "wav",
            },
        )
        assert accepted.status_code == 202
        job = wait_for_job(client, accepted.json()["id"])

        assert job["state"] == "completed", job
        assert "estimated_remaining_seconds" not in job
        assert job["result"]["files"] == [
            {"state": "completed", "name": "chapter_one.wav", "format": "wav", "chunks": 1}
        ]
        downloaded = client.get("/api/v1/output/chapter_one.wav")
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"] == "audio/wav"
        assert client.get("/api/v1/output/../../README.md").status_code == 404
        catalog = client.get("/api/v1/output").json()["files"]
        assert [item["name"] for item in catalog] == ["chapter_one.wav"]
        assert (tmp_path / "config" / "output-catalog.json").is_file()


def test_job_events_stream_sends_current_terminal_snapshot(tmp_path):
    with create_client(tmp_path) as client:
        accepted = client.post(
            "/api/v1/synthesis/text",
            data={
                "text": "A short synthetic test sentence.",
                "name": "Chapter One",
                "voice": "test_voice",
                "format": "wav",
            },
        )
        job = wait_for_job(client, accepted.json()["id"])
        response = client.get(f"/api/v1/jobs/{job['id']}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("event: job\ndata: ")
    event_data = response.text.split("data: ", 1)[1].strip()
    assert json.loads(event_data)["state"] == "completed"


def test_job_events_stream_rejects_unknown_job(tmp_path):
    with create_client(tmp_path) as client:
        response = client.get("/api/v1/jobs/unknown/events")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "job_not_found"


def test_file_api_processes_multiple_text_files_sequentially(tmp_path):
    with create_client(tmp_path) as client:
        accepted = client.post(
            "/api/v1/synthesis/files",
            data={"voice": "test_voice", "format": "wav"},
            files=[
                ("files", ("first.txt", b"First chapter text.", "text/plain")),
                ("files", ("second.txt", b"Second chapter text.", "text/plain")),
            ],
        )

        assert accepted.status_code == 202
        job = wait_for_job(client, accepted.json()["id"])
        assert [
            item["name"] for item in client.get("/api/v1/output").json()["files"]
        ] == ["second.wav", "first.wav"]

    assert job["state"] == "completed", job
    assert [item["name"] for item in job["result"]["files"]] == [
        "first.wav",
        "second.wav",
    ]


def test_long_text_api_reports_chunk_progress_until_completion(tmp_path, monkeypatch):
    class PausedSynthesisWorker(FakeSynthesisWorker):
        def __init__(self):
            super().__init__()
            self.first_chunk_started = threading.Event()
            self.release_first_chunk = threading.Event()

        async def execute(self, command, cancel_event):
            self.first_chunk_started.set()
            await asyncio.to_thread(self.release_first_chunk.wait, 10)
            return await super().execute(command, cancel_event)

    app = create_app(tmp_path, voice_profiles=object())
    worker = PausedSynthesisWorker()
    app.state.voice_worker = worker
    clock = [100.0]
    monkeypatch.setattr(synthesis_jobs, "monotonic", lambda: clock[0])
    snapshots = []
    publish = app.state.jobs._publish

    def capture_snapshot(job_id, snapshot):
        snapshots.append(snapshot.copy())
        publish(job_id, snapshot)

    app.state.jobs._publish = capture_snapshot
    voice = tmp_path / "voices" / "test_voice"
    voice.mkdir(parents=True)
    (voice / "metadata.json").write_text(
        json.dumps({"language": "en"}),
        encoding="utf-8",
    )
    (voice / "voice-prompt.npz").write_bytes(b"prompt")
    paragraphs = [
        f"Chapter section {index} contains a synthetic test sentence."
        for index in range(12)
    ]

    try:
        with TestClient(app) as client:
            accepted = client.post(
                "/api/v1/synthesis/text",
                data={
                    "text": "\n\n".join(paragraphs),
                    "name": "Long Book",
                    "voice": "test_voice",
                    "language": "en",
                    "format": "wav",
                },
            )
            assert accepted.status_code == 202
            job_id = accepted.json()["id"]
            assert worker.first_chunk_started.wait(10)

            active_job = client.get(f"/api/v1/jobs/{job_id}").json()
            assert active_job["state"] == "running"
            assert active_job["phase"] == "synthesizing"
            assert active_job["total"] == len(paragraphs)
            assert active_job["completed"] == 0
            assert "estimated_remaining_seconds" not in active_job

            clock[0] = 110.0
            worker.release_first_chunk.set()
            job = wait_for_job(client, job_id)

        assert job["state"] == "completed", job
        assert job["total"] == len(paragraphs)
        assert job["completed"] == len(paragraphs)
        assert job["result"]["files"][0]["chunks"] == len(paragraphs)
        assert not (tmp_path / "work" / job_id).exists()
        first_estimate = next(
            snapshot
            for snapshot in snapshots
            if snapshot.get("estimated_remaining_seconds") is not None
        )
        assert first_estimate["completed"] == 1
        assert first_estimate["estimated_remaining_seconds"] == 110.0
    finally:
        worker.release_first_chunk.set()


def test_bulk_synthesis_keeps_completed_files_when_one_file_fails(tmp_path):
    with create_client(tmp_path) as client:
        accepted = client.post(
            "/api/v1/synthesis/files",
            data={"voice": "test_voice", "format": "wav"},
            files=[
                ("files", ("good.txt", b"Successful chapter text.", "text/plain")),
                ("files", ("broken.txt", b"FAIL this file.", "text/plain")),
            ],
        )
        assert accepted.status_code == 202
        job = wait_for_job(client, accepted.json()["id"])
        assert [
            item["name"] for item in client.get("/api/v1/output").json()["files"]
        ] == ["good.wav"]

    assert job["state"] == "completed", job
    assert [item["state"] for item in job["result"]["files"]] == [
        "completed",
        "failed",
    ]
    assert job["total"] == 2
    assert job["completed"] == 1


def test_partial_file_failure_excludes_unfinished_chunks_from_eta(
    tmp_path,
    monkeypatch,
):
    clock = [100.0]
    monkeypatch.setattr(synthesis_jobs, "monotonic", lambda: clock[0])

    class TimedFailureWorker(FakeSynthesisWorker):
        async def execute(self, command, cancel_event):
            if command["text"].startswith("Successful"):
                clock[0] = 110.0
            return await super().execute(command, cancel_event)

    app = create_app(tmp_path, voice_profiles=object())
    app.state.voice_worker = TimedFailureWorker()
    snapshots = []
    publish = app.state.jobs._publish

    def capture_snapshot(job_id, snapshot):
        snapshots.append(snapshot.copy())
        publish(job_id, snapshot)

    app.state.jobs._publish = capture_snapshot
    voice = tmp_path / "voices" / "test_voice"
    voice.mkdir(parents=True)
    (voice / "metadata.json").write_text(
        json.dumps({"language": "en"}),
        encoding="utf-8",
    )
    (voice / "voice-prompt.npz").write_bytes(b"prompt")

    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/synthesis/files",
            data={"voice": "test_voice", "format": "wav"},
            files=[
                (
                    "files",
                    (
                        "partial.txt",
                        b"Successful first paragraph.\n\nFAIL second paragraph.",
                        "text/plain",
                    ),
                ),
                ("files", ("later.txt", b"Later successful paragraph.", "text/plain")),
            ],
        )
        assert accepted.status_code == 202
        job = wait_for_job(client, accepted.json()["id"])

    assert job["state"] == "completed"
    assert job["total"] == 3
    assert job["completed"] == 2
    assert [item["state"] for item in job["result"]["files"]] == [
        "failed",
        "completed",
    ]
    after_failed_file = next(
        snapshot
        for snapshot in snapshots
        if snapshot["completed"] == 1
        and snapshot["estimated_remaining_seconds"] == 10.0
    )
    assert after_failed_file["total"] == 3


def test_bulk_publishes_first_file_while_running_and_keeps_it_after_cancel(tmp_path):
    class PausedSecondWorker(FakeSynthesisWorker):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        async def execute(self, command, cancel_event):
            if command["text"].startswith("Second"):
                self.started.set()
                await asyncio.to_thread(self.release.wait, 10)
                if cancel_event.is_set():
                    raise VoiceWorkerCancelled()
            return await super().execute(command, cancel_event)

    worker = PausedSecondWorker()
    client = create_client(tmp_path)
    client.app.state.voice_worker = worker
    try:
        with client:
            accepted = client.post(
                "/api/v1/synthesis/files",
                data={"voice": "test_voice", "format": "wav"},
                files=[
                    ("files", ("first.txt", b"First chapter.", "text/plain")),
                    ("files", ("second.txt", b"Second chapter.", "text/plain")),
                ],
            )
            job_id = accepted.json()["id"]
            assert worker.started.wait(10)
            assert client.get(f"/api/v1/jobs/{job_id}").json()["state"] == "running"
            files = client.get("/api/v1/output").json()["files"]
            assert [item["name"] for item in files] == ["first.wav"]
            assert client.get(files[0]["url"]).status_code == 200
            assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 200
            worker.release.set()
            assert wait_for_job(client, job_id)["state"] == "cancelled"
            assert client.get("/api/v1/output").json()["files"] == files
    finally:
        worker.release.set()
