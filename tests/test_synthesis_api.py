import asyncio
import json
import threading
import time
import wave

from fastapi.testclient import TestClient

from hoerbuchgenerator.main import create_app
from hoerbuchgenerator.worker import VoiceWorkerCommandError


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
        assert job["result"]["files"] == [
            {"state": "completed", "name": "chapter_one.wav", "format": "wav", "chunks": 1}
        ]
        downloaded = client.get("/api/v1/output/chapter_one.wav")
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"] == "audio/wav"
        assert client.get("/api/v1/output/../../README.md").status_code == 404


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

    assert job["state"] == "completed", job
    assert [item["name"] for item in job["result"]["files"]] == [
        "first.wav",
        "second.wav",
    ]


def test_long_text_api_reports_chunk_progress_until_completion(tmp_path):
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

            worker.release_first_chunk.set()
            job = wait_for_job(client, job_id)

        assert job["state"] == "completed", job
        assert job["total"] == len(paragraphs)
        assert job["completed"] == len(paragraphs)
        assert job["result"]["files"][0]["chunks"] == len(paragraphs)
        assert not (tmp_path / "work" / job_id).exists()
    finally:
        worker.release_first_chunk.set()


def test_bulk_synthesis_keeps_completed_files_when_one_file_fails(tmp_path):
    with create_client(tmp_path) as client:
        accepted = client.post(
            "/api/v1/synthesis/files",
            data={"voice": "test_voice", "format": "wav"},
            files=[
                ("files", ("broken.txt", b"FAIL this file.", "text/plain")),
                ("files", ("good.txt", b"Successful chapter text.", "text/plain")),
            ],
        )
        assert accepted.status_code == 202
        job = wait_for_job(client, accepted.json()["id"])

    assert job["state"] == "completed", job
    assert [item["state"] for item in job["result"]["files"]] == [
        "failed",
        "completed",
    ]
