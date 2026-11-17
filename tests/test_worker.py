import asyncio
import os
from pathlib import Path
import time
from threading import Event
from types import SimpleNamespace

import pytest

from hoerbuchgenerator.worker import (
    VoiceWorker,
    VoiceWorkerCancelled,
    VoiceWorkerCommandError,
    VoiceWorkerError,
)


def fake_worker(connection, settings):
    connection.send({"type": "ready"})
    while True:
        message = connection.recv()
        if message["type"] == "stop":
            return
        command = message["command"]
        if command["action"] == "crash":
            os._exit(1)
        if command["action"] == "block":
            Path(command["marker"]).write_text(str(os.getpid()))
            while True:
                time.sleep(0.1)
        connection.send({"type": "completed", "result": command["value"]})


class FakeModelService:
    def create(self, upload, filename, name, voices_dir, work_dir):
        assert upload.read() == b"reference"
        assert filename == "reference.wav"
        return SimpleNamespace(name=name, slug="test_voice", language="de")

    def synthesize(self, text, language, prompt_path):
        import numpy as np

        assert text == "Ein Test."
        assert language == "de"
        assert prompt_path.name == "voice-prompt.npz"
        return np.zeros(240, dtype=np.float32), 24_000


def create_fake_model_service(settings):
    if settings.get("fail_start"):
        raise RuntimeError("model startup failed")
    return FakeModelService()


def model_runtime_worker(connection, settings):
    from hoerbuchgenerator import worker_runtime

    worker_runtime._create_voice_service = create_fake_model_service
    worker_runtime._run_voice_worker(connection, settings)


def test_spawned_model_runtime_dispatches_commands_and_survives_command_error(tmp_path):
    import wave

    async def exercise():
        worker = VoiceWorker(
            {"qwen_model": "fake-qwen", "whisper_model": "fake-whisper"},
            target=model_runtime_worker,
            startup_timeout=10,
        )
        await worker.start()
        try:
            with pytest.raises(VoiceWorkerCommandError, match="Unbekannter"):
                await worker.execute({"action": "unknown"}, Event())

            upload = tmp_path / "reference.wav"
            upload.write_bytes(b"reference")
            result = await worker.create_voice(
                upload_path=upload,
                filename=upload.name,
                name="Test Voice",
                voices_dir=tmp_path / "voices",
                work_dir=tmp_path / "work",
                cancel_event=Event(),
            )
            assert result == {"name": "Test Voice", "slug": "test_voice", "language": "de"}

            destination = tmp_path / "chunk.wav"
            result = await worker.execute(
                {
                    "action": "synthesize_chunk",
                    "text": "Ein Test.",
                    "language": "de",
                    "voice_dir": str(tmp_path / "voices" / "test_voice"),
                    "destination": str(destination),
                },
                Event(),
            )
            assert result == {"sample_rate": 24_000}
            with wave.open(str(destination), "rb") as audio:
                assert audio.getnframes() == 240
                assert audio.getframerate() == 24_000
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_model_runtime_startup_failure_releases_process():
    async def exercise():
        worker = VoiceWorker(
            {"fail_start": "yes"},
            target=model_runtime_worker,
            startup_timeout=10,
        )
        try:
            with pytest.raises(VoiceWorkerError, match="model startup failed"):
                await worker.start()
            assert not worker.ready
            assert worker.process_id is None
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_worker_recovers_after_unexpected_process_exit():
    async def exercise():
        worker = VoiceWorker({}, target=fake_worker, startup_timeout=5)
        await worker.start()
        first_pid = worker.process_id
        try:
            with pytest.raises(VoiceWorkerError, match="unerwartet beendet"):
                await worker.execute({"action": "crash"}, Event())
            assert worker.ready
            assert worker.process_id != first_pid
            assert await worker.execute(
                {"action": "echo", "value": {"state": "ok"}},
                Event(),
            ) == {"state": "ok"}
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_worker_executes_commands_and_restarts_after_forced_cancellation(tmp_path):
    async def exercise():
        marker = tmp_path / "worker.pid"
        worker = VoiceWorker({}, target=fake_worker, startup_timeout=5)
        await worker.start()
        first_pid = worker.process_id
        cancel_event = Event()
        running = asyncio.create_task(
            worker.execute(
                {"action": "block", "marker": str(marker)},
                cancel_event,
            )
        )
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert marker.exists()
            assert int(marker.read_text()) == first_pid

            cancel_event.set()
            with pytest.raises(VoiceWorkerCancelled):
                await running

            assert worker.ready
            assert worker.process_id != first_pid
            result = await worker.execute(
                {"action": "echo", "value": {"state": "ok"}},
                Event(),
            )
            assert result == {"state": "ok"}
        finally:
            if not running.done():
                cancel_event.set()
                await running
            await worker.close()

    asyncio.run(exercise())
