import asyncio
import os
from pathlib import Path
import time
from threading import Event

import pytest

from hoerbuchgenerator.worker import VoiceWorker, VoiceWorkerCancelled


def fake_worker(connection, settings):
    connection.send({"type": "ready"})
    while True:
        message = connection.recv()
        if message["type"] == "stop":
            return
        command = message["command"]
        if command["action"] == "block":
            Path(command["marker"]).write_text(str(os.getpid()))
            while True:
                time.sleep(0.1)
        connection.send({"type": "completed", "result": command["value"]})


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
