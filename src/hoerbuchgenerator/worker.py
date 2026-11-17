import asyncio
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable
from hoerbuchgenerator.worker_runtime import _run_voice_worker

logger = logging.getLogger(__name__)


class VoiceWorkerError(RuntimeError):
    pass


class VoiceWorkerCancelled(VoiceWorkerError):
    pass


class VoiceWorkerCommandError(VoiceWorkerError):
    pass


class VoiceWorker:
    def __init__(
        self,
        settings: dict[str, str],
        *,
        target: Callable[[Connection, dict[str, str]], None] | None = None,
        startup_timeout: float = 600,
    ):
        self._settings = settings
        self._target = target or _run_voice_worker
        self._startup_timeout = startup_timeout
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._closed = False

    @property
    def ready(self) -> bool:
        return (
            not self._closed
            and self._process is not None
            and self._process.is_alive()
            and self._connection is not None
        )

    @property
    def process_id(self) -> int | None:
        return self._process.pid if self._process is not None else None

    async def start(self) -> None:
        self._closed = False
        await asyncio.to_thread(self._start_sync)

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._stop_sync, False)

    async def create_voice(
        self,
        *,
        upload_path: Path,
        filename: str,
        name: str,
        voices_dir: Path,
        work_dir: Path,
        cancel_event: Event,
    ) -> dict[str, str]:
        return await self.execute(
            {
                "action": "create_voice",
                "upload_path": str(upload_path),
                "filename": filename,
                "name": name,
                "voices_dir": str(voices_dir),
                "work_dir": str(work_dir),
            },
            cancel_event,
        )

    async def execute(
        self,
        command: dict[str, Any],
        cancel_event: Event,
    ) -> dict[str, Any]:
        if not self.ready or self._connection is None or self._process is None:
            raise VoiceWorkerError("Der Stimmen-Worker ist nicht verfuegbar")

        connection = self._connection
        process = self._process
        try:
            connection.send({"type": "execute", "command": command})
        except (BrokenPipeError, EOFError, OSError) as error:
            await self._recover()
            raise VoiceWorkerError("Der Stimmen-Worker hat die Verbindung verloren") from error

        while True:
            try:
                has_response = connection.poll()
            except (OSError, ValueError):
                await self._recover()
                raise VoiceWorkerError("Der Stimmen-Worker ist unerwartet beendet worden")
            if has_response:
                try:
                    response = connection.recv()
                except (EOFError, OSError) as error:
                    await self._recover()
                    raise VoiceWorkerError("Der Stimmen-Worker ist unerwartet beendet worden") from error
                if response.get("type") == "completed":
                    return response["result"]
                if response.get("type") == "failed":
                    raise VoiceWorkerCommandError(response["error"])
                raise VoiceWorkerError("Der Stimmen-Worker lieferte eine ungueltige Antwort")

            if cancel_event.is_set():
                await self._restart()
                raise VoiceWorkerCancelled("Stimmerstellung wurde abgebrochen")

            if not process.is_alive():
                await self._recover()
                raise VoiceWorkerError("Der Stimmen-Worker wurde unerwartet beendet")
            await asyncio.sleep(0.05)

    async def _recover(self) -> None:
        try:
            await self._restart()
        except VoiceWorkerError:
            logger.exception("Stimmen-Worker konnte nach einem Fehler nicht neu starten")

    async def _restart(self) -> None:
        await asyncio.to_thread(self._stop_sync, True)
        if self._closed:
            raise VoiceWorkerError("Der Stimmen-Worker wird beendet")
        try:
            await self.start()
        except Exception as error:
            self._closed = True
            raise VoiceWorkerError("Der Stimmen-Worker konnte nicht neu gestartet werden") from error

    def _start_sync(self) -> None:
        if self._closed:
            raise VoiceWorkerError("Der Stimmen-Worker wird beendet")
        if self.ready:
            return

        context = get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=self._target,
            args=(child, self._settings),
            name="hoerbuchgenerator-voice-worker",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent

        try:
            if not parent.poll(self._startup_timeout):
                raise VoiceWorkerError("Zeitueberschreitung beim Start des Stimmen-Workers")
            response = parent.recv()
            if response.get("type") != "ready":
                raise VoiceWorkerError(response.get("error", "Stimmen-Worker konnte nicht starten"))
        except Exception:
            self._stop_sync(True)
            raise

    def _stop_sync(self, force: bool) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None

        if connection is not None:
            if not force:
                try:
                    connection.send({"type": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            connection.close()

        if process is not None:
            process.join(timeout=0 if force else 5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
