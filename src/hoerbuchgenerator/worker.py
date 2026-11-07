import asyncio
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable
import wave

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


def _create_voice_service(settings: dict[str, str]):
    import torch

    from hoerbuchgenerator.qwen import QwenTTSAdapter
    from hoerbuchgenerator.voice_profiles import VoiceProfileService
    from hoerbuchgenerator.voice_reference import VoiceReferenceService
    from hoerbuchgenerator.whisper import FasterWhisperAdapter

    whisper = FasterWhisperAdapter.load_local(
        settings["whisper_model"],
        Path(settings["whisper_models_dir"]),
        device=settings["whisper_device"],
        compute_type=settings["whisper_compute_type"],
    )
    qwen = QwenTTSAdapter.load_local(
        settings["qwen_model"],
        Path(settings["qwen_models_dir"]),
        loader_kwargs={
            "device_map": settings["qwen_device_map"],
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )
    return VoiceProfileService(
        VoiceReferenceService(whisper),
        qwen,
        settings["qwen_model"],
    )


def _run_voice_worker(connection: Connection, settings: dict[str, str]) -> None:
    try:
        service = _create_voice_service(settings)
        connection.send(
            {
                "type": "ready",
                "models": {
                    "qwen": settings["qwen_model"],
                    "whisper": settings["whisper_model"],
                },
            }
        )
    except Exception as error:
        logger.exception("Lokale Modelle konnten im Stimmen-Worker nicht geladen werden")
        connection.send({"type": "startup_failed", "error": str(error)})
        connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message.get("type") == "stop":
            return
        if message.get("type") != "execute":
            continue

        command = message["command"]
        if command.get("action") not in {"create_voice", "synthesize_chunk"}:
            connection.send({"type": "failed", "error": "Unbekannter Worker-Auftrag"})
            continue
        try:
            if command["action"] == "synthesize_chunk":
                audio, sample_rate = service.synthesize(
                    command["text"],
                    command["language"],
                    Path(command["voice_dir"]) / "voice-prompt.npz",
                )
                from hoerbuchgenerator.synthesis import _write_pcm

                _write_pcm(Path(command["destination"]), audio, sample_rate)
                connection.send({"type": "completed", "result": {"sample_rate": sample_rate}})
                continue

            with Path(command["upload_path"]).open("rb") as upload:
                created = service.create(
                    upload,
                    command["filename"],
                    command["name"],
                    Path(command["voices_dir"]),
                    Path(command["work_dir"]),
                )
            connection.send(
                {
                    "type": "completed",
                    "result": {
                        "name": created.name,
                        "slug": created.slug,
                        "language": created.language,
                    },
                }
            )
        except Exception as error:
            logger.exception("Stimmenauftrag im Worker fehlgeschlagen")
            connection.send(
                {
                    "type": "failed",
                    "error": str(error) if isinstance(error, RuntimeError) else (
                        "Interner Fehler bei der Stimmerstellung"
                    ),
                }
            )
