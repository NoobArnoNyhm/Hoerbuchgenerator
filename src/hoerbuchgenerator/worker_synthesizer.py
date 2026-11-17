import asyncio
from concurrent.futures import Future
from pathlib import Path
from threading import Event

from hoerbuchgenerator.synthesis import SynthesisError
from hoerbuchgenerator.worker import VoiceWorker, VoiceWorkerCommandError


class WorkerSynthesizer:
    def __init__(self, worker: VoiceWorker, loop: asyncio.AbstractEventLoop, cancel_event: Event):
        self._worker = worker
        self._loop = loop
        self._cancel_event = cancel_event

    def synthesize_chunk(
        self,
        *,
        voice_dir: Path,
        text: str,
        language: str,
        destination: Path,
    ) -> None:
        future: Future = asyncio.run_coroutine_threadsafe(
            self._worker.execute(
                {
                    "action": "synthesize_chunk",
                    "voice_dir": str(voice_dir),
                    "text": text,
                    "language": language,
                    "destination": str(destination),
                },
                self._cancel_event,
            ),
            self._loop,
        )
        try:
            future.result()
        except VoiceWorkerCommandError as error:
            raise SynthesisError(str(error)) from error
