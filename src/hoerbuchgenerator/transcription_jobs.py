import asyncio
from collections.abc import Callable
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.audio_types import MAX_UPLOAD_BYTES
from hoerbuchgenerator.jobs import JobSlotBusy
from hoerbuchgenerator.voice_reference import VoiceTranscriptionError
from hoerbuchgenerator.worker import VoiceWorkerCancelled, VoiceWorkerError


logger = logging.getLogger(__name__)


def create_transcription_job_handler(
    app: FastAPI,
    root: Path,
    voice_service_available: Callable[[], bool],
):
    async def accept_transcription_upload(
        filename: str,
        read_chunk: Callable[[int], Any],
    ) -> dict[str, str]:
        if not voice_service_available():
            raise HTTPException(
                status_code=503,
                detail={"code": "voice_service_not_initialized"},
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".mp3"}:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsupported_audio_format"},
            )

        if app.state.voice_worker is None and not callable(
            getattr(app.state.voice_profiles, "transcribe_audio", None)
        ):
            raise HTTPException(
                status_code=503,
                detail={"code": "whisper_not_available"},
            )

        try:
            job = await app.state.jobs.reserve("transcription")
        except JobSlotBusy as error:
            raise HTTPException(
                status_code=403,
                detail={"code": "busy"},
            ) from error

        work_root = root / "work" / job["id"]
        staged_upload: Path | None = None
        try:
            work_root.mkdir(parents=True, exist_ok=True)
            descriptor, staged_name = tempfile.mkstemp(
                prefix="transcription-upload-",
                suffix=suffix,
                dir=work_root,
            )
            staged_upload = Path(staged_name)
            size = 0
            with os.fdopen(descriptor, "wb") as staged:
                while chunk := await read_chunk(64 * 1024):
                    if not isinstance(chunk, bytes):
                        raise HTTPException(
                            status_code=422,
                            detail={"code": "invalid_upload"},
                        )
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail={"code": "upload_too_large"},
                        )
                    staged.write(chunk)
            if size == 0:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "empty_audio"},
                )
        except HTTPException:
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Transkriptionsupload wurde abgewiesen",
            )
            shutil.rmtree(work_root, ignore_errors=True)
            raise
        except asyncio.CancelledError:
            await app.state.jobs.finish(
                job["id"],
                "cancelled",
                error="Transkriptionsupload wurde abgebrochen",
            )
            shutil.rmtree(work_root, ignore_errors=True)
            raise
        except OSError as error:
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Transkriptionsupload konnte nicht gespeichert werden",
            )
            shutil.rmtree(work_root, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail={"code": "upload_staging_failed"},
            ) from error
        except Exception as error:
            logger.exception("Transkriptionsupload konnte nicht vorbereitet werden")
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Transkriptionsupload konnte nicht gespeichert werden",
            )
            shutil.rmtree(work_root, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail={"code": "upload_staging_failed"},
            ) from error

        async def run_transcription() -> None:
            job_id = job["id"]
            cancel_event = app.state.jobs.cancel_event(job_id)
            try:
                if not await app.state.jobs.update(job_id, "transcribing"):
                    await app.state.jobs.finish(job_id, "cancelled")
                    return

                if app.state.voice_worker is not None:
                    result = await app.state.voice_worker.transcribe_audio(
                        upload_path=staged_upload,
                        work_dir=work_root,
                        cancel_event=cancel_event,
                    )
                else:
                    transcription = await asyncio.to_thread(
                        app.state.voice_profiles.transcribe_audio,
                        staged_upload,
                        work_root,
                    )
                    result = {
                        "text": transcription.text,
                        "language": transcription.language,
                    }
                await app.state.jobs.finish(
                    job_id,
                    "completed",
                    result=result,
                )
            except VoiceWorkerCancelled:
                await app.state.jobs.finish(job_id, "cancelled")
            except VoiceWorkerError as error:
                app.state.voice_available = app.state.voice_worker.ready
                if not app.state.voice_available:
                    app.state.model_names = {"qwen": None, "whisper": None}
                if cancel_event.is_set():
                    await app.state.jobs.finish(job_id, "cancelled")
                else:
                    logger.exception(
                        "Transkriptions-Worker fuer Auftrag %s ausgefallen",
                        job_id,
                    )
                    await app.state.jobs.finish(job_id, "failed", error=str(error))
            except VoiceTranscriptionError as error:
                await app.state.jobs.finish(job_id, "failed", error=str(error))
            except asyncio.CancelledError:
                await app.state.jobs.finish(
                    job_id,
                    "cancelled",
                    error="Transkription wurde abgebrochen",
                )
                raise
            except Exception:
                logger.exception("Transkriptionsauftrag %s fehlgeschlagen", job_id)
                await app.state.jobs.finish(
                    job_id,
                    "failed",
                    error="Interner Fehler bei der Transkription",
                )
            finally:
                shutil.rmtree(work_root, ignore_errors=True)

        task = asyncio.create_task(run_transcription())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)
        return {"id": job["id"], "state": job["state"]}

    return accept_transcription_upload
