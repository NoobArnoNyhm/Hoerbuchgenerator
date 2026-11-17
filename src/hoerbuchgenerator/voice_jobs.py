import asyncio
from collections.abc import Callable
import logging
import os
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.audio_types import MAX_UPLOAD_BYTES
from hoerbuchgenerator.jobs import JobSlotBusy
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.voice_profiles import VoiceProfileError
from hoerbuchgenerator.worker import VoiceWorkerCancelled, VoiceWorkerError


logger = logging.getLogger(__name__)


def create_voice_job_handler(
    app: FastAPI,
    root: Path,
    voice_service_available: Callable[[], bool],
):
    async def accept_voice_upload(name: str, filename: str, read_chunk) -> dict[str, str]:
        if not voice_service_available():
            raise HTTPException(
                status_code=503,
                detail={"code": "voice_service_not_initialized"},
            )
        try:
            make_slug(name)
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_voice_name", "message": str(error)},
            ) from error

        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".mp3"}:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsupported_audio_format"},
            )

        try:
            job = await app.state.jobs.reserve("voice")
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
                prefix="voice-upload-",
                suffix=suffix,
                dir=work_root,
            )
            staged_upload = Path(staged_name)
            size = 0
            with os.fdopen(descriptor, "wb") as staged:
                while chunk := await read_chunk(64 * 1024):
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
                error="Stimmen-Upload wurde abgewiesen",
            )
            if staged_upload is not None:
                staged_upload.unlink(missing_ok=True)
            shutil.rmtree(work_root, ignore_errors=True)
            raise
        except asyncio.CancelledError:
            await app.state.jobs.finish(
                job["id"],
                "cancelled",
                error="Stimmen-Upload wurde abgebrochen",
            )
            if staged_upload is not None:
                staged_upload.unlink(missing_ok=True)
            shutil.rmtree(work_root, ignore_errors=True)
            raise
        except OSError as error:
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Stimmen-Upload konnte nicht gespeichert werden",
            )
            if staged_upload is not None:
                staged_upload.unlink(missing_ok=True)
            shutil.rmtree(work_root, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail={"code": "upload_staging_failed"},
            ) from error
        except Exception as error:
            logger.exception("Stimmen-Upload konnte nicht vorbereitet werden")
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Stimmen-Upload konnte nicht gespeichert werden",
            )
            if staged_upload is not None:
                staged_upload.unlink(missing_ok=True)
            shutil.rmtree(work_root, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail={"code": "upload_staging_failed"},
            ) from error

        async def run_voice_job() -> None:
            job_id = job["id"]
            cancel_event = app.state.jobs.cancel_event(job_id)
            staging_before: set[Path] = set()
            worker_job_started = False
            profile_existed_before = False
            try:
                if not await app.state.jobs.update(job_id, "creating_voice", total=1):
                    await app.state.jobs.finish(job_id, "cancelled")
                    return

                if app.state.voice_worker is not None:
                    voices_dir = root / "voices"
                    slug = make_slug(name)
                    profile_existed_before = (voices_dir / slug).exists()
                    staging_before = set(voices_dir.glob(f".{slug}-*"))
                    worker_job_started = True
                    created = await app.state.voice_worker.create_voice(
                        upload_path=staged_upload,
                        filename=filename,
                        name=name,
                        voices_dir=voices_dir,
                        work_dir=work_root,
                        cancel_event=cancel_event,
                    )
                    result = created
                else:
                    def create_profile():
                        with staged_upload.open("rb") as staged:
                            return app.state.voice_profiles.create(
                                staged,
                                filename,
                                name,
                                root / "voices",
                                work_root,
                                cancel_event=cancel_event,
                            )

                    created = await asyncio.to_thread(create_profile)
                    result = {
                        "name": created.name,
                        "slug": created.slug,
                        "language": created.language,
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
                    logger.exception("Stimmen-Worker fuer Auftrag %s ausgefallen", job_id)
                    await app.state.jobs.finish(job_id, "failed", error=str(error))
            except VoiceProfileError as error:
                await app.state.jobs.finish(job_id, "failed", error=str(error))
            except asyncio.CancelledError:
                await app.state.jobs.finish(
                    job["id"],
                    "cancelled",
                    error="Syntheseeingabe wurde abgebrochen",
                )
                shutil.rmtree(work_root, ignore_errors=True)
                raise
            except Exception:
                logger.exception("Stimmenauftrag %s fehlgeschlagen", job_id)
                await app.state.jobs.finish(
                    job_id,
                    "failed",
                    error="Interner Fehler bei der Stimmerstellung",
                )
            finally:
                staged_upload.unlink(missing_ok=True)
                shutil.rmtree(work_root, ignore_errors=True)
                if app.state.voice_worker is not None:
                    slug = make_slug(name)
                    voices_dir = root / "voices"
                    if worker_job_started and not profile_existed_before:
                        final_job = await app.state.jobs.get(job_id)
                        if final_job is not None and final_job["state"] != "completed":
                            shutil.rmtree(voices_dir / slug, ignore_errors=True)
                    if voices_dir.is_dir():
                        for staging in voices_dir.glob(f".{slug}-*"):
                            if staging not in staging_before and staging.is_dir():
                                shutil.rmtree(staging, ignore_errors=True)

        task = asyncio.create_task(run_voice_job())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)
        return {"id": job["id"], "state": job["state"]}

    return accept_voice_upload
