import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import os
import shutil
import tempfile
from concurrent.futures import Future
from typing import Any, Callable

import gradio as gr
from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.audio import MAX_UPLOAD_BYTES
from hoerbuchgenerator.api import register_api_routes
from hoerbuchgenerator.jobs import JobManager, JobSlotBusy
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QwenTTSAdapter
from hoerbuchgenerator.synthesis import (
    MAX_TEXT_BYTES,
    SynthesisError,
    _chunks_with_pauses,
    synthesize_file,
)
from hoerbuchgenerator.voice_profiles import (
    QWEN_LANGUAGES,
    VoiceProfileError,
    VoiceProfileService,
)
from hoerbuchgenerator.whisper import WHISPER_MODELS
from hoerbuchgenerator.worker import (
    VoiceWorker,
    VoiceWorkerCancelled,
    VoiceWorkerCommandError,
    VoiceWorkerError,
)
from hoerbuchgenerator.ui import create_ui, create_ui_callbacks


logger = logging.getLogger(__name__)


class _WorkerSynthesizer:
    def __init__(self, worker: VoiceWorker, loop: asyncio.AbstractEventLoop, cancel_event):
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


def create_app(
    data_dir: Path | None = None,
    voice_profiles: VoiceProfileService | None = None,
) -> FastAPI:
    root = data_dir or Path(os.environ.get("APP_DATA_DIR", "data"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        for directory in ("models/huggingface", "models/whisper", "voices", "output", "work", "config"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        if application.state.voice_worker is None and not application.state.voice_available:
            qwen_name = os.environ.get("QWEN_MODEL", QWEN_BASE_MODELS[0])
            whisper_name = os.environ.get("WHISPER_MODEL", "base")
            qwen_config = root / "models" / "huggingface" / qwen_name / "config.json"
            whisper_config = root / "models" / "whisper" / whisper_name / "config.json"
            if qwen_config.is_file() and whisper_config.is_file():
                if qwen_name not in QWEN_BASE_MODELS:
                    raise RuntimeError(f"Nicht unterstuetztes Qwen-Base-Modell: {qwen_name}")
                if whisper_name not in WHISPER_MODELS:
                    raise RuntimeError(f"Nicht unterstuetztes Whisper-Modell: {whisper_name}")
                application.state.voice_worker = VoiceWorker(
                    {
                        "qwen_model": qwen_name,
                        "whisper_model": whisper_name,
                        "qwen_models_dir": str(root / "models" / "huggingface"),
                        "whisper_models_dir": str(root / "models" / "whisper"),
                        "whisper_device": os.environ.get("WHISPER_DEVICE", "cpu"),
                        "whisper_compute_type": os.environ.get(
                            "WHISPER_COMPUTE_TYPE",
                            "int8",
                        ),
                        "qwen_device_map": os.environ.get("QWEN_DEVICE_MAP", "cuda:0"),
                    }
                )
                await application.state.voice_worker.start()
                application.state.voice_available = True
                application.state.model_names = {
                    "qwen": qwen_name,
                    "whisper": whisper_name,
                }
        try:
            yield
        finally:
            active_job = await application.state.jobs.active()
            if active_job is not None:
                await application.state.jobs.request_cancel(active_job["id"])
            if application.state.voice_worker is not None:
                await application.state.voice_worker.close()
            if application.state.background_tasks:
                await asyncio.gather(*application.state.background_tasks)

    app = FastAPI(
        title="Hoerbuchgenerator API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.jobs = JobManager()
    app.state.voice_profiles = voice_profiles
    app.state.voice_worker = None
    app.state.voice_available = voice_profiles is not None
    app.state.background_tasks = set()
    app.state.model_names = {"qwen": None, "whisper": None}

    def voice_service_available() -> bool:
        worker = app.state.voice_worker
        return app.state.voice_available and (worker is None or worker.ready)

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

    async def accept_synthesis(
        items: list[tuple[str, Callable[[int], Any]]],
        voice: str,
        language: str | None,
        output_format: str,
    ) -> dict[str, str]:
        if not voice_service_available() or app.state.voice_worker is None:
            raise HTTPException(
                status_code=503,
                detail={"code": "voice_service_not_initialized"},
            )
        try:
            voice_slug = make_slug(voice)
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_voice"},
            ) from error
        voice_dir = root / "voices" / voice_slug
        metadata_path = voice_dir / "metadata.json"
        prompt_path = voice_dir / "voice-prompt.npz"
        if not voice_dir.is_dir() or not metadata_path.is_file() or not prompt_path.is_file():
            raise HTTPException(status_code=404, detail={"code": "voice_not_found"})
        if output_format.lower() not in {"wav", "mp3"}:
            raise HTTPException(status_code=422, detail={"code": "invalid_output_format"})
        if language is not None and language.lower() not in QWEN_LANGUAGES:
            raise HTTPException(status_code=422, detail={"code": "unsupported_language"})
        if not items or len(items) > 20:
            raise HTTPException(status_code=422, detail={"code": "invalid_file_count"})

        seen_names: set[str] = set()
        for output_name, _ in items:
            try:
                slug = make_slug(output_name)
            except ValueError as error:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_output_name"},
                ) from error
            if slug in seen_names:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "duplicate_output_name"},
                )
            seen_names.add(slug)
            extension = output_format.lower()
            if (root / "output" / f"{slug}.{extension}").exists():
                raise HTTPException(
                    status_code=409,
                    detail={"code": "output_exists"},
                )

        try:
            job = await app.state.jobs.reserve("synthesis")
        except JobSlotBusy as error:
            raise HTTPException(status_code=403, detail={"code": "busy"}) from error

        work_root = root / "work" / job["id"]
        staged: list[tuple[str, Path]] = []
        try:
            work_root.mkdir(parents=True, exist_ok=True)
            total_bytes = 0
            for index, (output_name, read_chunk) in enumerate(items):
                text_path = work_root / f"input-{index:03d}.txt"
                with text_path.open("wb") as target:
                    while chunk := await read_chunk(64 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > MAX_TEXT_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail={"code": "text_too_large"},
                            )
                        target.write(chunk)
                try:
                    text = text_path.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError as error:
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "invalid_text_encoding"},
                    ) from error
                if not text.strip():
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "empty_text"},
                    )
                staged.append((output_name, text_path))
        except Exception:
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Syntheseeingabe konnte nicht vorbereitet werden",
            )
            shutil.rmtree(work_root, ignore_errors=True)
            raise

        async def run_synthesis() -> None:
            job_id = job["id"]
            cancel_event = app.state.jobs.cancel_event(job_id)
            outputs: list[dict[str, Any]] = []
            total_chunks = sum(
                len(_chunks_with_pauses(path.read_text(encoding="utf-8-sig")))
                for _, path in staged
            )
            completed_chunks = 0
            loop = asyncio.get_running_loop()
            synthesizer = _WorkerSynthesizer(
                app.state.voice_worker,
                loop,
                cancel_event,
            )
            try:
                await app.state.jobs.update(
                    job_id,
                    "synthesizing",
                    total=total_chunks,
                )
                for output_name, text_path in staged:
                    text = text_path.read_text(encoding="utf-8-sig")

                    def report_progress(current: int) -> None:
                        future = asyncio.run_coroutine_threadsafe(
                            app.state.jobs.update(
                                job_id,
                                "synthesizing",
                                total=total_chunks,
                                completed=completed_chunks + current,
                            ),
                            loop,
                        )
                        future.result()

                    try:
                        result = await asyncio.to_thread(
                            synthesize_file,
                            synthesizer,
                            voice_dir,
                            text,
                            output_name,
                            root / "output",
                            output_format.lower(),
                            work_root,
                            language=language,
                            cancel_check=cancel_event.is_set,
                            progress=report_progress,
                        )
                        outputs.append({"state": "completed", **result})
                    except SynthesisError as error:
                        if cancel_event.is_set():
                            raise
                        logger.exception(
                            "Synthesedatei %s im Auftrag %s fehlgeschlagen",
                            output_name,
                            job_id,
                        )
                        outputs.append(
                            {"name": output_name, "state": "failed", "error": str(error)}
                        )
                    completed_chunks += len(
                        _chunks_with_pauses(text)
                    )

                if cancel_event.is_set():
                    await app.state.jobs.finish(
                        job_id,
                        "cancelled",
                        result={"files": outputs},
                    )
                elif outputs and all(item["state"] == "failed" for item in outputs):
                    await app.state.jobs.finish(
                        job_id,
                        "failed",
                        error="Alle angeforderten Dateien sind fehlgeschlagen",
                        result={"files": outputs},
                    )
                else:
                    await app.state.jobs.finish(
                        job_id,
                        "completed",
                        result={"files": outputs},
                    )
            except VoiceWorkerCancelled:
                await app.state.jobs.finish(
                    job_id,
                    "cancelled",
                    result={"files": outputs},
                )
            except VoiceWorkerError as error:
                app.state.voice_available = app.state.voice_worker.ready
                if not app.state.voice_available:
                    app.state.model_names = {"qwen": None, "whisper": None}
                if cancel_event.is_set():
                    await app.state.jobs.finish(
                        job_id,
                        "cancelled",
                        result={"files": outputs},
                    )
                else:
                    logger.exception("Synthese-Worker fuer Auftrag %s ausgefallen", job_id)
                    await app.state.jobs.finish(
                        job_id,
                        "failed",
                        error=str(error),
                        result={"files": outputs},
                    )
            except Exception:
                logger.exception("Syntheseauftrag %s fehlgeschlagen", job_id)
                await app.state.jobs.finish(
                    job_id,
                    "failed",
                    error="Interner Fehler bei der Synthese",
                    result={"files": outputs},
                )
            finally:
                shutil.rmtree(work_root, ignore_errors=True)

        task = asyncio.create_task(run_synthesis())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)
        return {"id": job["id"], "state": job["state"]}

    (
        submit_voice_ui,
        submit_synthesis_ui,
        refresh_voice_ui,
        cancel_voice_ui,
    ) = create_ui_callbacks(
        app,
        root,
        accept_voice_upload,
        accept_synthesis,
        voice_service_available,
    )

    register_api_routes(
        app,
        root,
        accept_voice_upload,
        accept_synthesis,
        voice_service_available,
    )

    ui = create_ui(
        submit_voice_ui,
        submit_synthesis_ui,
        refresh_voice_ui,
        cancel_voice_ui,
    )
    return gr.mount_gradio_app(
        app,
        ui,
        path="/ui",
        enable_monitoring=False,
        mcp_server=False,
        show_api=False,
    )


app = create_app()
