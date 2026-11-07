import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import os
import tempfile

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from hoerbuchgenerator.audio import MAX_UPLOAD_BYTES
from hoerbuchgenerator.jobs import JobManager, JobSlotBusy
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QwenTTSAdapter
from hoerbuchgenerator.voice_profiles import (
    VoiceCreationCancelled,
    VoiceProfileError,
    VoiceProfileService,
)
from hoerbuchgenerator.voice_reference import VoiceReferenceService
from hoerbuchgenerator.whisper import WHISPER_MODELS, FasterWhisperAdapter


logger = logging.getLogger(__name__)


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Hoerbuchgenerator", analytics_enabled=False) as ui:
        gr.Markdown("# Hoerbuchgenerator")
        gr.Markdown("Der lokale Modell- und Auftragsdienst wird vorbereitet.")
    ui.queue(api_open=False)
    ui.run_history = False
    return ui


def create_app(
    data_dir: Path | None = None,
    voice_profiles: VoiceProfileService | None = None,
) -> FastAPI:
    root = data_dir or Path(os.environ.get("APP_DATA_DIR", "data"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        for directory in ("models/huggingface", "models/whisper", "voices", "output", "work", "config"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        if application.state.voice_profiles is None:
            qwen_name = os.environ.get("QWEN_MODEL", QWEN_BASE_MODELS[0])
            whisper_name = os.environ.get("WHISPER_MODEL", "base")
            qwen_config = root / "models" / "huggingface" / qwen_name / "config.json"
            whisper_config = root / "models" / "whisper" / whisper_name / "config.json"
            if qwen_config.is_file() and whisper_config.is_file():
                if qwen_name not in QWEN_BASE_MODELS:
                    raise RuntimeError(f"Nicht unterstuetztes Qwen-Base-Modell: {qwen_name}")
                if whisper_name not in WHISPER_MODELS:
                    raise RuntimeError(f"Nicht unterstuetztes Whisper-Modell: {whisper_name}")

                def load_models() -> VoiceProfileService:
                    import torch

                    whisper = FasterWhisperAdapter.load_local(
                        whisper_name,
                        root / "models" / "whisper",
                        device=os.environ.get("WHISPER_DEVICE", "cpu"),
                        compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
                    )
                    qwen = QwenTTSAdapter.load_local(
                        qwen_name,
                        root / "models" / "huggingface",
                        loader_kwargs={
                            "device_map": os.environ.get("QWEN_DEVICE_MAP", "cuda:0"),
                            "dtype": torch.bfloat16,
                            "attn_implementation": "sdpa",
                        },
                    )
                    return VoiceProfileService(
                        VoiceReferenceService(whisper),
                        qwen,
                        qwen_name,
                    )

                application.state.voice_profiles = await asyncio.to_thread(load_models)
                application.state.model_names = {
                    "qwen": qwen_name,
                    "whisper": whisper_name,
                }
        yield
        if application.state.background_tasks:
            await asyncio.gather(*application.state.background_tasks)

    app = FastAPI(
        title="Hoerbuchgenerator API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.jobs = JobManager()
    app.state.voice_profiles = voice_profiles
    app.state.background_tasks = set()
    app.state.model_names = {"qwen": None, "whisper": None}

    @app.get("/api/v1/status", tags=["status"])
    async def get_status() -> dict[str, object]:
        return {
            "state": "ready" if app.state.voice_profiles is not None else "not_initialized",
            "active_job": await app.state.jobs.active(),
            "models": app.state.model_names,
        }

    @app.get("/api/v1/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str) -> dict[str, object]:
        job = await app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found"})
        return job

    @app.post("/api/v1/voices", status_code=202, tags=["voices"])
    async def create_voice(
        name: str = Form(...),
        audio: UploadFile = File(...),
    ) -> JSONResponse:
        if app.state.voice_profiles is None:
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

        filename = audio.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in {".wav", ".mp3"}:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsupported_audio_format"},
            )

        try:
            job = await app.state.jobs.reserve("voice")
        except JobSlotBusy as error:
            await audio.close()
            raise HTTPException(
                status_code=403,
                detail={"code": "busy"},
            ) from error

        work_root = root / "work"
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
                while chunk := await audio.read(64 * 1024):
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
            raise
        except asyncio.CancelledError:
            await app.state.jobs.finish(
                job["id"],
                "cancelled",
                error="Stimmen-Upload wurde abgebrochen",
            )
            if staged_upload is not None:
                staged_upload.unlink(missing_ok=True)
            raise
        except OSError as error:
            await app.state.jobs.finish(
                job["id"],
                "failed",
                error="Stimmen-Upload konnte nicht gespeichert werden",
            )
            if staged_upload is not None:
                staged_upload.unlink(missing_ok=True)
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
            raise HTTPException(
                status_code=500,
                detail={"code": "upload_staging_failed"},
            ) from error
        finally:
            await audio.close()

        async def run_voice_job() -> None:
            job_id = job["id"]
            cancel_event = app.state.jobs.cancel_event(job_id)
            try:
                if not await app.state.jobs.update(job_id, "creating_voice", total=1):
                    await app.state.jobs.finish(job_id, "cancelled")
                    return

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
                await app.state.jobs.finish(
                    job_id,
                    "completed",
                    result={
                        "name": created.name,
                        "slug": created.slug,
                        "language": created.language,
                    },
                )
            except VoiceCreationCancelled:
                await app.state.jobs.finish(job_id, "cancelled")
            except VoiceProfileError as error:
                await app.state.jobs.finish(job_id, "failed", error=str(error))
            except Exception:
                logger.exception("Stimmenauftrag %s fehlgeschlagen", job_id)
                await app.state.jobs.finish(
                    job_id,
                    "failed",
                    error="Interner Fehler bei der Stimmerstellung",
                )
            finally:
                staged_upload.unlink(missing_ok=True)

        task = asyncio.create_task(run_voice_job())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)
        return JSONResponse(
            status_code=202,
            content={"id": job["id"], "state": job["state"]},
        )

    @app.post("/api/v1/jobs/{job_id}/cancel", tags=["jobs"])
    async def cancel_job(job_id: str) -> dict[str, object]:
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found"})
        if job["state"] in {"completed", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail={"code": "job_not_active"})
        return job

    return gr.mount_gradio_app(
        app,
        create_ui(),
        path="/ui",
        enable_monitoring=False,
        mcp_server=False,
        show_api=False,
    )


app = create_app()
