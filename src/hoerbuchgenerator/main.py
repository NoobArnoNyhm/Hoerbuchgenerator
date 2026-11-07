import asyncio
from contextlib import asynccontextmanager
import html
import logging
from pathlib import Path
import os
import tempfile
import wave

import gradio as gr
import numpy as np
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

    async def accept_voice_upload(name: str, filename: str, read_chunk) -> dict[str, str]:
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
        return {"id": job["id"], "state": job["state"]}

    async def submit_voice_ui(audio_path: str | None, name: str):
        if not audio_path:
            message = "Bitte zuerst eine WAV- oder MP3-Datei hochladen."
            return "", message, True, message
        if not isinstance(name, str) or not name.strip():
            message = "Bitte einen Namen fuer die Stimme eingeben."
            return "", message, True, message

        try:
            with Path(audio_path).open("rb") as audio_file:
                async def read_chunk(size: int) -> bytes:
                    return await asyncio.to_thread(audio_file.read, size)

                accepted = await accept_voice_upload(
                    name,
                    Path(audio_path).name,
                    read_chunk,
                )
        except HTTPException as error:
            detail = error.detail
            message = detail.get("message", detail.get("code", "Unbekannter Fehler"))
            message = f"Stimme konnte nicht gestartet werden: {message}"
            return "", message, True, message
        except OSError:
            logger.exception("Gradio-Stimmenupload konnte nicht gelesen werden")
            message = "Die hochgeladene Audiodatei ist nicht lesbar."
            return "", message, True, message
        return accepted["id"], "Stimmenauftrag angenommen.", False, ""

    async def refresh_voice_ui(job_id: str | None, notice: str | None):
        active = await app.state.jobs.active()
        job = await app.state.jobs.get(job_id) if job_id else None
        selected = active or job
        current_job_id = selected["id"] if selected else job_id
        models = app.state.model_names
        model_status = (
            f"Qwen: `{models['qwen'] or 'nicht geladen'}` · "
            f"Whisper: `{models['whisper'] or 'nicht geladen'}`"
        )
        profile_names = sorted(
            path.name for path in (root / "voices").iterdir() if path.is_dir()
        ) if (root / "voices").is_dir() else []
        voices_status = (
            "Vorhandene Stimmen: " + ", ".join(profile_names)
            if profile_names
            else "Noch keine Stimmenprofile."
        )
        example_audio = None
        if selected is None:
            status = notice or "Kein aktiver Auftrag."
        elif selected["state"] == "completed":
            result = selected["result"]
            status = f"Stimme **{html.escape(result['name'])}** wurde erstellt."
            example_path = root / "voices" / result["slug"] / "example.wav"
            if example_path.is_file():
                try:
                    with wave.open(str(example_path), "rb") as example_file:
                        example_audio = (
                            example_file.getframerate(),
                            np.frombuffer(
                                example_file.readframes(example_file.getnframes()),
                                dtype="<i2",
                            ),
                        )
                except (OSError, wave.Error):
                    logger.exception(
                        "Beispielaudio fuer Stimme %s konnte nicht geladen werden",
                        result["slug"],
                    )
                    status += "\n\nDas Beispielaudio konnte nicht geladen werden."
        elif selected["state"] == "failed":
            status = (
                "Fehler bei der Stimmerstellung: "
                f"{html.escape(str(selected['error']))}"
            )
        elif selected["state"] == "cancelled":
            status = "Stimmerstellung abgebrochen."
        elif selected["state"] == "cancel_requested":
            status = "Abbruch angefordert; laufende Modellinferenz wird beendet."
        else:
            status = "Stimme wird vorbereitet und synthetisiert ..."

        can_create = app.state.voice_profiles is not None and active is None
        return (
            current_job_id or "",
            f"{model_status}\n\n{voices_status}\n\n{status}",
            example_audio,
            gr.update(interactive=can_create),
            gr.update(interactive=active is not None),
            "" if selected else notice or "",
        )

    async def cancel_voice_ui(job_id: str | None):
        if not job_id:
            return "Kein aktiver Stimmenauftrag."
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            return "Stimmenauftrag nicht gefunden."
        if job["state"] in {"completed", "failed", "cancelled"}:
            return "Der Stimmenauftrag ist bereits beendet."
        return "Abbruch angefordert; laufende Modellinferenz wird danach beendet."

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
        async def read_chunk(size: int) -> bytes:
            return await audio.read(size)

        try:
            accepted = await accept_voice_upload(
                name,
                audio.filename or "",
                read_chunk,
            )
        finally:
            await audio.close()
        return JSONResponse(
            status_code=202,
            content=accepted,
        )

    @app.post("/api/v1/jobs/{job_id}/cancel", tags=["jobs"])
    async def cancel_job(job_id: str) -> dict[str, object]:
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found"})
        if job["state"] in {"completed", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail={"code": "job_not_active"})
        return job

    ui = create_ui(app, submit_voice_ui, refresh_voice_ui, cancel_voice_ui)
    return gr.mount_gradio_app(
        app,
        ui,
        path="/ui",
        enable_monitoring=False,
        mcp_server=False,
        show_api=False,
    )


def create_ui(app, submit_voice, refresh_voice, cancel_voice) -> gr.Blocks:
    with gr.Blocks(title="Hoerbuchgenerator", analytics_enabled=False) as ui:
        gr.Markdown("# Hoerbuchgenerator")
        gr.Markdown("### Stimme erstellen")
        with gr.Row():
            voice_name = gr.Textbox(
                label="Stimmenname",
                placeholder="Zum Beispiel: Meine Stimme",
                max_length=120,
            )
            voice_audio = gr.File(
                label="Referenzaufnahme (WAV oder MP3)",
                file_types=[".wav", ".mp3"],
                type="filepath",
            )
        with gr.Row():
            create_button = gr.Button("Stimme erstellen", variant="primary")
            cancel_button = gr.Button("Auftrag abbrechen", interactive=False)
        job_id = gr.State("")
        notice = gr.State("")
        status = gr.Markdown("Modelle und Stimmenstatus werden geladen ...")
        example = gr.Audio(
            label="Sprachbeispiel",
            type="filepath",
            interactive=False,
        )
        timer = gr.Timer(2)
        create_button.click(
            submit_voice,
            inputs=[voice_audio, voice_name],
            outputs=[job_id, status, create_button, notice],
        )
        cancel_button.click(
            cancel_voice,
            inputs=[job_id],
            outputs=[status],
        )
        timer.tick(
            refresh_voice,
            inputs=[job_id, notice],
            outputs=[job_id, status, example, create_button, cancel_button, notice],
        )
        ui.load(
            refresh_voice,
            inputs=[job_id, notice],
            outputs=[job_id, status, example, create_button, cancel_button, notice],
        )
    ui.queue(api_open=False)
    ui.run_history = False
    return ui


app = create_app()
