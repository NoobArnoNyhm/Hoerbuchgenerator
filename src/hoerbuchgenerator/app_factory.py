import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from hoerbuchgenerator.api import register_api_routes
from hoerbuchgenerator.app_lifecycle import create_lifespan
from hoerbuchgenerator.jobs import JobManager
from hoerbuchgenerator.synthesis_jobs import create_synthesis_job_handler
from hoerbuchgenerator.ui import create_ui, create_ui_callbacks
from hoerbuchgenerator.voice_jobs import create_voice_job_handler
from hoerbuchgenerator.voice_profiles import VoiceProfileService


def create_app(
    data_dir: Path | None = None,
    voice_profiles: VoiceProfileService | None = None,
) -> FastAPI:
    root = data_dir or Path(os.environ.get("APP_DATA_DIR", "data"))
    app = FastAPI(
        title="Hoerbuchgenerator API",
        version="0.1.0",
        lifespan=create_lifespan(root),
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

    accept_voice_upload = create_voice_job_handler(app, root, voice_service_available)
    accept_synthesis = create_synthesis_job_handler(app, root, voice_service_available)

    (
        submit_voice_ui,
        submit_synthesis_ui,
        refresh_voice_ui,
        cancel_voice_ui,
        load_voice_example_ui,
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
        load_voice_example_ui,
    )
    return gr.mount_gradio_app(
        app,
        ui,
        path="/ui",
        enable_monitoring=False,
        mcp_server=False,
        show_api=False,
    )
