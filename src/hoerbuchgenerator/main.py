from contextlib import asynccontextmanager
from pathlib import Path
import os

import gradio as gr
from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.jobs import JobManager


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Hoerbuchgenerator", analytics_enabled=False) as ui:
        gr.Markdown("# Hoerbuchgenerator")
        gr.Markdown("Der lokale Modell- und Auftragsdienst wird vorbereitet.")
    ui.queue(api_open=False)
    ui.run_history = False
    return ui


def create_app(data_dir: Path | None = None) -> FastAPI:
    root = data_dir or Path(os.environ.get("APP_DATA_DIR", "data"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for directory in ("models/huggingface", "models/whisper", "voices", "output", "work", "config"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(
        title="Hoerbuchgenerator API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.jobs = JobManager()

    @app.get("/api/v1/status", tags=["status"])
    async def get_status() -> dict[str, object]:
        return {
            "state": "not_initialized",
            "active_job": await app.state.jobs.active(),
            "models": {"qwen": None, "whisper": None},
        }

    @app.get("/api/v1/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str) -> dict[str, object]:
        job = await app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found"})
        return job

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
