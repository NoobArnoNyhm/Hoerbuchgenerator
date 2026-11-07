import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse


def register_api_routes(
    app: FastAPI,
    root: Path,
    accept_voice_upload: Callable[..., Awaitable[dict[str, str]]],
    accept_synthesis: Callable[..., Awaitable[dict[str, str]]],
    voice_service_available: Callable[[], bool],
) -> None:
    @app.get("/api/v1/status", tags=["status"])
    async def get_status() -> dict[str, object]:
        return {
            "state": "ready" if voice_service_available() else "not_initialized",
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
        return JSONResponse(status_code=202, content=accepted)

    @app.post("/api/v1/synthesis/text", status_code=202, tags=["synthesis"])
    async def synthesize_text(
        text: str = Form(...),
        name: str = Form(...),
        voice: str = Form(...),
        language: str | None = Form(default=None),
        output_format: str = Form(default="mp3", alias="format"),
    ) -> JSONResponse:
        encoded = text.encode("utf-8")
        offset = 0

        async def read_chunk(size: int) -> bytes:
            nonlocal offset
            chunk = encoded[offset:offset + size]
            offset += len(chunk)
            return chunk

        accepted = await accept_synthesis(
            [(name, read_chunk)],
            voice,
            language,
            output_format,
        )
        return JSONResponse(status_code=202, content=accepted)

    @app.post("/api/v1/synthesis/files", status_code=202, tags=["synthesis"])
    async def synthesize_files(
        files: list[UploadFile] = File(...),
        voice: str = Form(...),
        language: str | None = Form(default=None),
        output_format: str = Form(default="mp3", alias="format"),
    ) -> JSONResponse:
        items: list[tuple[str, Callable[[int], Any]]] = []
        try:
            for upload in files:
                filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
                if Path(filename).suffix.lower() != ".txt":
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "unsupported_text_format"},
                    )

                async def read_chunk(size: int, current=upload) -> bytes:
                    return await current.read(size)

                items.append((Path(filename).stem, read_chunk))
            accepted = await accept_synthesis(
                items,
                voice,
                language,
                output_format,
            )
        finally:
            for upload in files:
                await upload.close()
        return JSONResponse(status_code=202, content=accepted)

    @app.get("/api/v1/output/{name}", tags=["output"])
    async def download_output(name: str) -> FileResponse:
        if (
            Path(name).name != name
            or "\\" in name
            or re.fullmatch(
                r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\.(?:wav|mp3)",
                name,
            ) is None
        ):
            raise HTTPException(status_code=404, detail={"code": "output_not_found"})
        output_path = root / "output" / name
        if not output_path.is_file():
            raise HTTPException(status_code=404, detail={"code": "output_not_found"})
        media_type = "audio/wav" if name.endswith(".wav") else "audio/mpeg"
        return FileResponse(output_path, media_type=media_type, filename=name)

    @app.post("/api/v1/jobs/{job_id}/cancel", tags=["jobs"])
    async def cancel_job(job_id: str) -> dict[str, object]:
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found"})
        if job["state"] in {"completed", "failed", "cancelled"}:
            raise HTTPException(status_code=409, detail={"code": "job_not_active"})
        return job
