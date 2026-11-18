import asyncio
from collections.abc import Callable
import logging
from pathlib import Path
import shutil
from typing import Any

from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.jobs import JobSlotBusy
from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.synthesis import (
    MAX_TEXT_BYTES,
    MAX_SYNTHESIS_FILES,
    SynthesisError,
    _chunks_with_pauses,
    synthesize_file,
)
from hoerbuchgenerator.voice_profiles import QWEN_LANGUAGES
from hoerbuchgenerator.worker import VoiceWorkerCancelled, VoiceWorkerError
from hoerbuchgenerator.worker_synthesizer import WorkerSynthesizer


logger = logging.getLogger(__name__)


def create_synthesis_job_handler(
    app: FastAPI,
    root: Path,
    voice_service_available: Callable[[], bool],
):
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
        if not items or len(items) > MAX_SYNTHESIS_FILES:
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
            synthesizer = WorkerSynthesizer(
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
                            publish=app.state.outputs.publish,
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

    return accept_synthesis
