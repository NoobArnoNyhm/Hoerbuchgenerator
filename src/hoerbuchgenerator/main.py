import asyncio
from contextlib import asynccontextmanager
import html
import logging
from pathlib import Path
import os
import re
import shutil
import tempfile
import wave
from concurrent.futures import Future
from typing import Any, Callable

import gradio as gr
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from hoerbuchgenerator.audio import MAX_UPLOAD_BYTES
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

    async def submit_synthesis_ui(
        text: str | None,
        output_name: str | None,
        uploaded_files: list[str] | None,
        voice: str | None,
        language: str | None,
        output_format: str,
    ):
        items: list[tuple[str, Callable[[int], Any]]] = []
        if text and text.strip():
            if not output_name or not output_name.strip():
                message = "Bitte einen Ausgabenamen fuer den Text eingeben."
                return "", message, True, message

            encoded = text.encode("utf-8")
            offset = 0

            async def read_text(size: int) -> bytes:
                nonlocal offset
                chunk = encoded[offset:offset + size]
                offset += len(chunk)
                return chunk

            items.append((output_name.strip(), read_text))
        elif uploaded_files:
            total_upload_bytes = 0
            for file_path in uploaded_files:
                path = Path(file_path)
                if path.suffix.lower() != ".txt":
                    message = "Erlaubt sind nur TXT-Dateien."
                    return "", message, True, message
                total_upload_bytes += path.stat().st_size
                if total_upload_bytes > MAX_TEXT_BYTES:
                    message = "Die Textdatei ueberschreitet das Uploadlimit."
                    return "", message, True, message
                encoded = await asyncio.to_thread(path.read_bytes)
                offset = [0]

                async def read_file(size: int, content=encoded) -> bytes:
                    chunk = content[offset[0]:offset[0] + size]
                    offset[0] += len(chunk)
                    return chunk

                items.append((path.stem, read_file))
        else:
            message = "Bitte Text eingeben oder mindestens eine TXT-Datei hochladen."
            return "", message, True, message
        if not voice:
            message = "Bitte den Stimmen-Slug eingeben."
            return "", message, True, message

        try:
            selected_language = (
                language
                if language and language != "Automatisch aus Stimme"
                else None
            )
            accepted = await accept_synthesis(
                items,
                voice,
                selected_language,
                output_format,
            )
        except HTTPException as error:
            detail = error.detail
            message = detail.get("message", detail.get("code", "Unbekannter Fehler"))
            message = f"Synthese konnte nicht gestartet werden: {message}"
            return "", message, True, message
        return accepted["id"], "Syntheseauftrag angenommen.", False, ""

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
        elif selected["kind"] == "synthesis" and selected["state"] in {
            "completed",
            "failed",
            "cancelled",
        }:
            files = (selected.get("result") or {}).get("files", [])
            links = []
            for output in files:
                if output["state"] == "completed":
                    filename = output["name"]
                    links.append(
                        f"[{html.escape(filename)}](/api/v1/output/{html.escape(filename)})"
                    )
                    candidate = root / "output" / filename
                    if example_audio is None and candidate.is_file():
                        example_audio = str(candidate)
                else:
                    links.append(
                        f"{html.escape(output['name'])}: "
                        f"{html.escape(output.get('error', 'Fehlgeschlagen'))}"
                    )
            status = (
                "Synthese abgeschlossen."
                if selected["state"] == "completed"
                else "Synthese abgebrochen."
                if selected["state"] == "cancelled"
                else f"Synthese fehlgeschlagen: {html.escape(str(selected['error']))}"
            )
            if links:
                status += "\n\nAusgaben: " + " · ".join(links)
            if selected["state"] == "cancelled" and not links:
                status = "Synthese abgebrochen."
        elif selected["kind"] == "synthesis":
            status = (
                f"Synthese laeuft: {selected['completed']} von "
                f"{selected['total'] or '?'} Chunks."
            )
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
            status = "Abbruch angefordert; der Worker wird beendet und neu geladen."
        else:
            status = "Stimme wird vorbereitet und synthetisiert ..."

        can_create = voice_service_available() and active is None
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
            return "Kein aktiver Auftrag."
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            return "Auftrag nicht gefunden."
        if job["state"] in {"completed", "failed", "cancelled"}:
            return "Der Auftrag ist bereits beendet."
        return "Abbruch angefordert; der Worker wird beendet und neu geladen."

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
        return JSONResponse(
            status_code=202,
            content=accepted,
        )

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

    ui = create_ui(
        app,
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


def create_ui(app, submit_voice, submit_synthesis, refresh_voice, cancel_voice) -> gr.Blocks:
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
        gr.Markdown("### Text synthetisieren")
        synthesis_text = gr.Textbox(label="Text", lines=8)
        synthesis_name = gr.Textbox(label="Ausgabename (bei direkter Texteingabe)")
        synthesis_files = gr.File(
            label="TXT-Dateien (optional)",
            file_types=[".txt"],
            file_count="multiple",
            type="filepath",
        )
        with gr.Row():
            synthesis_voice = gr.Textbox(
                label="Stimmen-Slug",
                placeholder="Zum Beispiel: meine_stimme",
            )
            synthesis_language = gr.Dropdown(
                label="Sprache",
                choices=["Automatisch aus Stimme"] + sorted(QWEN_LANGUAGES),
                value="Automatisch aus Stimme",
            )
            synthesis_format = gr.Dropdown(
                label="Ausgabeformat",
                choices=["mp3", "wav"],
                value="mp3",
            )
        synthesis_button = gr.Button("Text synthetisieren", variant="primary")
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
        synthesis_button.click(
            submit_synthesis,
            inputs=[
                synthesis_text,
                synthesis_name,
                synthesis_files,
                synthesis_voice,
                synthesis_language,
                synthesis_format,
            ],
            outputs=[job_id, status, create_button, notice],
        )
    ui.queue(api_open=False)
    ui.run_history = False
    return ui


app = create_app()
