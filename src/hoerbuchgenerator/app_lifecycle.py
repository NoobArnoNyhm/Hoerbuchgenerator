import asyncio
from contextlib import asynccontextmanager, contextmanager
import logging
import os
from pathlib import Path
import shutil

from fastapi import FastAPI

from hoerbuchgenerator.model_cache import incomplete_model_files
from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QWEN_REQUIRED_FILES
from hoerbuchgenerator.whisper import WHISPER_MODELS, WHISPER_REQUIRED_FILES
from hoerbuchgenerator.worker import VoiceWorker


logger = logging.getLogger(__name__)

_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


@contextmanager
def _offline_runtime_environment(previous: dict[str, str | None]):
    os.environ.update(_OFFLINE_ENVIRONMENT)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cleanup_orphaned_work(work_root: Path) -> None:
    try:
        if work_root.is_symlink():
            raise OSError(f"Arbeitsverzeichnis ist ein symbolischer Link: {work_root}")
        for entry in work_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
            else:
                shutil.rmtree(entry)
    except OSError as error:
        logger.exception(
            "Verwaiste Arbeitsdateien unter %s konnten nicht bereinigt werden",
            work_root,
        )
        raise RuntimeError(
            f"Verwaiste Arbeitsdateien konnten nicht bereinigt werden: {work_root}"
        ) from error


def create_lifespan(root: Path):
    previous_environment = {
        key: os.environ.get(key)
        for key in _OFFLINE_ENVIRONMENT
    }

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        with _offline_runtime_environment(previous_environment):
            for directory in ("models/huggingface", "models/whisper", "voices", "output", "work", "config"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            _cleanup_orphaned_work(root / "work")
            if application.state.voice_worker is None and not application.state.voice_available:
                qwen_name = os.environ.get("QWEN_MODEL", QWEN_BASE_MODELS[0])
                whisper_name = os.environ.get("WHISPER_MODEL", "base")
                if qwen_name not in QWEN_BASE_MODELS:
                    raise RuntimeError(
                        f"Nicht unterstuetztes Qwen-Base-Modell: {qwen_name}"
                    )
                if whisper_name not in WHISPER_MODELS:
                    raise RuntimeError(
                        f"Nicht unterstuetztes Whisper-Modell: {whisper_name}"
                    )
                qwen_path = root / "models" / "huggingface" / qwen_name
                whisper_path = root / "models" / "whisper" / whisper_name
                incomplete_qwen = incomplete_model_files(
                    qwen_path,
                    QWEN_REQUIRED_FILES,
                )
                incomplete_whisper = incomplete_model_files(
                    whisper_path,
                    WHISPER_REQUIRED_FILES,
                )
                if not incomplete_qwen and not incomplete_whisper:
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
                else:
                    logger.warning(
                        "Lokale Modell-Caches fehlen oder sind unvollstaendig; "
                        "Modellinitialisierung uebersprungen "
                        "(Qwen %s fehlend in %s; Whisper %s fehlend in %s)",
                        incomplete_qwen or "vollstaendig",
                        qwen_path,
                        incomplete_whisper or "vollstaendig",
                        whisper_path,
                    )
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

    return lifespan
