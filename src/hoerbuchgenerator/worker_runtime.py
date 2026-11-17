import logging
from multiprocessing.connection import Connection
from pathlib import Path


logger = logging.getLogger(__name__)


def _create_voice_service(settings: dict[str, str]):
    import torch

    from hoerbuchgenerator.qwen import QwenTTSAdapter
    from hoerbuchgenerator.voice_profiles import VoiceProfileService
    from hoerbuchgenerator.voice_reference import VoiceReferenceService
    from hoerbuchgenerator.whisper import FasterWhisperAdapter

    whisper = FasterWhisperAdapter.load_local(
        settings["whisper_model"],
        Path(settings["whisper_models_dir"]),
        device=settings["whisper_device"],
        compute_type=settings["whisper_compute_type"],
    )
    qwen = QwenTTSAdapter.load_local(
        settings["qwen_model"],
        Path(settings["qwen_models_dir"]),
        loader_kwargs={
            "device_map": settings["qwen_device_map"],
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )
    return VoiceProfileService(
        VoiceReferenceService(whisper),
        qwen,
        settings["qwen_model"],
    )


def _run_voice_worker(connection: Connection, settings: dict[str, str]) -> None:
    try:
        service = _create_voice_service(settings)
        connection.send(
            {
                "type": "ready",
                "models": {
                    "qwen": settings["qwen_model"],
                    "whisper": settings["whisper_model"],
                },
            }
        )
    except Exception as error:
        logger.exception("Lokale Modelle konnten im Stimmen-Worker nicht geladen werden")
        connection.send({"type": "startup_failed", "error": str(error)})
        connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message.get("type") == "stop":
            return
        if message.get("type") != "execute":
            continue

        command = message["command"]
        if command.get("action") not in {"create_voice", "synthesize_chunk"}:
            connection.send({"type": "failed", "error": "Unbekannter Worker-Auftrag"})
            continue
        try:
            if command["action"] == "synthesize_chunk":
                audio, sample_rate = service.synthesize(
                    command["text"],
                    command["language"],
                    Path(command["voice_dir"]) / "voice-prompt.npz",
                )
                from hoerbuchgenerator.synthesis import _write_pcm

                _write_pcm(Path(command["destination"]), audio, sample_rate)
                connection.send({"type": "completed", "result": {"sample_rate": sample_rate}})
                continue

            with Path(command["upload_path"]).open("rb") as upload:
                created = service.create(
                    upload,
                    command["filename"],
                    command["name"],
                    Path(command["voices_dir"]),
                    Path(command["work_dir"]),
                )
            connection.send(
                {
                    "type": "completed",
                    "result": {
                        "name": created.name,
                        "slug": created.slug,
                        "language": created.language,
                    },
                }
            )
        except Exception as error:
            logger.exception("Stimmenauftrag im Worker fehlgeschlagen")
            connection.send(
                {
                    "type": "failed",
                    "error": str(error) if isinstance(error, RuntimeError) else (
                        "Interner Fehler bei der Stimmerstellung"
                    ),
                }
            )
