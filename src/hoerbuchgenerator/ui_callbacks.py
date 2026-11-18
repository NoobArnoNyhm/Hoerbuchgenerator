import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.synthesis import MAX_SYNTHESIS_FILES, MAX_TEXT_BYTES
from hoerbuchgenerator.ui_status import create_status_callbacks


logger = logging.getLogger(__name__)


def create_ui_callbacks(
    app: FastAPI,
    root: Path,
    accept_voice_upload: Callable[..., Any],
    accept_synthesis: Callable[..., Any],
    voice_service_available: Callable[[], bool],
) -> tuple[Callable[..., Any], ...]:
    async def submit_voice_ui(audio_path: str | None, name: str):
        if not audio_path:
            message = "Bitte zuerst eine WAV- oder MP3-Datei hochladen."
            return "", message, gr.skip(), gr.skip()
        if not isinstance(name, str) or not name.strip():
            message = "Bitte einen Namen fuer die Stimme eingeben."
            return "", message, gr.skip(), gr.skip()

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
            return "", message, gr.skip(), gr.skip()
        except OSError:
            logger.exception("Gradio-Stimmenupload konnte nicht gelesen werden")
            message = "Die hochgeladene Audiodatei ist nicht lesbar."
            return "", message, gr.skip(), gr.skip()
        return (
            accepted["id"],
            "Stimmenauftrag angenommen.",
            gr.update(value="Stimme erstellen", interactive=False),
            gr.update(value="Text synthetisieren", interactive=False),
        )

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
                return "", message, gr.skip(), gr.skip()

            encoded = text.encode("utf-8")
            if len(encoded) > MAX_TEXT_BYTES:
                message = "Die Texteingabe ueberschreitet das Uploadlimit."
                return "", message, gr.skip(), gr.skip()
            offset = 0

            async def read_text(size: int) -> bytes:
                nonlocal offset
                chunk = encoded[offset:offset + size]
                offset += len(chunk)
                return chunk

            items.append((output_name.strip(), read_text))
        elif uploaded_files:
            if len(uploaded_files) > MAX_SYNTHESIS_FILES:
                message = (
                    f"Es koennen maximal {MAX_SYNTHESIS_FILES} Textdateien "
                    "gleichzeitig verarbeitet werden."
                )
                return "", message, gr.skip(), gr.skip()
            total_upload_bytes = 0
            file_paths = [Path(file_path) for file_path in uploaded_files]
            for path in file_paths:
                if path.suffix.lower() != ".txt":
                    message = "Erlaubt sind nur TXT-Dateien."
                    return "", message, gr.skip(), gr.skip()
                try:
                    total_upload_bytes += path.stat().st_size
                except OSError:
                    logger.exception(
                        "Gradio-Textdatei %s konnte nicht geprueft werden",
                        path.name,
                    )
                    message = "Eine TXT-Datei konnte nicht gelesen werden."
                    return "", message, gr.skip(), gr.skip()
            if total_upload_bytes > MAX_TEXT_BYTES:
                message = "Die Textdateien ueberschreiten gemeinsam das Uploadlimit."
                return "", message, gr.skip(), gr.skip()

            for path in file_paths:
                try:
                    encoded = await asyncio.to_thread(path.read_bytes)
                except OSError:
                    logger.exception(
                        "Gradio-Textdatei %s konnte nicht gelesen werden",
                        path.name,
                    )
                    message = "Eine TXT-Datei konnte nicht gelesen werden."
                    return "", message, gr.skip(), gr.skip()
                offset = [0]

                async def read_file(
                    size: int,
                    content=encoded,
                    position=offset,
                ) -> bytes:
                    chunk = content[position[0]:position[0] + size]
                    position[0] += len(chunk)
                    return chunk

                items.append((path.stem, read_file))
        else:
            message = "Bitte Text eingeben oder mindestens eine TXT-Datei hochladen."
            return "", message, gr.skip(), gr.skip()
        if not voice:
            message = "Bitte eine Stimme auswaehlen."
            return "", message, gr.skip(), gr.skip()

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
            return "", message, gr.skip(), gr.skip()
        return (
            accepted["id"],
            "Syntheseauftrag angenommen.",
            gr.update(value="Text synthetisieren", interactive=False),
            gr.update(value="Stimme erstellen", interactive=False),
        )

    (
        refresh_voice_ui,
        cancel_voice_ui,
        load_voice_example_ui,
        stream_job_ui,
    ) = create_status_callbacks(app, root, voice_service_available)

    return (
        submit_voice_ui,
        submit_synthesis_ui,
        refresh_voice_ui,
        cancel_voice_ui,
        load_voice_example_ui,
        stream_job_ui,
    )
