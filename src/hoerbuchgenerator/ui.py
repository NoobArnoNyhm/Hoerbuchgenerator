import asyncio
import html
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
import wave

import gradio as gr
import numpy as np
from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.synthesis import MAX_TEXT_BYTES
from hoerbuchgenerator.voice_profiles import QWEN_LANGUAGES


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

    return (
        submit_voice_ui,
        submit_synthesis_ui,
        refresh_voice_ui,
        cancel_voice_ui,
    )


def create_ui(
    submit_voice: Callable[..., Any],
    submit_synthesis: Callable[..., Any],
    refresh_voice: Callable[..., Any],
    cancel_voice: Callable[..., Any],
) -> gr.Blocks:
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
