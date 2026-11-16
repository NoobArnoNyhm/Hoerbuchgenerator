import asyncio
import html
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException

from hoerbuchgenerator.synthesis import MAX_SYNTHESIS_FILES, MAX_TEXT_BYTES
from hoerbuchgenerator.voice_profiles import QWEN_LANGUAGES
from hoerbuchgenerator.voice_catalog import VoiceCatalogError, list_voice_profiles


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
            gr.update(interactive=False),
            gr.update(interactive=False),
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
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    async def refresh_voice_ui(
        job_id: str | None,
        preview_voice: str | None,
        synthesis_voice: str | None,
        previous_ui_state: dict[str, Any] | None,
    ):
        active = await app.state.jobs.active()
        job = await app.state.jobs.get(job_id) if job_id else None
        selected = active or job
        current_job_id = selected["id"] if selected else job_id
        models = app.state.model_names
        previous = previous_ui_state or {}
        next_ui_state = dict(previous)

        def update_if_changed(key: str, value: Any) -> Any:
            if previous.get(key) == value:
                return gr.skip()
            next_ui_state[key] = value
            return value

        def update_interactive_if_changed(key: str, value: bool) -> Any:
            if previous.get(key) == value:
                return gr.skip()
            next_ui_state[key] = value
            return gr.update(interactive=value)

        catalog_error = None
        try:
            voices = list_voice_profiles(root)
        except VoiceCatalogError as error:
            voices = []
            catalog_error = error
        profile_names = [voice["slug"] for voice in voices]
        voice_choices = [
            (f"{voice['name']} ({voice['language']})", voice["slug"])
            for voice in voices
        ]
        new_voice_slug = (
            selected["result"]["slug"]
            if selected is not None
            and selected["kind"] == "voice"
            and selected["state"] == "completed"
            and selected["result"]["slug"] in profile_names
            else profile_names[0] if profile_names else None
        )
        requested_preview_voice = preview_voice
        requested_synthesis_voice = synthesis_voice
        if preview_voice not in profile_names:
            preview_voice = new_voice_slug
        if synthesis_voice not in profile_names:
            synthesis_voice = new_voice_slug

        service_ready = bool(models["qwen"] and models["whisper"])
        status_lines = [
            f"**{'Bereit' if service_ready else 'Modelle werden geladen'}**",
            f"Qwen: `{models['qwen'] or 'nicht geladen'}`",
            f"Whisper: `{models['whisper'] or 'nicht geladen'}`",
        ]
        if active is not None:
            kind = (
                "Stimmerstellung"
                if active["kind"] == "voice"
                else "Sprachsynthese"
            )
            phase = active.get("phase") or active["state"]
            phase = {
                "creating_voice": "Stimme wird erstellt",
                "synthesizing": "Sprache wird synthetisiert",
            }.get(phase, str(phase).replace("_", " "))
            progress = (
                f"{active['completed']} / {active['total']} "
                f"{'Schritte' if active['kind'] == 'voice' else 'Chunks'}"
                if active["total"]
                else "wird vorbereitet"
            )
            status_lines.append(
                f"**Auftrag:** {kind} · {phase} · {progress}"
            )
        app_status = "  \n".join(status_lines)

        voice_status = None
        synthesis_status = None
        if catalog_error is not None:
            voice_status = (
                f"Stimmenprofil {html.escape(str(catalog_error))} ist ungueltig."
            )
        if selected is not None and selected["kind"] == "voice":
            if selected["state"] == "completed":
                result = selected["result"]
                voice_status = (
                    f"Stimme **{html.escape(result['name'])}** wurde erstellt."
                )
            elif selected["state"] == "failed":
                voice_status = (
                    "Fehler bei der Stimmerstellung: "
                    f"{html.escape(str(selected['error']))}"
                )
            elif selected["state"] == "cancelled":
                voice_status = "Stimmerstellung abgebrochen."
            elif selected["state"] == "cancel_requested":
                voice_status = (
                    "Abbruch angefordert; der Worker wird beendet und neu geladen."
                )
            elif selected["state"] == "queued":
                voice_status = "Stimme wird vorbereitet ..."
            else:
                voice_status = (
                    f"Stimmerstellung laeuft: {selected['completed']} von "
                    f"{selected['total'] or '?'} Schritten."
                )
        elif selected is not None and selected["kind"] == "synthesis":
            if selected["state"] in {"completed", "failed", "cancelled"}:
                files = (selected.get("result") or {}).get("files", [])
                links = []
                for output in files:
                    filename = html.escape(output["name"])
                    if output["state"] == "completed":
                        links.append(
                            f"[{filename}](/api/v1/output/{filename})"
                        )
                    else:
                        links.append(
                            f"{filename}: "
                            f"{html.escape(output.get('error', 'Fehlgeschlagen'))}"
                        )
                synthesis_status = (
                    "Sprachsynthese abgeschlossen."
                    if selected["state"] == "completed"
                    else "Sprachsynthese abgebrochen."
                    if selected["state"] == "cancelled"
                    else (
                        "Sprachsynthese fehlgeschlagen: "
                        f"{html.escape(str(selected['error']))}"
                    )
                )
                if links:
                    synthesis_status += "\n\nAusgaben: " + " · ".join(links)
            elif selected["state"] == "cancel_requested":
                synthesis_status = (
                    "Abbruch angefordert; der Worker wird beendet und neu geladen."
                )
            else:
                synthesis_status = (
                    f"Synthese laeuft: {selected['completed']} von "
                    f"{selected['total'] or '?'} Chunks."
                )

        example_audio = None
        if preview_voice in profile_names:
            example_path = root / "voices" / preview_voice / "example.wav"
            if example_path.is_file() and not example_path.is_symlink():
                example_audio = str(example_path)

        can_create = voice_service_available() and active is None
        can_synthesize = active is None

        def dropdown_update(
            key: str,
            requested_value: str | None,
            selected_value: str | None,
        ):
            choices_key = f"{key}_choices"
            changed = (
                previous.get(choices_key) != voice_choices
                or selected_value != requested_value
            )
            if not changed:
                return gr.skip()
            next_ui_state[choices_key] = voice_choices
            return gr.update(choices=voice_choices, value=selected_value)

        current_job_id = current_job_id or ""
        job_id_update = update_if_changed("job_id", current_job_id)
        app_status_update = update_if_changed("app_status", app_status)
        voice_status_update = (
            update_if_changed("voice_status", voice_status)
            if voice_status is not None
            else gr.skip()
        )
        synthesis_status_update = (
            update_if_changed("synthesis_status", synthesis_status)
            if synthesis_status is not None
            else gr.skip()
        )
        example_audio_update = update_if_changed("example_audio", example_audio)
        create_button_update = update_interactive_if_changed("can_create", can_create)
        cancel_button_update = update_interactive_if_changed(
            "can_cancel",
            active is not None,
        )
        synthesis_button_update = update_interactive_if_changed(
            "can_synthesize",
            can_synthesize,
        )
        preview_dropdown_update = dropdown_update(
            "preview_voice",
            requested_preview_voice,
            preview_voice,
        )
        synthesis_dropdown_update = dropdown_update(
            "synthesis_voice",
            requested_synthesis_voice,
            synthesis_voice,
        )
        snapshot_update = next_ui_state if next_ui_state != previous else gr.skip()
        return (
            job_id_update,
            app_status_update,
            voice_status_update,
            synthesis_status_update,
            example_audio_update,
            create_button_update,
            cancel_button_update,
            synthesis_button_update,
            preview_dropdown_update,
            synthesis_dropdown_update,
            snapshot_update,
        )

    async def cancel_voice_ui(job_id: str | None):
        if not job_id:
            message = "Kein aktiver Auftrag."
            return message, message
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            message = "Auftrag nicht gefunden."
            return message, message
        if job["state"] in {"completed", "failed", "cancelled"}:
            message = "Der Auftrag ist bereits beendet."
            return message, message
        message = "Abbruch angefordert; der Worker wird beendet und neu geladen."
        return message, message

    def load_voice_example_ui(selected_voice: str | None) -> str | None:
        if not selected_voice:
            return None
        try:
            voices = list_voice_profiles(root)
        except VoiceCatalogError:
            return None
        voice = next(
            (voice for voice in voices if voice["slug"] == selected_voice),
            None,
        )
        if voice is None or voice["example_audio_url"] is None:
            return None
        return str(root / "voices" / selected_voice / "example.wav")

    return (
        submit_voice_ui,
        submit_synthesis_ui,
        refresh_voice_ui,
        cancel_voice_ui,
        load_voice_example_ui,
    )


def create_ui(
    submit_voice: Callable[..., Any],
    submit_synthesis: Callable[..., Any],
    refresh_voice: Callable[..., Any],
    cancel_voice: Callable[..., Any],
    load_voice_example: Callable[..., Any],
) -> gr.Blocks:
    with gr.Blocks(title="Hoerbuchgenerator", analytics_enabled=False) as ui:
        gr.Markdown("# Hoerbuchgenerator")
        with gr.Row():
            status = gr.Markdown("Modelle und Auftragsstatus werden geladen ...")
            with gr.Column(scale=0, min_width=180):
                cancel_button = gr.Button("Auftrag abbrechen", interactive=False)
        job_id = gr.State("")
        ui_state = gr.State({})
        with gr.Tabs():
            with gr.Tab("Stimmsynthese"):
                gr.Markdown("### Neue Stimme erstellen")
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
                create_button = gr.Button("Stimme erstellen", variant="primary")
                voice_status = gr.Markdown("Noch kein Stimmsyntheseauftrag.")
                gr.Markdown("### Vorhandene Stimmen")
                preview_voice = gr.Dropdown(
                    label="Vorhandene Stimme",
                    choices=[],
                    value=None,
                )
                example = gr.Audio(
                    label="Beispielaudio",
                    type="filepath",
                    interactive=False,
                )
            with gr.Tab("Sprachsynthese"):
                gr.Markdown("### Text oder Dateien synthetisieren")
                synthesis_text = gr.Textbox(label="Text", lines=8)
                synthesis_name = gr.Textbox(
                    label="Ausgabename (bei direkter Texteingabe)"
                )
                synthesis_files = gr.File(
                    label="TXT-Dateien (optional)",
                    file_types=[".txt"],
                    file_count="multiple",
                    type="filepath",
                )
                with gr.Row():
                    synthesis_voice = gr.Dropdown(
                        label="Stimme",
                        choices=[],
                        value=None,
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
                synthesis_button = gr.Button(
                    "Text synthetisieren",
                    variant="primary",
                )
                synthesis_status = gr.Markdown("Noch kein Sprachsyntheseauftrag.")
        timer = gr.Timer(2)
        create_button.click(
            submit_voice,
            inputs=[voice_audio, voice_name],
            outputs=[job_id, voice_status, create_button, synthesis_button],
        )
        cancel_button.click(
            cancel_voice,
            inputs=[job_id],
            outputs=[voice_status, synthesis_status],
        )
        timer.tick(
            refresh_voice,
            inputs=[job_id, preview_voice, synthesis_voice, ui_state],
            outputs=[
                job_id,
                status,
                voice_status,
                synthesis_status,
                example,
                create_button,
                cancel_button,
                synthesis_button,
                preview_voice,
                synthesis_voice,
                ui_state,
            ],
        )
        ui.load(
            refresh_voice,
            inputs=[job_id, preview_voice, synthesis_voice, ui_state],
            outputs=[
                job_id,
                status,
                voice_status,
                synthesis_status,
                example,
                create_button,
                cancel_button,
                synthesis_button,
                preview_voice,
                synthesis_voice,
                ui_state,
            ],
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
            outputs=[
                job_id,
                synthesis_status,
                synthesis_button,
                create_button,
            ],
        )
        preview_voice.change(
            load_voice_example,
            inputs=[preview_voice],
            outputs=[example],
        )
    ui.queue(api_open=False)
    ui.run_history = False
    return ui
