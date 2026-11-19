import html
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI

from hoerbuchgenerator.jobs import TERMINAL_STATES
from hoerbuchgenerator.voice_catalog import VoiceCatalogError, list_voice_profiles


def _format_remaining_time(seconds: float) -> str:
    remaining = max(math.ceil(seconds), 1)
    hours, remainder = divmod(remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours} Std.")
    if minutes:
        parts.append(f"{minutes} Min.")
    if seconds or not parts:
        parts.append(f"{seconds} Sek.")
    return " ".join(parts)


def create_status_callbacks(
    app: FastAPI,
    root: Path,
    voice_service_available: Callable[[], bool],
) -> tuple[Callable[..., Any], ...]:
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

        def update_interactive_if_changed(
            key: str,
            value: bool,
            label: str,
        ) -> Any:
            if previous.get(key) == value:
                return gr.skip()
            next_ui_state[key] = value
            return gr.update(value=label, interactive=value)

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
                {
                    "voice": "Stimmerstellung",
                    "synthesis": "Sprachsynthese",
                    "transcription": "Transkription",
                }.get(active["kind"], "Auftrag")
            )
            phase = active.get("phase") or active["state"]
            phase = {
                "creating_voice": "Stimme wird erstellt",
                "synthesizing": "Sprache wird synthetisiert",
                "transcribing": "Audio wird transkribiert",
            }.get(phase, str(phase).replace("_", " "))
            progress = (
                f"{active['completed']} / {active['total']} "
                f"{'Schritte' if active['kind'] == 'voice' else 'Chunks'}"
                if active["total"]
                else (
                    "wird verarbeitet"
                    if active["kind"] == "transcription"
                    else "wird vorbereitet"
                )
            )
            status_lines.append(
                f"**Auftrag:** {kind} · {phase} · {progress}"
            )
        app_status = "  \n".join(status_lines)

        voice_status = None
        synthesis_status = None
        transcription_status = None
        transcription_text = gr.skip()
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
        elif selected is not None and selected["kind"] == "transcription":
            transcription_text = ""
            if selected["state"] == "completed":
                result = selected.get("result") or {}
                text = result.get("text")
                language = result.get("language")
                if isinstance(text, str):
                    transcription_text = text
                    language_status = (
                        f" (Sprache: `{html.escape(language)}`)"
                        if isinstance(language, str) and language
                        else ""
                    )
                    transcription_status = (
                        f"Transkription abgeschlossen{language_status}."
                    )
                else:
                    transcription_status = (
                        "Transkription fehlgeschlagen: Whisper lieferte keinen Text."
                    )
            elif selected["state"] == "failed":
                transcription_status = (
                    "Transkription fehlgeschlagen: "
                    f"{html.escape(str(selected['error']))}"
                )
            elif selected["state"] == "cancelled":
                transcription_status = "Transkription abgebrochen."
            elif selected["state"] == "cancel_requested":
                transcription_status = (
                    "Abbruch angefordert; der Worker wird beendet und neu geladen."
                )
            elif selected["state"] == "queued":
                transcription_status = "Transkription wird vorbereitet ..."
            else:
                transcription_status = "Transkription laeuft ..."

        example_audio = None
        if preview_voice in profile_names:
            example_path = root / "voices" / preview_voice / "example.wav"
            if example_path.is_file() and not example_path.is_symlink():
                example_audio = str(example_path)

        can_create = voice_service_available() and active is None
        can_synthesize = active is None
        can_transcribe = voice_service_available() and active is None

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
        create_button_update = update_interactive_if_changed(
            "can_create",
            can_create,
            "Stimme erstellen",
        )
        cancel_button_update = update_interactive_if_changed(
            "can_cancel",
            active is not None,
            "Auftrag abbrechen",
        )
        synthesis_button_update = update_interactive_if_changed(
            "can_synthesize",
            can_synthesize,
            "Text synthetisieren",
        )
        transcription_button_update = update_interactive_if_changed(
            "can_transcribe",
            can_transcribe,
            "Transkribieren",
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
        transcription_status_update = (
            update_if_changed("transcription_status", transcription_status)
            if transcription_status is not None
            else gr.skip()
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
            transcription_status_update,
            transcription_text,
            transcription_button_update,
        )

    async def cancel_voice_ui(job_id: str | None):
        if not job_id:
            message = "Kein aktiver Auftrag."
            return message, message, message
        job = await app.state.jobs.request_cancel(job_id)
        if job is None:
            message = "Auftrag nicht gefunden."
            return message, message, message
        if job["state"] in {"completed", "failed", "cancelled"}:
            message = "Der Auftrag ist bereits beendet."
            return message, message, message
        message = "Abbruch angefordert; der Worker wird beendet und neu geladen."
        return message, message, message

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

    async def stream_job_ui(
        job_id: str | None,
        preview_voice: str | None,
        synthesis_voice: str | None,
    ):
        unchanged = (gr.skip(),) * 15
        if not job_id:
            yield unchanged
            return
        queue = await app.state.jobs.subscribe(job_id)
        if queue is None:
            yield unchanged
            return
        try:
            while True:
                snapshot = await queue.get()
                refreshed = await refresh_voice_ui(
                    job_id,
                    preview_voice,
                    synthesis_voice,
                    {},
                )
                if (
                    snapshot["kind"] == "synthesis"
                    and snapshot["state"] == "running"
                    and snapshot["phase"] == "synthesizing"
                    and isinstance(refreshed[3], str)
                ):
                    estimate = snapshot.get("estimated_remaining_seconds")
                    if isinstance(estimate, (int, float)):
                        refreshed = (
                            *refreshed[:3],
                            refreshed[3]
                            + " Geschaetzte Restzeit: ca. "
                            + _format_remaining_time(estimate)
                            + ".",
                            *refreshed[4:],
                        )
                    elif snapshot["completed"] == 0:
                        refreshed = (
                            *refreshed[:3],
                            refreshed[3]
                            + " Restzeit wird nach dem ersten Chunk berechnet.",
                            *refreshed[4:],
                        )
                if snapshot["state"] in TERMINAL_STATES:
                    library_refresh = (
                        job_id
                        if snapshot["kind"] == "synthesis"
                        and snapshot["state"] == "completed"
                        else gr.skip()
                    )
                    yield (*refreshed, library_refresh)
                    return
                updates = [gr.skip()] * len(refreshed)
                for index in (1, 2, 3, 5, 6, 7, 11, 13):
                    updates[index] = refreshed[index]
                yield (*updates, gr.skip())
        finally:
            await app.state.jobs.unsubscribe(job_id, queue)

    return (
        refresh_voice_ui,
        cancel_voice_ui,
        load_voice_example_ui,
        stream_job_ui,
    )
