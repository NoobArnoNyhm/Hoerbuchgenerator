from collections.abc import Callable
from typing import Any

import gradio as gr

from hoerbuchgenerator.ui_callbacks import create_ui_callbacks as create_ui_callbacks
from hoerbuchgenerator.ui_outputs import OUTPUT_LIBRARY_HTML, OUTPUT_LIBRARY_REFRESH_JS
from hoerbuchgenerator.voice_profiles import QWEN_LANGUAGES


def create_ui(
    submit_voice: Callable[..., Any],
    submit_synthesis: Callable[..., Any],
    refresh_voice: Callable[..., Any],
    cancel_voice: Callable[..., Any],
    load_voice_example: Callable[..., Any],
    stream_job: Callable[..., Any],
    submit_transcription: Callable[..., Any],
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
                gr.Markdown(
                    "**Achtung: Keine fremden Stimmen ohne Einwilligung verwenden!**\n\n"
                    "Das Klonen und Verwenden einer fremden Stimme ohne Einwilligung "
                    "kann illegal sein und Persoenlichkeitsrechte verletzen. "
                    "Die Nutzung gegen den Willen der betroffenen Person ist "
                    "moralisch hoechst verwerflich. Verwende ausschliesslich deine "
                    "eigene Stimme oder Stimmen von Personen, die dem Klonen und "
                    "der beabsichtigten Nutzung ausdruecklich zugestimmt haben. "
                    "Oeffentlich zugaengliche Aufnahmen sind keine Einwilligung."
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
            with gr.Tab("Sprachsynthese", render_children=True):
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
                gr.HTML(OUTPUT_LIBRARY_HTML, padding=False)
            with gr.Tab("Audio zu Text"):
                gr.Markdown("### Audio mit Whisper transkribieren")
                transcription_audio = gr.File(
                    label="Audiodatei (WAV oder MP3)",
                    file_types=[".wav", ".mp3"],
                    type="filepath",
                )
                transcription_button = gr.Button(
                    "Transkribieren",
                    variant="primary",
                )
                transcription_status = gr.Markdown(
                    "Noch kein Transkriptionsauftrag."
                )
                transcription_text = gr.Textbox(
                    label="Erkannter Text",
                    lines=12,
                )
        output_refresh_signal = gr.Textbox(value="", visible=False)
        status_outputs = [
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
            transcription_status,
            transcription_text,
            transcription_button,
        ]
        stream_outputs = [*status_outputs, output_refresh_signal]
        stream_inputs = [job_id, preview_voice, synthesis_voice]

        create_event = create_button.click(
            submit_voice,
            inputs=[voice_audio, voice_name],
            outputs=[
                job_id,
                voice_status,
                create_button,
                synthesis_button,
                transcription_button,
            ],
        )
        create_event.then(
            stream_job,
            inputs=stream_inputs,
            outputs=stream_outputs,
        )
        cancel_button.click(
            cancel_voice,
            inputs=[job_id],
            outputs=[
                voice_status,
                synthesis_status,
                transcription_status,
            ],
        )
        load_event = ui.load(
            refresh_voice,
            inputs=[job_id, preview_voice, synthesis_voice, ui_state],
            outputs=status_outputs,
        )
        load_event.then(
            stream_job,
            inputs=stream_inputs,
            outputs=stream_outputs,
        )
        synthesis_event = synthesis_button.click(
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
                transcription_button,
            ],
        )
        synthesis_event.then(
            stream_job,
            inputs=stream_inputs,
            outputs=stream_outputs,
        )
        transcription_event = transcription_button.click(
            submit_transcription,
            inputs=[transcription_audio],
            outputs=[
                job_id,
                transcription_status,
                transcription_text,
                transcription_button,
                create_button,
                synthesis_button,
            ],
        )
        transcription_event.then(
            stream_job,
            inputs=stream_inputs,
            outputs=stream_outputs,
        )
        preview_voice.change(
            load_voice_example,
            inputs=[preview_voice],
            outputs=[example],
        )
        ui.load(fn=None, js=OUTPUT_LIBRARY_REFRESH_JS)
        output_refresh_signal.change(fn=None, js=OUTPUT_LIBRARY_REFRESH_JS)
    ui.queue(api_open=False)
    ui.run_history = False
    return ui
