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
                gr.HTML(OUTPUT_LIBRARY_HTML, padding=False)
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
        ui.load(fn=None, js=OUTPUT_LIBRARY_REFRESH_JS)
        timer.tick(fn=None, js=OUTPUT_LIBRARY_REFRESH_JS)
    ui.queue(api_open=False)
    ui.run_history = False
    return ui
