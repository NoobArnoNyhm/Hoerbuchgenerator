import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import gradio as gr
import pytest

from hoerbuchgenerator.main import create_app
from hoerbuchgenerator.worker import VoiceWorkerCancelled
from hoerbuchgenerator.voice_reference import VoiceTranscriptionError


def ui_callback(ui, name):
    return next(
        function.fn
        for function in ui.fns.values()
        if function.fn is not None and function.fn.__name__ == name
    )


class FakeTranscriptionService:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def transcribe_audio(self, audio_path: Path, work_dir: Path):
        self.calls.append((audio_path.read_bytes(), work_dir))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            text="Das vollstaendige Transkript.",
            language="de",
        )


@pytest.mark.parametrize("filename", ["recording.wav", "RECORDING.MP3"])
def test_gradio_transcription_returns_full_result_and_cleans_work(
    filename,
    tmp_path,
):
    service = FakeTranscriptionService()
    app = create_app(tmp_path, service)
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / filename
    audio_path.write_bytes(b"encoded audio")

    async def submit_and_wait():
        submitted = await ui_callback(ui, "submit_transcription_ui")(
            str(audio_path)
        )
        await asyncio.gather(*tuple(app.state.background_tasks))
        refreshed = await ui_callback(ui, "refresh_voice_ui")(
            submitted[0],
            None,
            None,
            {},
        )
        return submitted, refreshed

    submitted, refreshed = asyncio.run(submit_and_wait())

    assert submitted[1] == "Transkriptionsauftrag angenommen."
    assert submitted[2] == ""
    assert submitted[3] == gr.update(value="Transkribieren", interactive=False)
    assert service.calls == [
        (b"encoded audio", tmp_path / "work" / submitted[0])
    ]
    assert refreshed[11] == "Transkription abgeschlossen (Sprache: `de`)."
    assert refreshed[12] == "Das vollstaendige Transkript."
    assert refreshed[13]["interactive"] is True
    assert app.state.jobs._active_id is None
    assert not (tmp_path / "work").exists() or list((tmp_path / "work").iterdir()) == []
    assert not list(tmp_path.rglob("*.txt"))


def test_gradio_transcription_requires_an_upload_without_reserving_a_job(tmp_path):
    app = create_app(tmp_path, FakeTranscriptionService())
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()

    submitted = asyncio.run(
        ui_callback(ui, "submit_transcription_ui")(None)
    )

    assert submitted == (
        "",
        "Bitte zuerst eine WAV- oder MP3-Datei hochladen.",
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )
    assert app.state.jobs._active_id is None


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("recording.txt", b"audio", "Erlaubt sind nur WAV- und MP3-Dateien."),
        ("recording.wav", b"", "Die Audiodatei ist leer."),
    ],
)
def test_transcription_rejects_invalid_uploads_and_releases_slot(
    filename,
    content,
    message,
    tmp_path,
):
    service = FakeTranscriptionService()
    app = create_app(tmp_path, service)
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / filename
    audio_path.write_bytes(content)

    submitted = asyncio.run(
        ui_callback(ui, "submit_transcription_ui")(str(audio_path))
    )

    assert submitted[0] == ""
    assert submitted[1] == f"Transkription konnte nicht gestartet werden: {message}"
    assert app.state.jobs._active_id is None
    assert not (tmp_path / "work").exists() or list((tmp_path / "work").iterdir()) == []
    assert service.calls == []


def test_transcription_enforces_upload_limit_before_worker_execution(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "hoerbuchgenerator.transcription_jobs.MAX_UPLOAD_BYTES",
        4,
    )
    service = FakeTranscriptionService()
    app = create_app(tmp_path, service)
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"12345")

    submitted = asyncio.run(
        ui_callback(ui, "submit_transcription_ui")(str(audio_path))
    )

    assert submitted[1] == (
        "Transkription konnte nicht gestartet werden: "
        "Die Audiodatei ueberschreitet das Uploadlimit."
    )
    assert app.state.jobs._active_id is None
    assert list((tmp_path / "work").iterdir()) == []
    assert service.calls == []


def test_transcription_failure_is_reported_without_a_success_result(tmp_path):
    service = FakeTranscriptionService(
        error=VoiceTranscriptionError("Kein verwertbarer Sprachabschnitt")
    )
    app = create_app(tmp_path, service)
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"encoded audio")

    async def submit_and_wait():
        submitted = await ui_callback(ui, "submit_transcription_ui")(
            str(audio_path)
        )
        await asyncio.gather(*tuple(app.state.background_tasks))
        refreshed = await ui_callback(ui, "refresh_voice_ui")(
            submitted[0],
            None,
            None,
            {},
        )
        return submitted, refreshed

    submitted, refreshed = asyncio.run(submit_and_wait())

    assert submitted[1] == "Transkriptionsauftrag angenommen."
    assert "Kein verwertbarer Sprachabschnitt" in refreshed[11]
    assert refreshed[12] == ""
    assert app.state.jobs._active_id is None
    assert list((tmp_path / "work").iterdir()) == []


def test_transcription_job_cancellation_releases_worker_and_temp_files(tmp_path):
    class BlockingWorker:
        ready = True

        def __init__(self):
            self.started = Event()

        async def transcribe_audio(
            self,
            *,
            upload_path,
            work_dir,
            cancel_event,
        ):
            self.started.set()
            await asyncio.to_thread(cancel_event.wait, 5)
            raise VoiceWorkerCancelled("Transkription abgebrochen")

    app = create_app(tmp_path, FakeTranscriptionService())
    worker = BlockingWorker()
    app.state.voice_worker = worker
    ui = next(route.app for route in app.routes if route.path == "/ui").get_blocks()
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"encoded audio")

    async def submit_and_cancel():
        submitted = await ui_callback(ui, "submit_transcription_ui")(
            str(audio_path)
        )
        assert await asyncio.to_thread(worker.started.wait, 5)
        await app.state.jobs.request_cancel(submitted[0])
        await asyncio.gather(*tuple(app.state.background_tasks))
        return await app.state.jobs.get(submitted[0])

    job = asyncio.run(submit_and_cancel())

    assert job["state"] == "cancelled"
    assert job["result"] is None
    assert app.state.jobs._active_id is None
    assert list((tmp_path / "work").iterdir()) == []
