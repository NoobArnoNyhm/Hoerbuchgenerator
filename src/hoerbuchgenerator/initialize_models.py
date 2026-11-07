import os
from pathlib import Path

from hoerbuchgenerator.model_cache import incomplete_model_files
from hoerbuchgenerator.qwen import QWEN_BASE_MODELS, QWEN_REQUIRED_FILES
from hoerbuchgenerator.whisper import WHISPER_MODELS, WHISPER_REQUIRED_FILES


_QWEN_REPOSITORIES = {
    name: f"Qwen/{name}"
    for name in QWEN_BASE_MODELS
}
_WHISPER_REPOSITORIES = {
    name: f"Systran/faster-whisper-{name}"
    for name in WHISPER_MODELS
}


def _ensure_model(
    *,
    repo_id: str,
    destination: Path,
    required_files: tuple[str, ...],
    token: str | None,
) -> None:
    incomplete = incomplete_model_files(destination, required_files)
    if not incomplete:
        print(f"Modell bereits vollstaendig vorhanden: {destination}", flush=True)
        return

    from huggingface_hub import snapshot_download

    print(f"Lade {repo_id} nach {destination}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
        token=token,
    )
    incomplete = incomplete_model_files(destination, required_files)
    if incomplete:
        raise RuntimeError(
            f"Modell {repo_id} bleibt unvollstaendig: {', '.join(incomplete)}"
        )
    print(f"Modell erfolgreich initialisiert: {destination}", flush=True)


def main() -> None:
    root = Path(os.environ.get("APP_DATA_DIR", "data"))
    qwen_name = os.environ.get("QWEN_MODEL", QWEN_BASE_MODELS[0])
    whisper_name = os.environ.get("WHISPER_MODEL", "base")
    if qwen_name not in _QWEN_REPOSITORIES:
        raise ValueError(f"Nicht unterstuetztes Qwen-Base-Modell: {qwen_name}")
    if whisper_name not in _WHISPER_REPOSITORIES:
        raise ValueError(f"Nicht unterstuetztes Whisper-Modell: {whisper_name}")

    token = os.environ.get("HF_TOKEN") or None
    _ensure_model(
        repo_id=_QWEN_REPOSITORIES[qwen_name],
        destination=root / "models" / "huggingface" / qwen_name,
        required_files=QWEN_REQUIRED_FILES,
        token=token,
    )
    _ensure_model(
        repo_id=_WHISPER_REPOSITORIES[whisper_name],
        destination=root / "models" / "whisper" / whisper_name,
        required_files=WHISPER_REQUIRED_FILES,
        token=token,
    )


if __name__ == "__main__":
    main()
