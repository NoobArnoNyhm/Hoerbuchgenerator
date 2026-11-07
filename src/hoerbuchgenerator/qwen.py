from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from hoerbuchgenerator.model_cache import incomplete_model_files
from hoerbuchgenerator.voice_prompt import VoicePrompt, VoicePromptError, VoicePromptStore


QWEN_BASE_MODELS = (
    "Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-12Hz-1.7B-Base",
)
QWEN_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


class QwenError(RuntimeError):
    pass


@contextmanager
def _offline_model_loading() -> Iterator[None]:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class QwenModelProtocol:
    device: Any

    def create_voice_clone_prompt(self, **kwargs): ...

    def generate_voice_clone(self, **kwargs): ...


class QwenTTSAdapter:
    def __init__(
        self,
        model: QwenModelProtocol,
        tensor_factory: Callable[[Any, Any], Any] | None = None,
    ):
        self._model = model
        self._tensor_factory = tensor_factory or self._torch_tensor

    @staticmethod
    def _torch_tensor(array: Any, device: Any) -> Any:
        import torch

        return torch.as_tensor(array, device=device)

    @classmethod
    def load_local(
        cls,
        model_name: str,
        models_dir: Path,
        *,
        loader: Callable[..., QwenModelProtocol] | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> "QwenTTSAdapter":
        if model_name not in QWEN_BASE_MODELS:
            raise QwenError(f"Nicht unterstuetztes Qwen-Base-Modell: {model_name}")
        model_path = Path(models_dir) / model_name
        incomplete = incomplete_model_files(model_path, QWEN_REQUIRED_FILES)
        if incomplete:
            raise QwenError(
                f"Lokales Qwen-Modell '{model_name}' fehlt oder ist unvollstaendig: "
                f"{', '.join(incomplete)} in {model_path}"
            )

        if loader is None:
            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError as error:
                raise QwenError("qwen-tts fehlt; Installation mit pip install -e '.[qwen]' ") from error
            loader = Qwen3TTSModel.from_pretrained

        kwargs = dict(loader_kwargs or {})
        if "pretrained_model_name_or_path" in kwargs:
            raise QwenError("Der Modellpfad wird aus der validierten Modellwahl gesetzt")
        kwargs["local_files_only"] = True

        try:
            with _offline_model_loading():
                model = loader(str(model_path), **kwargs)
        except Exception as error:
            raise QwenError(f"Qwen-Base-Modell '{model_name}' konnte nicht lokal geladen werden") from error
        return cls(model)

    def create_voice_prompt(self, reference_audio: Path, transcript: str) -> VoicePrompt:
        if not transcript.strip():
            raise QwenError("Das Referenztranskript darf nicht leer sein")
        try:
            items = self._model.create_voice_clone_prompt(
                ref_audio=str(reference_audio),
                ref_text=transcript,
                x_vector_only_mode=False,
            )
        except Exception as error:
            raise QwenError("Qwen konnte das Stimmenprompt nicht aus dem Referenzaudio erstellen") from error
        if len(items) != 1:
            raise QwenError("Qwen lieferte nicht genau ein Stimmenprompt")

        item = items[0]
        try:
            return VoicePrompt(
                ref_code=item.ref_code,
                ref_spk_embedding=item.ref_spk_embedding,
                x_vector_only_mode=bool(item.x_vector_only_mode),
                icl_mode=bool(item.icl_mode),
                ref_text=item.ref_text,
            )
        except AttributeError as error:
            raise QwenError("Qwen lieferte ein unbekanntes Promptformat") from error

    def save_voice_prompt(self, path: Path, prompt: VoicePrompt) -> None:
        try:
            VoicePromptStore.save(path, prompt)
        except VoicePromptError as error:
            raise QwenError(str(error)) from error

    def load_voice_prompt(self, path: Path) -> VoicePrompt:
        try:
            prompt = VoicePromptStore.load(path)
            return VoicePrompt(
                ref_code=self._tensor_factory(prompt.ref_code, self._model.device)
                if prompt.ref_code is not None else None,
                ref_spk_embedding=self._tensor_factory(prompt.ref_spk_embedding, self._model.device),
                x_vector_only_mode=prompt.x_vector_only_mode,
                icl_mode=prompt.icl_mode,
                ref_text=prompt.ref_text,
            )
        except VoicePromptError as error:
            raise QwenError(str(error)) from error
        except (ImportError, RuntimeError, TypeError) as error:
            raise QwenError("Qwen-Prompt konnte nicht fuer das Modell wiederhergestellt werden") from error

    def synthesize(self, text: str, language: str, prompt: VoicePrompt):
        if not isinstance(text, str) or not text.strip() or not isinstance(language, str) or not language.strip():
            raise QwenError("Synthesetext und Sprache muessen angegeben sein")
        try:
            import numpy as np

            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=[prompt],
            )
        except Exception as error:
            raise QwenError("Qwen-Sprachsynthese fehlgeschlagen") from error

        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise QwenError("Qwen lieferte kein einzelnes gueltiges Audiosample")
        try:
            if len(wavs) != 1:
                raise QwenError("Qwen lieferte kein einzelnes gueltiges Audiosample")
            audio = np.asarray(wavs[0])
            valid = (
                audio.ndim == 1
                and audio.size > 0
                and np.issubdtype(audio.dtype, np.number)
                and np.isfinite(audio).all()
            )
        except (TypeError, ValueError) as error:
            raise QwenError("Qwen lieferte leere oder ungueltige Audiodaten") from error
        if not valid:
            raise QwenError("Qwen lieferte leere oder ungueltige Audiodaten")
        return audio, sample_rate