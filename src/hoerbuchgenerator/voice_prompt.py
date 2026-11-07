from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any


class VoicePromptError(ValueError):
    pass


@dataclass(frozen=True)
class VoicePrompt:
    ref_code: Any | None
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None


def _array(value: Any, name: str, *, optional: bool = False):
    if value is None and optional:
        return None
    try:
        import numpy as np

        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            try:
                value = tensor.numpy()
            except TypeError:
                value = tensor.float().numpy()
        result = np.asarray(value)
    except Exception as error:
        raise VoicePromptError(f"Qwen-Promptfeld '{name}' ist kein Tensor/Array") from error

    if result.size == 0 or result.dtype.hasobject or not np.issubdtype(result.dtype, np.number):
        raise VoicePromptError(f"Qwen-Promptfeld '{name}' ist leer oder ungueltig")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise VoicePromptError(f"Qwen-Promptfeld '{name}' enthaelt nicht-endliche Werte")
    return np.ascontiguousarray(result)


class VoicePromptStore:
    SCHEMA_VERSION = 1

    @staticmethod
    def _validate_modes(x_vector_only_mode: bool, icl_mode: bool, ref_text: str | None) -> None:
        if not isinstance(x_vector_only_mode, bool) or not isinstance(icl_mode, bool):
            raise VoicePromptError("Qwen-Prompt-Modusflags muessen boolesch sein")
        if x_vector_only_mode == icl_mode:
            raise VoicePromptError("Qwen-Prompt-Modusflags sind widerspruechlich")
        if icl_mode and (not isinstance(ref_text, str) or not ref_text.strip()):
            raise VoicePromptError("ICL-Prompt benoetigt einen Referenztext")

    @classmethod
    def save(cls, path: Path, prompt: VoicePrompt) -> None:
        import numpy as np

        cls._validate_modes(prompt.x_vector_only_mode, prompt.icl_mode, prompt.ref_text)
        if prompt.ref_text is not None and not isinstance(prompt.ref_text, str):
            raise VoicePromptError("Qwen-Referenztext muss Text oder None sein")

        arrays = {
            "ref_spk_embedding": _array(prompt.ref_spk_embedding, "ref_spk_embedding"),
        }
        if not np.issubdtype(arrays["ref_spk_embedding"].dtype, np.floating):
            raise VoicePromptError("Qwen-Speaker-Embedding muss Gleitkommazahlen enthalten")
        if prompt.ref_code is not None:
            arrays["ref_code"] = _array(prompt.ref_code, "ref_code")
            if not np.issubdtype(arrays["ref_code"].dtype, np.integer):
                raise VoicePromptError("Qwen-Referenzcodes muessen ganzzahlig sein")
        arrays["metadata"] = np.asarray(
            json.dumps(
                {
                    "schema_version": cls.SCHEMA_VERSION,
                    "has_ref_code": prompt.ref_code is not None,
                    "x_vector_only_mode": prompt.x_vector_only_mode,
                    "icl_mode": prompt.icl_mode,
                    "ref_text": prompt.ref_text,
                },
                ensure_ascii=False,
            )
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(temporary, **arrays)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        except OSError as error:
            raise VoicePromptError("Qwen-Prompt konnte nicht gespeichert werden") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> VoicePrompt:
        import numpy as np

        try:
            with np.load(Path(path), allow_pickle=False) as archive:
                if "metadata" not in archive or "ref_spk_embedding" not in archive:
                    raise VoicePromptError("Qwen-Promptdatei enthaelt erforderliche Felder nicht")
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("schema_version") != cls.SCHEMA_VERSION:
                    raise VoicePromptError("Qwen-Promptdatei hat eine unbekannte Schema-Version")
                has_ref_code = metadata.get("has_ref_code")
                if not isinstance(has_ref_code, bool):
                    raise VoicePromptError("Qwen-Promptdatei hat ungueltige Metadaten")
                ref_code = _array(archive["ref_code"], "ref_code") if has_ref_code and "ref_code" in archive else None
                if has_ref_code and ref_code is None:
                    raise VoicePromptError("Qwen-Promptdatei enthaelt den Referenzcode nicht")
                embedding = _array(archive["ref_spk_embedding"], "ref_spk_embedding")
                if not np.issubdtype(embedding.dtype, np.floating):
                    raise VoicePromptError("Qwen-Speaker-Embedding muss Gleitkommazahlen enthalten")
                if ref_code is not None and not np.issubdtype(ref_code.dtype, np.integer):
                    raise VoicePromptError("Qwen-Referenzcodes muessen ganzzahlig sein")
                x_vector_only_mode = metadata["x_vector_only_mode"]
                icl_mode = metadata["icl_mode"]
                ref_text = metadata["ref_text"]
                if ref_text is not None and not isinstance(ref_text, str):
                    raise VoicePromptError("Qwen-Promptdatei hat ungueltigen Referenztext")
                cls._validate_modes(x_vector_only_mode, icl_mode, ref_text)
        except VoicePromptError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise VoicePromptError("Qwen-Promptdatei ist unlesbar oder ungueltig") from error

        return VoicePrompt(
            ref_code=ref_code,
            ref_spk_embedding=embedding,
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
            ref_text=ref_text,
        )