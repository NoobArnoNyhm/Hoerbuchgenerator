import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class VoiceCatalogError(RuntimeError):
    pass


def list_voice_profiles(root: Path) -> list[dict[str, object]]:
    voices_dir = root / "voices"
    if not voices_dir.is_dir():
        return []

    voices: list[dict[str, object]] = []
    for profile_dir in sorted(voices_dir.iterdir(), key=lambda path: path.name):
        if (
            profile_dir.name.startswith(".")
            or profile_dir.is_symlink()
            or not profile_dir.is_dir()
        ):
            continue

        metadata_path = profile_dir / "metadata.json"
        if not metadata_path.is_file() or metadata_path.is_symlink():
            logger.error(
                "Metadaten fuer Stimmenprofil %s fehlen oder sind ein symbolischer Link",
                profile_dir.name,
            )
            raise VoiceCatalogError(profile_dir.name)

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            logger.exception(
                "Metadaten fuer Stimmenprofil %s konnten nicht gelesen werden",
                profile_dir.name,
            )
            raise VoiceCatalogError(profile_dir.name) from error

        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("name"), str)
            or metadata.get("slug") != profile_dir.name
            or not isinstance(metadata.get("language"), str)
        ):
            logger.error(
                "Metadaten fuer Stimmenprofil %s haben ein ungueltiges Format",
                profile_dir.name,
            )
            raise VoiceCatalogError(profile_dir.name)

        example_path = profile_dir / "example.wav"
        voices.append(
            {
                "slug": profile_dir.name,
                "name": metadata["name"],
                "language": metadata["language"],
                "qwen_model": metadata.get("qwen_model"),
                "created_at": metadata.get("created_at"),
                "example_audio_url": (
                    f"/api/v1/voices/{profile_dir.name}/example"
                    if example_path.is_file() and not example_path.is_symlink()
                    else None
                ),
            }
        )
    return voices
