from collections.abc import Sequence
from pathlib import Path


def incomplete_model_files(model_dir: Path, required_files: Sequence[str]) -> list[str]:
    incomplete = []
    for filename in required_files:
        path = model_dir / filename
        try:
            if not path.is_file():
                incomplete.append(f"{filename} (missing)")
            elif path.stat().st_size == 0:
                incomplete.append(f"{filename} (empty)")
        except OSError:
            incomplete.append(f"{filename} (unreadable)")
    return incomplete
