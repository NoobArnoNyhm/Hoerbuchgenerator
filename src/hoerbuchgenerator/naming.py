import re
import unicodedata


_GERMAN_TRANSLATIONS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def make_slug(name: str, max_length: int = 120) -> str:
    if max_length < 1:
        raise ValueError("max_length muss mindestens 1 sein")

    normalized = name.strip().lower().translate(_GERMAN_TRANSLATIONS)
    normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9._-]+", "_", normalized).strip(" ._-")
    slug = slug[:max_length].rstrip(" ._-")

    if not slug:
        raise ValueError("Der Name ergibt keinen gueltigen Dateinamen")
    if slug.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError("Der Name ist unter Windows reserviert")
    return slug
