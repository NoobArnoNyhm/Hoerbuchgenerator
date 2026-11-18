from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
from threading import RLock
import time
from typing import NoReturn, TypedDict
from urllib.parse import quote
from uuid import uuid4


logger = logging.getLogger(__name__)


class OutputError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 500):
        super().__init__(message)
        self.code = code
        self.status = status


class OutputEntry(TypedDict):
    name: str
    format: str
    created_at: str
    version: str
    url: str


class _Record(TypedDict):
    created_ns: int
    identity: str
    generation: str


def _allowed_name(name: str) -> bool:
    return (
        bool(name)
        and not name.startswith(".")
        and not any(character in name for character in '/\\:\x00')
        and not any(ord(character) < 32 for character in name)
        and Path(name).suffix.lower() in {".wav", ".mp3"}
    )


def _identity(info: os.stat_result) -> str:
    return f"{info.st_dev}:{info.st_ino}"


def _version(info: os.stat_result, record: _Record) -> str:
    value = f"{record['generation']}:{_identity(info)}:{info.st_size}:{info.st_mtime_ns}"
    return hashlib.sha256(value.encode()).hexdigest()


class OutputCatalog:
    def __init__(self, root: Path):
        self.directory = root / "output"
        self._index = root / "config" / "output-catalog.json"
        self._lock = RLock()
        self._last_publication_ns = 0

    def _check_directory(self) -> None:
        if self.directory.is_symlink():
            raise OutputError("output_unavailable", "Output darf kein symbolischer Link sein.")

    def _path(self, name: str) -> Path:
        if not _allowed_name(name):
            raise OutputError("output_not_found", "Hoerbuch nicht gefunden.", 404)
        self._check_directory()
        return self.directory / name

    def _stat(self, name: str) -> os.stat_result:
        try:
            info = self._path(name).lstat()
        except FileNotFoundError as error:
            raise OutputError("output_not_found", "Hoerbuch nicht gefunden.", 404) from error
        if not stat.S_ISREG(info.st_mode):
            raise OutputError("output_not_found", "Hoerbuch nicht gefunden.", 404)
        return info

    def _read(self) -> dict[str, _Record]:
        if self._index.is_symlink() or self._index.parent.is_symlink():
            raise OutputError("output_catalog_invalid", "Der Output-Katalog ist ungueltig.")
        try:
            value = json.loads(self._index.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (ValueError, UnicodeError) as error:
            raise OutputError("output_catalog_invalid", "Der Output-Katalog ist unlesbar.") from error
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("files"), dict):
            raise OutputError("output_catalog_invalid", "Der Output-Katalog ist ungueltig.")
        last_publication_ns = value.get("last_publication_ns", 0)
        if type(last_publication_ns) is not int or not 0 <= last_publication_ns < 2**63:
            raise OutputError("output_catalog_invalid", "Der Output-Katalog ist ungueltig.")
        self._last_publication_ns = max(self._last_publication_ns, last_publication_ns)
        records: dict[str, _Record] = {}
        for name, record in value["files"].items():
            if (
                not _allowed_name(name)
                or not isinstance(record, dict)
                or type(record.get("created_ns")) is not int
                or not 0 <= record["created_ns"] < 2**63
                or not isinstance(record.get("identity"), str)
                or not isinstance(record.get("generation"), str)
            ):
                raise OutputError("output_catalog_invalid", "Der Output-Katalog ist ungueltig.")
            records[name] = {
                "created_ns": record["created_ns"],
                "identity": record["identity"],
                "generation": record["generation"],
            }
        return records

    def _write(self, records: dict[str, _Record]) -> None:
        self._index.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._index.with_name(f".output-catalog-{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as target:
                json.dump({
                    "version": 1,
                    "last_publication_ns": self._last_publication_ns,
                    "files": records,
                }, target, ensure_ascii=True)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(self._index)
        finally:
            temporary.unlink(missing_ok=True)

    def _sync(self) -> tuple[dict[str, _Record], dict[str, os.stat_result]]:
        self._check_directory()
        previous = self._read()
        records: dict[str, _Record] = {}
        infos: dict[str, os.stat_result] = {}
        if self.directory.exists():
            for path in self.directory.iterdir():
                if not _allowed_name(path.name):
                    continue
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                record = previous.get(path.name)
                if record is None or record["identity"] != _identity(info):
                    created_ns = getattr(info, "st_birthtime_ns", None)
                    if created_ns is None:
                        birthtime = getattr(info, "st_birthtime", None)
                        created_ns = (
                            int(birthtime * 1_000_000_000)
                            if birthtime is not None else info.st_mtime_ns
                        )
                    record = {
                        "created_ns": max(0, created_ns),
                        "identity": _identity(info),
                        "generation": uuid4().hex,
                    }
                records[path.name] = record
                infos[path.name] = info
        if records != previous:
            self._write(records)
        return records, infos

    def list(self) -> list[OutputEntry]:
        try:
            with self._lock:
                records, infos = self._sync()
                entries: list[OutputEntry] = []
                for name in sorted(records, key=lambda name: (-records[name]["created_ns"], name)):
                    record = records[name]
                    version = _version(infos[name], record)
                    seconds, nanos = divmod(record["created_ns"], 1_000_000_000)
                    created = datetime.fromtimestamp(seconds, timezone.utc)
                    entries.append({
                        "name": name,
                        "format": Path(name).suffix[1:].lower(),
                        "created_at": f"{created:%Y-%m-%dT%H:%M:%S}.{nanos:09d}+00:00",
                        "version": version,
                        "url": f"/api/v1/output/{quote(name, safe='')}?version={version}",
                    })
                return entries
        except (OSError, OutputError) as error:
            self._raise_error(error)

    def get(self, name: str, version: str | None = None) -> tuple[Path, str]:
        try:
            with self._lock:
                self._stat(name)
                records, _ = self._sync()
                info = self._stat(name)
                if name not in records:
                    raise OutputError("output_changed", "Das Hoerbuch wurde geaendert. Bitte aktualisieren.", 412)
                current = _version(info, records[name])
                if version is not None and version != current:
                    raise OutputError("output_changed", "Das Hoerbuch wurde geaendert. Bitte aktualisieren.", 412)
                return self._path(name), current
        except (OSError, OutputError) as error:
            self._raise_error(error)

    def delete(self, name: str, if_match: str | None) -> None:
        try:
            with self._lock:
                self._stat(name)
                if if_match is None:
                    raise OutputError("output_version_required", "Die Dateiversion fehlt.", 428)
                path, current = self.get(name)
                if if_match != f'"{current}"':
                    raise OutputError("output_changed", "Das Hoerbuch wurde geaendert. Bitte erneut auswaehlen.", 412)
                path.unlink()
                # A crash here leaves only a stale index entry, removed by the next scan.
                records = self._read()
                records.pop(name, None)
                try:
                    self._write(records)
                except OSError as error:
                    raise OutputError(
                        "output_catalog_write_failed",
                        "Datei entfernt, aber der Output-Katalog konnte nicht aktualisiert werden.",
                    ) from error
        except (OSError, OutputError) as error:
            self._raise_error(error)

    def publish(self, source: Path, destination: Path) -> None:
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._path(destination.name)
                if destination.parent.resolve() != self.directory.resolve():
                    raise OutputError("output_invalid_target", "Ungueltiges Output-Ziel.")
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(destination)
                records, _ = self._sync()
                previous = dict(records)
                info = source.stat()
                self._last_publication_ns = max(
                    time.time_ns(),
                    self._last_publication_ns + 1,
                    max((record["created_ns"] for record in records.values()), default=0) + 1,
                )
                records[destination.name] = {
                    "created_ns": self._last_publication_ns,
                    "identity": _identity(info),
                    "generation": uuid4().hex,
                }
                # Save the publication intent before making the audio visible.
                self._write(records)
                try:
                    os.link(source, destination)
                except OSError:
                    self._write(previous)
                    raise
        except FileExistsError:
            raise
        except (OSError, OutputError) as error:
            self._raise_error(error)

    @staticmethod
    def _raise_error(error: OSError | OutputError) -> NoReturn:
        if isinstance(error, OutputError):
            if error.status >= 500:
                logger.exception("Output-Katalog fehlgeschlagen: %s", error.code)
            raise error
        logger.exception("Dateisystemzugriff auf Output fehlgeschlagen")
        raise OutputError(
            "output_io_failed", "Der Hoerbuchbestand konnte nicht verarbeitet werden.",
        ) from error
