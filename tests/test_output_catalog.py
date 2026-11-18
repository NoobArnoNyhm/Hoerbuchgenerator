from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading

import pytest

from hoerbuchgenerator.output_catalog import OutputCatalog, OutputError


def add_file(root, name, timestamp=1_700_000_000):
    directory = root / "output"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_bytes(b"synthetic audio")
    os.utime(path, (timestamp, timestamp))
    return path


def test_catalog_discovers_regular_audio_only_and_sorts_deterministically(tmp_path):
    catalog = OutputCatalog(tmp_path)
    assert catalog.list() == []
    add_file(tmp_path, "Book One.WAV")
    add_file(tmp_path, "Book One.mp3")
    add_file(tmp_path, "new.wav", 1_700_000_002)
    add_file(tmp_path, ".working.tmp.wav")
    add_file(tmp_path, "other.txt")
    (tmp_path / "output" / "directory.wav").mkdir()
    (tmp_path / "output" / "link.mp3").symlink_to(tmp_path / "output" / "new.wav")

    files = catalog.list()

    assert [file["name"] for file in files] == ["new.wav", "Book One.WAV", "Book One.mp3"]
    assert [file["format"] for file in files] == ["wav", "wav", "mp3"]
    assert "Book%20One.WAV" in files[1]["url"]
    assert str(tmp_path) not in json.dumps(files)


def test_adopted_creation_survives_restart_and_touch_but_version_changes(tmp_path):
    path = add_file(tmp_path, "old.wav")
    original = OutputCatalog(tmp_path).list()[0]
    os.utime(path, (1_800_000_000, 1_800_000_000))

    adopted = OutputCatalog(tmp_path).list()[0]

    assert adopted["created_at"] == original["created_at"]
    assert adopted["version"] != original["version"]


def test_publication_is_newest_and_recreation_rejects_old_selection(tmp_path):
    catalog = OutputCatalog(tmp_path)
    add_file(tmp_path, "future.wav", 2_000_000_000)
    source = tmp_path / "synthetic.wav"
    source.write_bytes(b"synthetic")
    destination = tmp_path / "output" / "new.wav"
    catalog.publish(source, destination)
    first = catalog.list()[0]
    assert first["name"] == "new.wav"
    catalog.delete(first["name"], f'"{first["version"]}"')
    catalog = OutputCatalog(tmp_path)
    catalog.publish(source, destination)
    replacement = catalog.list()[0]
    assert replacement["created_at"] > first["created_at"]
    assert replacement["version"] != first["version"]
    with pytest.raises(OutputError) as changed:
        catalog.delete(first["name"], f'"{first["version"]}"')
    assert changed.value.status == 412
    assert destination.read_bytes() == b"synthetic"


def test_external_replacement_and_removal_reconcile_with_disk(tmp_path):
    catalog = OutputCatalog(tmp_path)
    path = add_file(tmp_path, "book.wav")
    first = catalog.list()[0]
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(b"replacement")
    replacement.replace(path)
    second = catalog.list()[0]
    assert first["version"] != second["version"]
    path.unlink()
    assert catalog.list() == []
    assert OutputCatalog(tmp_path).list() == []


def test_catalog_reconciles_interrupted_publication_intent(tmp_path):
    catalog = OutputCatalog(tmp_path)
    add_file(tmp_path, "good.wav")
    catalog.list()
    index = tmp_path / "config" / "output-catalog.json"
    value = json.loads(index.read_text())
    value["files"]["missing.wav"] = dict(value["files"]["good.wav"])
    index.write_text(json.dumps(value))

    assert [file["name"] for file in OutputCatalog(tmp_path).list()] == ["good.wav"]
    assert set(json.loads(index.read_text())["files"]) == {"good.wav"}


@pytest.mark.parametrize("contents", ["not json", '{"version": 2, "files": {}}'])
def test_corrupt_catalog_is_reported_without_hiding_outputs(tmp_path, contents, caplog):
    path = add_file(tmp_path, "keep.wav")
    config = tmp_path / "config"
    config.mkdir()
    (config / "output-catalog.json").write_text(contents)

    with pytest.raises(OutputError) as failure:
        OutputCatalog(tmp_path).list()
    assert failure.value.code == "output_catalog_invalid"
    assert path.exists()
    assert "Output-Katalog fehlgeschlagen" in caplog.text


def test_failed_catalog_write_prevents_publication_and_keeps_source(tmp_path, monkeypatch):
    catalog = OutputCatalog(tmp_path)
    source = tmp_path / "synthetic.wav"
    source.write_bytes(b"synthetic")
    destination = tmp_path / "output" / "new.wav"
    original_replace = Path.replace

    def fail_index_replace(path, target):
        if target.name == "output-catalog.json":
            raise PermissionError("synthetic read-only catalog")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_index_replace)
    with pytest.raises(OutputError) as failure:
        catalog.publish(source, destination)
    assert failure.value.code == "output_io_failed"
    assert not destination.exists()
    assert source.exists()
    assert list((tmp_path / "config").iterdir()) == []


def test_delete_permission_failure_keeps_file_and_catalog(tmp_path, monkeypatch):
    catalog = OutputCatalog(tmp_path)
    path = add_file(tmp_path, "keep.wav")
    first = catalog.list()[0]
    original_unlink = Path.unlink

    def fail_audio_unlink(target, *args, **kwargs):
        if target == path:
            raise PermissionError("synthetic read-only output")
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_audio_unlink)
    with pytest.raises(OutputError) as failure:
        catalog.delete(path.name, f'"{first["version"]}"')
    assert failure.value.code == "output_io_failed"
    assert catalog.list() == [first]
    assert path.exists()


def test_publication_and_stale_delete_are_serialized(tmp_path, monkeypatch):
    catalog = OutputCatalog(tmp_path)
    source = tmp_path / "synthetic.wav"
    source.write_bytes(b"synthetic")
    destination = tmp_path / "output" / "book.wav"
    catalog.publish(source, destination)
    first = catalog.list()[0]
    catalog.delete(first["name"], f'"{first["version"]}"')
    writing = threading.Event()
    release = threading.Event()
    original_write = catalog._write

    def pause_publication(records):
        original_write(records)
        if "book.wav" in records:
            writing.set()
            assert release.wait(5)

    monkeypatch.setattr(catalog, "_write", pause_publication)
    with ThreadPoolExecutor(max_workers=2) as pool:
        publication = pool.submit(catalog.publish, source, destination)
        assert writing.wait(5)
        deletion = pool.submit(catalog.delete, "book.wav", f'"{first["version"]}"')
        try:
            assert not deletion.done()
        finally:
            release.set()
        publication.result(timeout=5)
        with pytest.raises(OutputError) as failure:
            deletion.result(timeout=5)
        assert failure.value.status == 412
    assert destination.exists()


def test_failed_link_rolls_back_catalog_without_overwriting_existing_audio(tmp_path, monkeypatch):
    catalog = OutputCatalog(tmp_path)
    existing = add_file(tmp_path, "keep.wav")
    first = catalog.list()
    source = tmp_path / "synthetic.wav"
    source.write_bytes(b"synthetic")

    def fail_link(_source, _destination):
        raise OSError("synthetic link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OutputError):
        catalog.publish(source, tmp_path / "output" / "new.wav")
    assert catalog.list() == first
    assert existing.read_bytes() == b"synthetic audio"
    with pytest.raises(FileExistsError):
        catalog.publish(source, existing)
    assert catalog.list() == first


def test_partial_delete_failure_is_explicit_and_recovers_on_next_scan(tmp_path, monkeypatch):
    catalog = OutputCatalog(tmp_path)
    path = add_file(tmp_path, "book.wav")
    item = catalog.list()[0]
    original_write = catalog._write

    def fail_write(_records):
        raise PermissionError("synthetic catalog failure")

    monkeypatch.setattr(catalog, "_write", fail_write)
    with pytest.raises(OutputError) as failure:
        catalog.delete(path.name, f'"{item["version"]}"')
    assert failure.value.code == "output_catalog_write_failed"
    assert "Datei entfernt" in str(failure.value)
    assert not path.exists()
    monkeypatch.setattr(catalog, "_write", original_write)
    assert catalog.list() == []
