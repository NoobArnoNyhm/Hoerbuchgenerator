from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient
import pytest

from hoerbuchgenerator.main import create_app


def test_output_api_lists_legacy_unicode_audio_supports_ranges_and_deletes(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/v1/output").json() == {"files": []}
        name = "H\u00f6rbuch Eins.MP3"
        path = tmp_path / "output" / name
        path.write_bytes(b"synthetic broken audio")
        listing = client.get("/api/v1/output")
        assert listing.headers["cache-control"] == "no-store"
        item = listing.json()["files"][0]
        assert item["name"] == name
        assert item["format"] == "mp3"
        assert item["created_at"]
        response = client.get(item["url"], headers={"Range": "bytes=0-8"})
        assert response.status_code == 206
        assert response.content == b"synthetic"
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["etag"] == f'"{item["version"]}"'
        url = f"/api/v1/output/{quote(name)}"
        assert client.delete(url).status_code == 428
        assert client.delete(url, headers={"If-Match": '"stale"'}).status_code == 412
        assert path.exists()
        assert client.delete(url, headers={"If-Match": response.headers["etag"]}).status_code == 204
        assert not path.exists()
        assert client.get(item["url"]).status_code == 404
        assert client.get("/api/v1/output").json() == {"files": []}
        assert client.delete(url, headers={"If-Match": response.headers["etag"]}).status_code == 404


@pytest.mark.parametrize("name", [
    ".hidden.wav", "nested.wav", "linked.wav", r"..\outside.wav",
    "bad%2Fname.wav", "bad%5Cname.wav", "outside.txt", "C%3Aoutside.wav",
])
def test_output_endpoints_reject_hidden_directory_link_and_traversal(tmp_path, name):
    with TestClient(create_app(tmp_path)) as client:
        output = tmp_path / "output"
        (output / ".hidden.wav").write_bytes(b"hidden")
        (output / "nested.wav").mkdir()
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"outside")
        (output / "linked.wav").symlink_to(outside)
        url = f"/api/v1/output/{name}"
        assert client.get(url).status_code == 404
        assert client.delete(url, headers={"If-Match": '"any"'}).status_code == 404
        assert outside.read_bytes() == b"outside"


def test_stale_download_and_delete_cannot_target_recreated_file(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        path = tmp_path / "output" / "book.wav"
        path.write_bytes(b"old")
        first = client.get("/api/v1/output").json()["files"][0]
        client.delete("/api/v1/output/book.wav", headers={"If-Match": f'"{first["version"]}"'})
        path.write_bytes(b"new")
        second = client.get("/api/v1/output").json()["files"][0]
        assert second["version"] != first["version"]
        assert client.get(first["url"]).status_code == 412
        assert client.get(second["url"]).content == b"new"
        assert client.delete(
            "/api/v1/output/book.wav", headers={"If-Match": f'"{first["version"]}"'},
        ).status_code == 412
        assert path.read_bytes() == b"new"


def test_output_operations_do_not_consume_or_require_inference_slot(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        path = tmp_path / "output" / "book.wav"
        path.write_bytes(b"book")
        job = client.portal.call(app.state.jobs.reserve, "synthesis")
        item = client.get("/api/v1/output").json()["files"][0]
        assert client.get(item["url"]).content == b"book"
        assert client.delete(
            "/api/v1/output/book.wav", headers={"If-Match": f'"{item["version"]}"'},
        ).status_code == 204
        assert client.get("/api/v1/status").json()["active_job"]["id"] == job["id"]


def test_output_io_error_is_explicit_and_never_reported_as_empty(tmp_path, monkeypatch, caplog):
    with TestClient(create_app(tmp_path)) as client:
        path = tmp_path / "output" / "keep.wav"
        path.write_bytes(b"keep")
        assert client.get("/api/v1/output").status_code == 200
        original_iterdir = Path.iterdir

        def fail_output_scan(directory):
            if directory == path.parent:
                raise PermissionError("synthetic inaccessible output")
            return original_iterdir(directory)

        monkeypatch.setattr(Path, "iterdir", fail_output_scan)
        response = client.get("/api/v1/output")
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "output_io_failed"
        assert path.exists()
        assert "Dateisystemzugriff auf Output fehlgeschlagen" in caplog.text


def test_unreadable_audio_reports_error_before_streaming_but_remains_deletable(
    tmp_path, monkeypatch, caplog,
):
    with TestClient(create_app(tmp_path)) as client:
        path = tmp_path / "output" / "book.wav"
        path.write_bytes(b"synthetic audio")
        item = client.get("/api/v1/output").json()["files"][0]
        original_open = open

        def fail_audio_read(target, *args, **kwargs):
            if target == path or target == str(path):
                raise PermissionError("synthetic unreadable audio")
            return original_open(target, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_audio_read)
        response = client.get(item["url"])
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "output_io_failed"
        assert "Hoerbuch konnte nicht gelesen werden" in caplog.text
        assert client.get("/api/v1/output").json()["files"] == [item]
        assert client.delete(
            "/api/v1/output/book.wav", headers={"If-Match": f'"{item["version"]}"'},
        ).status_code == 204
