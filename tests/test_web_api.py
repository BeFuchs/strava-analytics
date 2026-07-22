"""Web API tests via the FastAPI TestClient."""

from __future__ import annotations

import zipfile
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from tests.conftest import SPORT_RUNNING, build_fit

from ride_analytics.config import AthleteConfig
from ride_analytics.web.app import create_app

CONFIG = AthleteConfig(ftp_watts=250, threshold_hr=160, weight_kg=80.0, max_hr=190)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(CONFIG))


def fit_bytes(tmp_path, name="ride.fit", start=datetime(2024, 5, 1, 8, 0), **kwargs) -> bytes:
    records = [{"power": 200, "heart_rate": 140} for _ in range(60)]
    path = build_fit(tmp_path / name, records, start=start, **kwargs)
    data = path.read_bytes()
    path.unlink()
    return data


def upload(client, files, session_id=None):
    headers = {"X-Session-Id": session_id} if session_id else {}
    return client.post(
        "/api/upload",
        files=[("files", (name, data, "application/octet-stream")) for name, data in files],
        headers=headers,
    )


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_single_fit(client, tmp_path):
    response = upload(client, [("ride.fit", fit_bytes(tmp_path))])
    assert response.status_code == 200
    body = response.json()
    assert body["rides_processed"] == 1
    assert body["rides_skipped"] == 0
    assert body["session_id"]
    assert body["date_range"] == {"min": "2024-05-01", "max": "2024-05-01"}


def test_upload_zip_with_fit_inside(client, tmp_path):
    inner = fit_bytes(tmp_path)
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("activities/ride.fit", inner)
        zf.writestr("activities/notes.txt", "ignore me")
    response = upload(client, [("export.zip", archive.read_bytes())])
    assert response.status_code == 200
    body = response.json()
    assert body["rides_processed"] == 1
    assert body["rides_skipped"] == 0


def test_zip_slip_entry_is_not_extracted(client, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.fit", fit_bytes(tmp_path))
    response = upload(client, [("evil.zip", archive.read_bytes())])
    assert response.status_code == 200
    body = response.json()
    assert body["rides_processed"] == 0
    assert body["skip_reasons"] == [{"file": "../evil.fit", "reason": "unsicherer Pfad im ZIP"}]


def test_duplicate_rides_are_skipped(client, tmp_path):
    data = fit_bytes(tmp_path)
    response = upload(client, [("a.fit", data), ("b.fit", data)])
    body = response.json()
    assert body["rides_processed"] == 1
    assert body["skip_reasons"] == [{"file": "b.fit", "reason": "Duplikat"}]

    # Re-uploading into the same session is also a duplicate.
    again = upload(client, [("c.fit", data)], session_id=body["session_id"])
    assert again.json()["skip_reasons"] == [{"file": "c.fit", "reason": "Duplikat"}]


def test_non_cycling_activity_is_skipped(client, tmp_path):
    response = upload(client, [("run.fit", fit_bytes(tmp_path, sport=SPORT_RUNNING))])
    body = response.json()
    assert body["rides_processed"] == 0
    assert body["skip_reasons"] == [{"file": "run.fit", "reason": "kein Radsport"}]


def test_file_without_fit_magic_is_skipped(client):
    response = upload(client, [("junk.fit", b"this is not a fit file at all")])
    body = response.json()
    assert body["rides_processed"] == 0
    assert body["skip_reasons"] == [{"file": "junk.fit", "reason": "keine gültige FIT-Datei"}]


def test_unsupported_extension_is_skipped(client):
    response = upload(client, [("route.gpx", b"<gpx/>")])
    body = response.json()
    assert body["rides_processed"] == 0
    assert "nicht unterstütztes Format" in body["skip_reasons"][0]["reason"]


def test_upload_without_files_is_rejected(client):
    response = client.post("/api/upload")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid request"


def test_missing_session_returns_404(client):
    response = client.delete("/api/session")
    assert response.status_code == 404
    assert response.json()["error"] == "session not found"


def test_delete_session(client, tmp_path):
    session_id = upload(client, [("ride.fit", fit_bytes(tmp_path))]).json()["session_id"]
    response = client.delete("/api/session", headers={"X-Session-Id": session_id})
    assert response.status_code == 200
    # Deleted session is gone.
    response = client.delete("/api/session", headers={"X-Session-Id": session_id})
    assert response.status_code == 404
