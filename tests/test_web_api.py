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


@pytest.fixture
def seeded(client, tmp_path):
    """Session with three rides in May, June and July 2024; returns headers."""
    files = [
        (f"ride{i}.fit", fit_bytes(tmp_path, start=datetime(2024, month, 1, 8, 0)))
        for i, month in enumerate((5, 6, 7))
    ]
    session_id = upload(client, files).json()["session_id"]
    return {"X-Session-Id": session_id}


def test_rides_sorted_by_date_descending(client, seeded):
    body = client.get("/api/rides", headers=seeded).json()
    assert [r["date"] for r in body["rides"]] == ["2024-07-01", "2024-06-01", "2024-05-01"]


def test_date_filter_limits_rides(client, seeded):
    response = client.get(
        "/api/rides",
        params={"date_from": "2024-05-15", "date_to": "2024-06-15"},
        headers=seeded,
    )
    body = response.json()
    assert body["n_rides"] == 1
    assert body["rides"][0]["date"] == "2024-06-01"


def test_date_filter_bounds_are_inclusive(client, seeded):
    response = client.get(
        "/api/rides",
        params={"date_from": "2024-06-01", "date_to": "2024-06-01"},
        headers=seeded,
    )
    assert response.json()["n_rides"] == 1


def test_inverted_date_range_is_rejected(client, seeded):
    response = client.get(
        "/api/rides",
        params={"date_from": "2024-07-01", "date_to": "2024-05-01"},
        headers=seeded,
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid range"


def test_malformed_date_is_rejected(client, seeded):
    response = client.get("/api/rides", params={"date_from": "not-a-date"}, headers=seeded)
    assert response.status_code == 400


def test_data_endpoints_require_session(client):
    for path in ("/api/summary", "/api/pmc", "/api/rides", "/api/climbs", "/api/zones"):
        assert client.get(path).status_code == 404


def test_summary_metrics(client, seeded):
    body = client.get("/api/summary", headers=seeded).json()
    assert body["n_rides"] == 3
    assert body["total_tss"] > 0  # rides carry power -> real TSS
    assert body["moving_time_s"] > 0
    assert body["avg_ctl"] is not None


def test_pmc_figure_present(client, seeded):
    body = client.get("/api/pmc", headers=seeded).json()
    assert body["n_rides"] == 3
    assert body["figure"] is not None
    assert body["figure"]["data"]  # plotly traces


def test_zones_figures_present(client, seeded):
    body = client.get("/api/zones", headers=seeded).json()
    assert body["power"] is not None
    assert body["hr"] is not None


def test_empty_range_returns_empty_state(client, seeded):
    params = {"date_from": "2023-01-01", "date_to": "2023-12-31"}
    rides = client.get("/api/rides", params=params, headers=seeded).json()
    assert rides == {"n_rides": 0, "rides": []}
    pmc = client.get("/api/pmc", params=params, headers=seeded).json()
    assert pmc == {"n_rides": 0, "figure": None}
    summary = client.get("/api/summary", params=params, headers=seeded).json()
    assert summary["n_rides"] == 0
    assert summary["avg_ctl"] is None


def test_delete_session(client, tmp_path):
    session_id = upload(client, [("ride.fit", fit_bytes(tmp_path))]).json()["session_id"]
    response = client.delete("/api/session", headers={"X-Session-Id": session_id})
    assert response.status_code == 200
    # Deleted session is gone.
    response = client.delete("/api/session", headers={"X-Session-Id": session_id})
    assert response.status_code == 404
