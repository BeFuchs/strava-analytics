"""FastAPI app: routing and serialization only — all metric math lives in
``metrics/`` and ``clustering/``; this layer calls them and shapes JSON.
"""

from __future__ import annotations

import math
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from plotly.offline import get_plotlyjs

from ride_analytics.clustering.climb_clusters import (
    ClimbCluster,
    ClimbEffort,
    climb_avg_hr,
    cluster_climbs,
)
from ride_analytics.config import AthleteConfig
from ride_analytics.export.csv_export import export_csv
from ride_analytics.ingest import IngestError, Ride, load_fit
from ride_analytics.metrics.climbs import (
    Climb,
    detect_climbs,
    match_climbs,
    ride_elevation_gain_m,
)
from ride_analytics.metrics.durability import compute_durability
from ride_analytics.metrics.pmc import CTL_DAYS, compute_pmc
from ride_analytics.metrics.power_curve import (
    aggregate_power_curve,
    estimate_ftp,
    ride_power_curve,
)
from ride_analytics.metrics.single_ride import compute_ride_metrics
from ride_analytics.metrics.zones import (
    aggregate_zone_distributions,
    hr_zone_distribution,
    power_zone_distribution,
)
from ride_analytics.report.builder import AnalyzedRide, ReportData
from ride_analytics.web import charts
from ride_analytics.web.schemas import (
    ClusterNameRequest,
    DateRange,
    HealthResponse,
    SkippedFile,
    UploadResponse,
)
from ride_analytics.web.session import Session, SessionStore

STATIC_DIR = Path(__file__).parent / "static"

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
# Zip-bomb guards: cap on total unpacked size and on per-entry compression ratio.
MAX_UNPACKED_BYTES = 500 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100

FIT_MAGIC = b".FIT"
FIT_MAGIC_OFFSET = 8

MAX_CLUSTER_NAME_LENGTH = 60

_COPY_CHUNK = 1024 * 1024

_plotly_js_cache: bytes | None = None


def api_error(status_code: int, error: str, detail: str) -> HTTPException:
    """HTTPException whose payload renders as ``{"error": ..., "detail": ...}``."""
    return HTTPException(status_code=status_code, detail={"error": error, "detail": detail})


def create_app(config: AthleteConfig) -> FastAPI:
    app = FastAPI(title="Ride Analytics", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.sessions = SessionStore()

    def get_session(request: Request, x_session_id: str | None = Header(default=None)) -> Session:
        store: SessionStore = request.app.state.sessions
        session = store.get(x_session_id) if x_session_id else None
        if session is None:
            raise api_error(
                404,
                "session not found",
                "Keine oder abgelaufene Sitzung — bitte Dateien erneut hochladen.",
            )
        return session

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid request", "detail": "Ungültige Anfrageparameter."},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            content = exc.detail
        else:
            content = {"error": "request failed", "detail": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak tracebacks to the client.
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal error",
                "detail": "Unerwarteter Serverfehler — Details stehen im Server-Log.",
            },
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # Sync (threadpool) on purpose: building the ~5 MB bundle is blocking work
    # and would stall the event loop for every other request in an async route.
    @app.get("/static/js/plotly.min.js", include_in_schema=False)
    def plotly_js() -> Response:
        # Served from the installed plotly package instead of a vendored copy:
        # keeps the repo small, works offline, and needs no CDN.
        global _plotly_js_cache
        if _plotly_js_cache is None:
            _plotly_js_cache = get_plotlyjs().encode()
        return Response(
            content=_plotly_js_cache,
            media_type="text/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.delete("/api/session")
    async def delete_session(session: Session = Depends(get_session)) -> dict:
        app.state.sessions.delete(session.session_id)
        return {"deleted": True}

    @app.post("/api/upload", response_model=UploadResponse)
    def upload(
        files: list[UploadFile] = File(...),
        x_session_id: str | None = Header(default=None),
    ) -> UploadResponse:
        store: SessionStore = app.state.sessions
        # Re-uploads extend an existing session; otherwise a new one starts.
        session = store.get(x_session_id) if x_session_id else None
        if session is None:
            session = store.create()

        skipped: list[SkippedFile] = []
        processed = 0
        known_starts = session.ride_start_times()

        with tempfile.TemporaryDirectory() as tmp:
            fit_files = _collect_fit_files(files, Path(tmp), skipped)
            for display_name, path in fit_files:
                ride = _parse_fit(display_name, path, skipped)
                if ride is None:
                    continue
                if ride.metadata.start_time in known_starts:
                    skipped.append(SkippedFile(file=display_name, reason="Duplikat"))
                    continue
                known_starts.add(ride.metadata.start_time)
                analyzed, efforts = _analyze_ride(ride, app.state.config)
                session.analyzed.append(analyzed)
                session.climb_efforts.extend(efforts)
                processed += 1

        session.analyzed.sort(key=lambda a: a.ride.metadata.start_time)
        return UploadResponse(
            session_id=session.session_id,
            rides_processed=processed,
            rides_skipped=len(skipped),
            skip_reasons=skipped,
            date_range=_session_date_range(session),
        )

    def date_filter(
        date_from: date | None = Query(default=None),
        date_to: date | None = Query(default=None),
    ) -> tuple[date | None, date | None]:
        if date_from is not None and date_to is not None and date_from > date_to:
            raise api_error(400, "invalid range", "„Von“ liegt nach „Bis“.")
        return date_from, date_to

    @app.get("/api/summary")
    def summary(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        rides = _filtered(session, *dates)
        curve = aggregate_power_curve([a.power_curve for a in rides])
        pmc = _pmc_frame(session, *dates)
        return {
            "n_rides": len(rides),
            "distance_km": sum(a.metrics.distance_km or 0.0 for a in rides),
            "elevation_gain_m": sum(a.elevation_gain_m or 0.0 for a in rides),
            "moving_time_s": sum(a.metrics.moving_time_s for a in rides),
            "total_tss": sum(a.metrics.tss or 0.0 for a in rides),
            "avg_ctl": float(pmc["ctl"].mean()) if not pmc.empty else None,
            "ftp_estimate_watts": estimate_ftp(curve),
        }

    @app.get("/api/pmc")
    def pmc(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        frame = _pmc_frame(session, *dates)
        return {
            "n_rides": len(_filtered(session, *dates)),
            "figure": charts.pmc_figure(frame) if not frame.empty else None,
        }

    @app.get("/api/power-curve")
    def power_curve(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        rides = _filtered(session, *dates)
        curve = aggregate_power_curve([a.power_curve for a in rides])
        return {
            "n_rides": len(rides),
            "figure": charts.power_curve_figure(curve) if curve else None,
            "ftp_estimate_watts": estimate_ftp(curve),
        }

    @app.get("/api/durability")
    def durability(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        rides = _filtered(session, *dates)
        table = compute_durability([a.ride.df for a in rides])
        has_data = (not table.empty) and table["mmp_watts"].notna().any()
        return {
            "n_rides": len(rides),
            "figure": charts.durability_figure(table) if has_data else None,
            "index": _durability_index(table) if has_data else None,
        }

    @app.get("/api/zones")
    def zones(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        rides = _filtered(session, *dates)
        power = aggregate_zone_distributions([a.power_zones for a in rides])
        hr = aggregate_zone_distributions([a.hr_zones for a in rides])
        return {
            "n_rides": len(rides),
            "power": charts.zones_figure(power, charts.POWER_RAMP) if power else None,
            "hr": charts.zones_figure(hr, charts.HR_RAMP) if hr else None,
        }

    @app.get("/api/rides")
    def rides_table(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        rides = _filtered(session, *dates)
        rides.sort(key=lambda a: a.ride.metadata.start_time, reverse=True)
        return {"n_rides": len(rides), "rides": [_ride_row(a) for a in rides]}

    @app.get("/api/climbs")
    def climbs_table(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        rides = _filtered(session, *dates)
        climbs = [climb for a in rides for climb in a.climbs]
        climbs.sort(key=lambda c: c.start_time, reverse=True)
        return {"n_climbs": len(climbs), "climbs": [_climb_row(c) for c in climbs]}

    @app.get("/api/climbs/clusters")
    def climb_clusters(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        clusters = cluster_climbs(_filtered_efforts(session, *dates))
        clusters.sort(key=lambda c: (c.ascent_count, c.last_ridden_date), reverse=True)
        return {
            "n_clusters": len(clusters),
            "clusters": [_cluster_summary(c, session.cluster_names) for c in clusters],
        }

    @app.get("/api/climbs/clusters/{cluster_id}")
    def climb_cluster_detail(
        cluster_id: str,
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> dict:
        efforts = _filtered_efforts(session, *dates)
        cluster = next((c for c in cluster_climbs(efforts) if c.cluster_id == cluster_id), None)
        if cluster is None:
            raise api_error(
                404,
                "cluster not found",
                "Kein Anstieg mit dieser ID im gewählten Zeitraum.",
            )
        return _cluster_detail(cluster, session.cluster_names, efforts)

    @app.get("/api/export/csv")
    def export_csv_zip(
        session: Session = Depends(get_session),
        dates: tuple[date | None, date | None] = Depends(date_filter),
    ) -> Response:
        rides = _filtered(session, *dates)
        if not rides:
            raise api_error(
                404,
                "no rides",
                "Keine Fahrten im gewählten Zeitraum — nichts zu exportieren.",
            )
        data = _report_data_for(session, *dates)
        clusters = cluster_climbs(_filtered_efforts(session, *dates))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            written = export_csv(data, app.state.config.weight_kg, tmp_path)
            written.append(_write_clusters_csv(clusters, session.cluster_names, tmp_path))

            zip_path = tmp_path / "export.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for csv_path in written:
                    archive.write(csv_path, csv_path.name)
            payload = zip_path.read_bytes()

        filename = _export_filename(session, *dates)
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.put("/api/climbs/clusters/{cluster_id}/name")
    def rename_cluster(
        cluster_id: str,
        payload: ClusterNameRequest,
        session: Session = Depends(get_session),
    ) -> dict:
        name = payload.name.strip()
        if len(name) > MAX_CLUSTER_NAME_LENGTH:
            raise api_error(
                400,
                "name too long",
                f"Name darf höchstens {MAX_CLUSTER_NAME_LENGTH} Zeichen haben.",
            )
        if name:
            session.cluster_names[cluster_id] = name
        else:
            # Empty name clears the override; the coordinate label takes over again.
            session.cluster_names.pop(cluster_id, None)
        return {"cluster_id": cluster_id, "name": session.cluster_names.get(cluster_id)}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def _collect_fit_files(
    files: list[UploadFile], tmpdir: Path, skipped: list[SkippedFile]
) -> list[tuple[str, Path]]:
    """Save uploads to the temp dir, unpack ZIPs, and return (name, path) pairs.

    Size violations and unusable archives land in ``skipped`` instead of failing
    the whole upload; only exceeding the 500 MB request total aborts.
    """
    fit_files: list[tuple[str, Path]] = []
    total_bytes = 0
    for i, uploaded in enumerate(files):
        name = Path(uploaded.filename or f"upload_{i}").name
        suffix = Path(name).suffix.lower()
        if suffix not in (".fit", ".zip"):
            skipped.append(
                SkippedFile(file=name, reason="nicht unterstütztes Format (nur .fit und .zip)")
            )
            continue

        target_dir = tmpdir / f"upload_{i}"
        target_dir.mkdir()
        target = target_dir / name
        size = _save_capped(uploaded.file, target, MAX_FILE_BYTES)
        if size is None:
            skipped.append(SkippedFile(file=name, reason="Datei zu groß (max. 50 MB)"))
            continue
        total_bytes += size
        if total_bytes > MAX_UPLOAD_BYTES:
            raise api_error(
                413,
                "upload too large",
                "Upload überschreitet 500 MB — bitte in kleineren Teilen hochladen.",
            )

        if suffix == ".zip":
            fit_files.extend(_extract_zip(target, target_dir / "unpacked", skipped))
        else:
            fit_files.append((name, target))
    return fit_files


def _save_capped(stream: BinaryIO, target: Path, max_bytes: int) -> int | None:
    """Write the stream to ``target``; ``None`` (and no file) when over the cap."""
    written = 0
    with target.open("wb") as out:
        while chunk := stream.read(_COPY_CHUNK):
            written += len(chunk)
            if written > max_bytes:
                target.unlink()
                return None
            out.write(chunk)
    return written


def _extract_zip(
    zip_path: Path, out_dir: Path, skipped: list[SkippedFile]
) -> list[tuple[str, Path]]:
    """Extract only safe ``.fit`` entries; other entries are ignored silently."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        skipped.append(SkippedFile(file=zip_path.name, reason="defektes ZIP-Archiv"))
        return []

    results: list[tuple[str, Path]] = []
    total_unpacked = 0
    with archive:
        for j, info in enumerate(archive.infolist()):
            entry = info.filename
            if info.is_dir() or not entry.lower().endswith(".fit"):
                continue
            # Zip-slip: never extract absolute paths or paths escaping upwards.
            if entry.startswith(("/", "\\")) or ".." in Path(entry).parts or ":" in entry:
                skipped.append(SkippedFile(file=entry, reason="unsicherer Pfad im ZIP"))
                continue
            if info.file_size > MAX_FILE_BYTES:
                skipped.append(SkippedFile(file=entry, reason="Datei zu groß (max. 50 MB)"))
                continue
            compressed = max(info.compress_size, 1)
            if info.file_size / compressed > MAX_COMPRESSION_RATIO:
                skipped.append(SkippedFile(file=entry, reason="verdächtige Kompressionsrate"))
                continue
            total_unpacked += info.file_size
            if total_unpacked > MAX_UNPACKED_BYTES:
                skipped.append(
                    SkippedFile(
                        file=zip_path.name, reason="ZIP-Inhalt zu groß (max. 500 MB entpackt)"
                    )
                )
                break

            entry_dir = out_dir / str(j)
            entry_dir.mkdir(parents=True)
            dest = entry_dir / Path(entry).name
            with archive.open(info) as src:
                if _save_capped(src, dest, MAX_FILE_BYTES) is None:
                    skipped.append(SkippedFile(file=entry, reason="Datei zu groß (max. 50 MB)"))
                    continue
            results.append((Path(entry).name, dest))
    return results


def _parse_fit(display_name: str, path: Path, skipped: list[SkippedFile]) -> Ride | None:
    """Parse one FIT file; unusable files land in ``skipped`` and yield ``None``."""
    with path.open("rb") as fh:
        header = fh.read(FIT_MAGIC_OFFSET + len(FIT_MAGIC))
    if header[FIT_MAGIC_OFFSET:] != FIT_MAGIC:
        skipped.append(SkippedFile(file=display_name, reason="keine gültige FIT-Datei"))
        return None
    try:
        ride = load_fit(path)
    except IngestError:
        skipped.append(SkippedFile(file=display_name, reason="defekte oder unlesbare Datei"))
        return None
    if ride is None:
        skipped.append(SkippedFile(file=display_name, reason="kein Radsport"))
        return None
    return ride


def _analyze_ride(ride: Ride, config: AthleteConfig) -> tuple[AnalyzedRide, list[ClimbEffort]]:
    """Per-ride metrics, computed once at upload (same shape the report uses)."""
    climbs = detect_climbs(ride.df, config)
    analyzed = AnalyzedRide(
        ride=ride,
        metrics=compute_ride_metrics(ride.df, config),
        power_curve=ride_power_curve(ride.df),
        power_zones=power_zone_distribution(ride.df, config),
        hr_zones=hr_zone_distribution(ride.df, config),
        climbs=climbs,
        elevation_gain_m=ride_elevation_gain_m(ride.df),
    )
    efforts = [ClimbEffort(climb=c, avg_hr=climb_avg_hr(ride.df, c)) for c in climbs]
    return analyzed, efforts


def _session_date_range(session: Session) -> DateRange:
    if not session.analyzed:
        return DateRange(min=None, max=None)
    starts = [a.ride.metadata.start_time.date() for a in session.analyzed]
    return DateRange(min=min(starts).isoformat(), max=max(starts).isoformat())


def _filtered(session: Session, date_from: date | None, date_to: date | None) -> list[AnalyzedRide]:
    """Rides whose start date falls inside the inclusive range."""

    def in_range(analyzed: AnalyzedRide) -> bool:
        day = analyzed.ride.metadata.start_time.date()
        return (date_from is None or day >= date_from) and (date_to is None or day <= date_to)

    return [a for a in session.analyzed if in_range(a)]


def _pmc_frame(session: Session, date_from: date | None, date_to: date | None) -> pd.DataFrame:
    """PMC series for the displayed range.

    CTL/ATL are exponentially weighted histories — computing them only from the
    filtered range would restart fitness at zero on the filter boundary. So the
    model runs with a 42-day lead-in before ``date_from`` and the series is
    trimmed to the displayed range afterwards.
    """
    lead_from = date_from - timedelta(days=CTL_DAYS) if date_from else None
    rides = _filtered(session, lead_from, date_to)
    frame = compute_pmc(
        pd.DataFrame(
            {
                "date": [a.ride.metadata.start_time for a in rides],
                "tss": [a.metrics.tss for a in rides],
            }
        )
    )
    if frame.empty:
        return frame
    if date_from is not None:
        frame = frame[frame["date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        frame = frame[frame["date"] <= pd.Timestamp(date_to)]
    return frame.reset_index(drop=True)


def _durability_index(durability: pd.DataFrame) -> dict:
    """Durability index per window x bucket for the dashboard's mini table."""
    buckets = list(dict.fromkeys(durability["bucket"]))
    windows = sorted(durability["window_s"].unique())
    by_key = {
        (row["bucket"], row["window_s"]): row["durability_index"]
        for _, row in durability.iterrows()
    }
    rows = []
    for window in windows:
        cells = [
            None if (value := by_key.get((bucket, window))) is None or pd.isna(value) else value
            for bucket in buckets
        ]
        rows.append({"window": charts.WINDOW_LABELS.get(window, f"{window}s"), "cells": cells})
    return {"buckets": buckets, "rows": rows}


def _ride_row(analyzed: AnalyzedRide) -> dict:
    meta = analyzed.ride.metadata
    m = analyzed.metrics
    return {
        "date": meta.start_time.date().isoformat(),
        "source": meta.source,
        "distance_km": m.distance_km,
        "moving_time_s": m.moving_time_s,
        "elevation_gain_m": analyzed.elevation_gain_m,
        "np_watts": m.np_watts,
        "intensity_factor": m.intensity_factor,
        "tss": m.tss,
        "tss_estimated": m.tss_estimated,
        "avg_hr": m.avg_hr,
    }


def _climb_row(climb: Climb) -> dict:
    return {
        "date": climb.start_time.date().isoformat(),
        "length_km": climb.length_m / 1000,
        "elevation_gain_m": climb.elevation_gain_m,
        "avg_gradient_pct": climb.avg_gradient_pct,
        "max_gradient_pct": climb.max_gradient_pct,
        "duration_s": climb.duration_s,
        "vam_m_per_h": climb.vam_m_per_h,
        "avg_power_watts": climb.avg_power_watts,
        "watts_per_kg": climb.watts_per_kg,
        "kj_before_climb": climb.kj_before_climb,
    }


def _filtered_efforts(
    session: Session, date_from: date | None, date_to: date | None
) -> list[ClimbEffort]:
    """Climb efforts whose start date falls inside the inclusive range."""

    def in_range(effort: ClimbEffort) -> bool:
        day = effort.climb.start_time.date()
        return (date_from is None or day >= date_from) and (date_to is None or day <= date_to)

    return [e for e in session.climb_efforts if in_range(e)]


def _cluster_summary(cluster: ClimbCluster, names: dict[str, str]) -> dict:
    """Cluster metadata without the per-ascent list (list view)."""
    return {
        "cluster_id": cluster.cluster_id,
        "name": names.get(cluster.cluster_id),
        "location_label": cluster.location_label,
        "length_km": cluster.length_km,
        "avg_gradient_pct": cluster.avg_gradient_pct,
        "elevation_gain_m": cluster.elevation_gain_m,
        "ascent_count": cluster.ascent_count,
        "best_time_s": cluster.best_time_s,
        "last_ridden_date": cluster.last_ridden_date.isoformat(),
    }


def _cluster_detail(
    cluster: ClimbCluster, names: dict[str, str], efforts: list[ClimbEffort]
) -> dict:
    """Full cluster incl. its ascents, newest first (detail view).

    Pacing quarters live on the original ``Climb`` rather than on the cluster
    ascent, so they are looked up by start time instead of re-clustering.
    """
    quarters_by_start = {e.climb.start_time: e.climb.quarter_avg_power_watts for e in efforts}
    # Chart runs oldest -> newest; a single ascent has no trend to show.
    chronological = sorted(cluster.ascents, key=lambda a: a.date)
    trend = (
        charts.climb_trend_figure(
            [a.date for a in chronological], [a.duration_s for a in chronological]
        )
        if len(chronological) > 1
        else None
    )
    return {
        **_cluster_summary(cluster, names),
        "trend_figure": trend,
        "ascents": [
            {
                "date": ascent.date.date().isoformat(),
                "duration_s": ascent.duration_s,
                "vam_m_per_h": ascent.vam_m_per_h,
                "avg_power_watts": ascent.avg_power_watts,
                "watts_per_kg": ascent.watts_per_kg,
                "avg_hr": ascent.avg_hr,
                "pacing_quarters": _clean_quarters(quarters_by_start.get(ascent.date)),
            }
            for ascent in cluster.ascents
        ],
    }


def _clean_quarters(quarters: tuple[float, ...] | None) -> list[float | None] | None:
    """Pacing quarters as JSON-safe values — NaN would break strict JSON parsing."""
    if quarters is None:
        return None
    return [None if q is None or math.isnan(q) else q for q in quarters]


def _report_data_for(session: Session, date_from: date | None, date_to: date | None) -> ReportData:
    """Assemble a ReportData from the filtered session rides for CSV export.

    Built from the already-computed per-ride metrics so the export matches what
    the dashboard shows, including the 42-day PMC lead-in.
    """
    rides = _filtered(session, date_from, date_to)
    return ReportData(
        rides=rides,
        pmc=_pmc_frame(session, date_from, date_to),
        power_curve=aggregate_power_curve([a.power_curve for a in rides]),
        ftp_estimate=estimate_ftp(aggregate_power_curve([a.power_curve for a in rides])),
        power_zones=aggregate_zone_distributions([a.power_zones for a in rides]),
        hr_zones=aggregate_zone_distributions([a.hr_zones for a in rides]),
        durability=compute_durability([a.ride.df for a in rides]),
        climb_groups=match_climbs([climb for a in rides for climb in a.climbs]),
    )


CLUSTER_CSV_COLUMNS = [
    "cluster_id",
    "name",
    "location_label",
    "length_km",
    "avg_gradient_pct",
    "elevation_gain_m",
    "ascent_count",
    "best_time_s",
    "last_ridden_date",
]


def _write_clusters_csv(clusters: list[ClimbCluster], names: dict[str, str], out_dir: Path) -> Path:
    """Write ``climb_clusters.csv``: one row per cluster, same conventions as export."""
    rows = [
        {
            "cluster_id": c.cluster_id,
            "name": names.get(c.cluster_id, ""),
            "location_label": c.location_label,
            "length_km": round(c.length_km, 2),
            "avg_gradient_pct": round(c.avg_gradient_pct, 1),
            "elevation_gain_m": round(c.elevation_gain_m),
            "ascent_count": c.ascent_count,
            "best_time_s": round(c.best_time_s),
            "last_ridden_date": c.last_ridden_date.isoformat(),
        }
        for c in clusters
    ]
    frame = pd.DataFrame(rows, columns=CLUSTER_CSV_COLUMNS)
    path = out_dir / "climb_clusters.csv"
    frame.to_csv(path, index=False, encoding="utf-8", na_rep="")
    return path


def _export_filename(session: Session, date_from: date | None, date_to: date | None) -> str:
    """Download name spanning the active range, falling back to the session range."""
    available = _session_date_range(session)
    start = date_from.isoformat() if date_from else available.min
    end = date_to.isoformat() if date_to else available.max
    return f"ride-analytics_{start}_{end}.zip"
