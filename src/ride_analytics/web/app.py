"""FastAPI app: routing and serialization only — all metric math lives in
``metrics/`` and ``clustering/``; this layer calls them and shapes JSON.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from plotly.offline import get_plotlyjs

from ride_analytics.clustering.climb_clusters import ClimbEffort, climb_avg_hr
from ride_analytics.config import AthleteConfig
from ride_analytics.ingest import IngestError, Ride, load_fit
from ride_analytics.metrics.climbs import detect_climbs, ride_elevation_gain_m
from ride_analytics.metrics.power_curve import ride_power_curve
from ride_analytics.metrics.single_ride import compute_ride_metrics
from ride_analytics.metrics.zones import hr_zone_distribution, power_zone_distribution
from ride_analytics.report.builder import AnalyzedRide
from ride_analytics.web.schemas import (
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

    @app.get("/static/js/plotly.min.js", include_in_schema=False)
    async def plotly_js() -> Response:
        # Served from the installed plotly package instead of a vendored copy:
        # keeps the repo small, works offline, and needs no CDN.
        global _plotly_js_cache
        if _plotly_js_cache is None:
            _plotly_js_cache = get_plotlyjs().encode()
        return Response(content=_plotly_js_cache, media_type="text/javascript")

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
