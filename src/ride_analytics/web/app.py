"""FastAPI app: routing and serialization only — all metric math lives in
``metrics/`` and ``clustering/``; this layer calls them and shapes JSON.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from plotly.offline import get_plotlyjs

from ride_analytics.config import AthleteConfig
from ride_analytics.web.schemas import HealthResponse
from ride_analytics.web.session import Session, SessionStore

STATIC_DIR = Path(__file__).parent / "static"

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

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
