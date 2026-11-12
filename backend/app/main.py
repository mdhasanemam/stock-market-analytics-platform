from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.app.api.routes import router
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import DataProviderError, InsufficientData, ProviderUnavailable, SymbolNotFound
from backend.app.core.logging import setup_logging
from backend.app.database.db import Database
from backend.app.services.data_service import DataService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.providers import get_provider

log = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


class RateLimiter:
    """Minimal in-memory sliding-window limiter per client IP (single process). Use a gateway/Redis in multi-worker prod."""
    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    is_prod = settings.environment.lower() == "production"
    # In production the interactive API docs/schema are not exposed (smaller attack surface, no free API map for scanners).
    app = FastAPI(title=settings.app_name, version="1.0.0",
                  description="Stock analytics and walk-forward ML forecasting. Educational use only - not investment advice.",
                  docs_url=None if is_prod else "/docs", redoc_url=None if is_prod else "/redoc",
                  openapi_url=None if is_prod else "/openapi.json")
    app.state.settings = settings
    app.state.db = Database(settings.database_url)
    app.state.data = DataService(get_provider(settings), settings)
    app.state.predictions = PredictionService(app.state.data, settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.url.path.startswith("/api/") and not limiter.allow(request.client.host if request.client else "unknown"):
            return JSONResponse({"detail": "Rate limit exceeded. Please slow down."}, status_code=429)
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        return resp

    @app.exception_handler(SymbolNotFound)
    async def _nf(_, exc):  # noqa: ANN001
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ProviderUnavailable)
    async def _pu(_, exc):  # noqa: ANN001
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(DataProviderError)
    async def _dp(_, exc):  # noqa: ANN001
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.exception_handler(InsufficientData)
    async def _id(_, exc):  # noqa: ANN001
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(ValueError)
    async def _ve(_, exc):  # noqa: ANN001
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_, exc):  # noqa: ANN001
        log.exception("Unhandled error")
        return JSONResponse({"detail": "Internal server error."}, status_code=500)

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
    templates = Jinja2Templates(directory=str(FRONTEND / "templates"))

    def page(name: str, title: str):
        async def handler(request: Request):
            return templates.TemplateResponse(request, name, {"title": title, "app_name": settings.app_name, "provider": app.state.data.provider.name})
        return handler

    for path, tpl, title in [("/", "index.html", "Home"), ("/dashboard", "dashboard.html", "Dashboard"), ("/predict", "predict.html", "Predict"),
                             ("/compare", "compare.html", "Model Comparison"), ("/watchlist", "watchlist.html", "Watchlist"), ("/history", "history.html", "Prediction History")]:
        app.add_api_route(path, page(tpl, title), methods=["GET"], include_in_schema=False)
    return app


app = create_app()
