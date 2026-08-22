"""FastAPI application factory (§7). Base path `/api/v1`."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fcesapi.routers import assets, attachments, audit, auth, categories, floorplans, imports, labels, service


def create_app() -> FastAPI:
    app = FastAPI(title="FCES Asset Register API")

    @app.exception_handler(StarletteHTTPException)
    async def _envelope(request, exc: StarletteHTTPException):
        # Every error leaves this shape (§12.2): {"detail": {"code": ..., "message": ...}}.
        # A route that raises HTTPException(detail="plain string") would otherwise leak
        # FastAPI's default shape and break every client parsing this contract.
        detail = exc.detail
        if not isinstance(detail, dict) or "code" not in detail:
            detail = {"code": "error", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(assets.router, prefix="/api/v1")
    app.include_router(categories.router, prefix="/api/v1")
    app.include_router(imports.router, prefix="/api/v1")
    app.include_router(attachments.router, prefix="/api/v1")
    app.include_router(floorplans.router, prefix="/api/v1")
    app.include_router(labels.router, prefix="/api/v1")
    app.include_router(labels.public_router)
    app.include_router(service.router, prefix="/api/v1")
    app.include_router(audit.router, prefix="/api/v1")

    return app


app = create_app()
