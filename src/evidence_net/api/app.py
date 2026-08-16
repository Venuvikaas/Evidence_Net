"""FastAPI application entry point for EVIDENCE-Net service (Phase 12).

Initializes FastAPI app, configures CORS middleware, mounts API routes, and
serves static frontend files if built.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from evidence_net.api.database import init_db
from evidence_net.api.routes import router as api_router


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="EVIDENCE-Net API Service",
        description="Product and Technical Review Platform for Evidence-aware Restoration",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Database
    init_db(db_path)

    # Include API routes under /api/v1 and root
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(api_router)

    # Optional static mount for frontend build if present
    dist_dir = Path("frontend/dist")
    if dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
