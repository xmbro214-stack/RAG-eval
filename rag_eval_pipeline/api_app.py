"""FastAPI application factory for the RAG evaluation API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_eval_pipeline import api
from rag_eval_pipeline.api_services.state import ApiState


DEFAULT_STATIC_ROOT = api.ROOT / "web" / "dist"


def create_app(
    *,
    state: ApiState | None = None,
    static_root: Path = DEFAULT_STATIC_ROOT,
    enable_cors: bool = True,
) -> FastAPI:
    app = FastAPI(title="RAG Eval API", version="0.1.0")
    app.state.rag_eval = state or ApiState()

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "service": "rag-eval-api"}

    static_root = static_root.resolve()
    index_html = static_root / "index.html"
    if static_root.exists():
        assets_root = static_root / "assets"
        if assets_root.exists():
            app.mount("/assets", StaticFiles(directory=assets_root), name="assets")

        @app.get("/")
        @app.get("/datasets")
        def frontend_index() -> FileResponse:
            return FileResponse(index_html)

    return app


app = create_app()
