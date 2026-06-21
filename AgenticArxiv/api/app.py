# AgenticArxiv/api/app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ensure tools are registered (side-effect import)
import tools.arxiv_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401

from api.endpoints import router as api_router
from models.db import init_db
from models.store import store
from utils.logger import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Initializing database tables...")
    init_db()
    store.validate_local_paths()
    log.info("Database ready.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Arxiv API",
        version="0.2.0",
        description="Expose ToolRegistry tools via FastAPI (MySQL-backed)",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"msg": "Agentic Arxiv API is running", "docs": "/docs"}

    app.include_router(api_router)
    return app


app = create_app()
