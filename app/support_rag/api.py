"""HTTP API used by n8n (ingestion), by the web UI and by other clients (questions)."""

import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .file_store import (
    FileBusyError,
    FileStore,
    FileTooLargeError,
    InvalidFileNameError,
    UnsupportedFileError,
)
from .ingestion import IngestionError, IngestionPipeline
from .logging_setup import request_id_var, setup_logging
from .qa import QAService
from .topics import TopicNamer
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting support-rag %s", __version__)

    files = FileStore(settings)
    files.ensure_dirs()
    recovered = files.recover_stale()
    if recovered:
        logger.warning("Recovered %d file(s) left in processing/; call /ingest/pending to process them", len(recovered))

    store = VectorStore(settings)
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries,
        api_key=settings.openai_api_key,
    )

    app.state.files = files
    app.state.store = store
    app.state.pipeline = IngestionPipeline(settings, files, store, TopicNamer(llm, settings.topic_max_chars))
    app.state.qa = QAService(settings, store, llm)
    logger.info("Service ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Support RAG", version=__version__, lifespan=lifespan)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = get_settings().api_key.get_secret_value()
    # compare_digest prevents timing attacks on the key comparison.
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


class IngestRequest(BaseModel):
    filename: str = Field(..., examples=["vacation-policy.pdf"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    # The page itself is public; every question it sends still needs the API key.
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict:
    try:
        request.app.state.store.heartbeat()
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ChromaDB unreachable") from exc
    return {"status": "ok", "version": __version__}


# Sync (def) endpoints run in a thread pool, so blocking OCR does not freeze the event loop.
@app.post("/ingest", dependencies=[Depends(require_api_key)])
def ingest(body: IngestRequest, request: Request) -> dict:
    try:
        return asdict(request.app.state.pipeline.ingest(body.filename))
    except InvalidFileNameError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except FileBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except UnsupportedFileError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, {"error": str(exc), "archived_to": exc.archived_to}
        ) from exc


@app.post("/ingest/pending", dependencies=[Depends(require_api_key)])
def ingest_pending(request: Request) -> dict:
    """Safety net: process everything still sitting in inbox/ (missed events, restarts)."""
    results, errors = [], []
    for name in request.app.state.files.list_inbox():
        try:
            results.append(asdict(request.app.state.pipeline.ingest(name)))
        except Exception as exc:
            errors.append({"filename": name, "error": str(exc)})
    return {"processed": results, "errors": errors}


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(body: AskRequest, request: Request) -> dict:
    try:
        return asdict(request.app.state.qa.ask(body.question))
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Knowledge base temporarily unavailable") from exc
