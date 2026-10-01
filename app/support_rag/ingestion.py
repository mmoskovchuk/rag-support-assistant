"""Document ingestion pipeline: file -> text (digital or OCR) -> topic -> chunks -> embeddings -> ChromaDB."""

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .chunking import build_splitter, split_pages
from .config import Settings
from .extraction import extract_pages
from .file_store import FileStore
from .topics import TopicNamer
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    def __init__(self, message: str, archived_to: str) -> None:
        super().__init__(message)
        self.archived_to = archived_to


@dataclass
class IngestResult:
    status: Literal["indexed", "duplicate"]
    filename: str
    doc_id: str
    topic: str | None
    pages: int
    ocr_pages: int
    chunks: int
    archived_to: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionPipeline:
    def __init__(self, settings: Settings, files: FileStore, store: VectorStore, topics: TopicNamer) -> None:
        self._s = settings
        self._files = files
        self._store = store
        self._topics = topics
        self._splitter = build_splitter(settings.chunk_size, settings.chunk_overlap)
        # OCR and embedding are CPU-heavy; processing one document at a time keeps
        # memory predictable on a small server.
        self._lock = threading.Lock()

    def ingest(self, filename: str) -> IngestResult:
        path = self._files.resolve_inbox_file(filename)
        self._files.wait_until_stable(path)
        claimed = self._files.claim(path)

        try:
            with self._lock:
                return self._process(claimed, original_name=path.name)
        except Exception as exc:
            logger.exception("Ingestion of %s failed", path.name)
            try:
                failed = self._files.mark_failed(claimed, f"{type(exc).__name__}: {exc}")
            except OSError:
                # The file stays in processing/ and will be recovered on next start.
                logger.critical("Could not move %s to failed/; left in processing/", claimed.name, exc_info=True)
                raise IngestionError(str(exc), archived_to=str(claimed)) from exc
            raise IngestionError(str(exc), archived_to=self._rel(failed)) from exc

    def _process(self, path: Path, original_name: str) -> IngestResult:
        doc_id = sha256_of(path)

        if self._store.has_document(doc_id):
            logger.info("%s is a duplicate of an indexed document (%s), skipping", original_name, doc_id[:12])
            archived = self._files.mark_processed(path, extracted_text=None)
            return IngestResult("duplicate", original_name, doc_id, None, 0, 0, 0, self._rel(archived))

        pages = extract_pages(path, self._s)
        full_text = "\n\n".join(p.text for p in pages)
        topic = self._topics.name(full_text, original_name)
        documents, ids = split_pages(pages, self._splitter, doc_id=doc_id, source=original_name, topic=topic)
        ocr_pages = sum(p.method == "ocr" for p in pages)
        logger.info("%s: topic %r, %d page(s) -> %d chunk(s)", original_name, topic, len(pages), len(documents))

        # Order matters for data safety: store the new version first, then drop the old one.
        self._store.upsert(documents, ids)
        self._store.delete_other_versions(original_name, keep_doc_id=doc_id)

        archive_text = f"Topic: {topic}\n\n" + "\n\n".join(
            f"--- page {p.page_number} ({p.method}) ---\n{p.text}" for p in pages
        )
        archived = self._files.mark_processed(path, extracted_text=archive_text)
        return IngestResult(
            "indexed", original_name, doc_id, topic, len(pages), ocr_pages, len(documents), self._rel(archived)
        )

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self._s.data_dir))
