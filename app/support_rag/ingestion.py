"""Document ingestion pipeline: file -> text (digital or OCR) -> topic -> chunks -> embeddings -> ChromaDB."""

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .archive import ArchivedText, format_archive_text, parse_archive_text
from .chunking import build_splitter, split_pages
from .config import Settings
from .extraction import extract_pages
from .file_store import FileStore
from .topics import TopicNamer

if TYPE_CHECKING:  # chromadb is heavy; unit tests use a fake store instead
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


@dataclass
class ReindexResult:
    documents: int
    chunks: int
    # Older versions of a file that was later re-ingested; only the newest one is indexed.
    skipped_old_versions: list[str] = field(default_factory=list)


class ReindexError(Exception):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        super().__init__(f"{len(errors)} archive(s) could not be read; the index was not changed")
        self.errors = errors


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionPipeline:
    def __init__(self, settings: Settings, files: FileStore, store: "VectorStore", topics: TopicNamer) -> None:
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

        archived = self._files.mark_processed(path, extracted_text=format_archive_text(original_name, topic, pages))
        return IngestResult(
            "indexed", original_name, doc_id, topic, len(pages), ocr_pages, len(documents), self._rel(archived)
        )

    def reindex(self, regenerate_topics: bool = False) -> ReindexResult:
        """Rebuild the whole index from the text archives in processed/, without OCR.

        Use it after changing CHUNK_SIZE, CHUNK_OVERLAP or EMBEDDING_MODEL.
        All archives are parsed and chunked first; if any of them fails, the
        live index is left untouched (all or nothing).
        """
        with self._lock:  # no ingestion may write to the old collection during the swap
            latest: dict[str, tuple[Path, ArchivedText]] = {}
            skipped: list[str] = []
            errors: list[dict[str, str]] = []
            for original, text_path in self._files.list_archives():  # oldest first
                try:
                    archived = parse_archive_text(text_path.read_text(encoding="utf-8"), original.name)
                    if not archived.pages:
                        raise ValueError("archive contains no text")
                except Exception as exc:
                    errors.append({"archive": self._rel(text_path), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if archived.source in latest:
                    skipped.append(self._rel(latest[archived.source][0]))
                latest[archived.source] = (original, archived)

            batches = []
            for source, (original, archived) in latest.items():
                try:
                    topic = archived.topic
                    if regenerate_topics or not topic:
                        topic = self._topics.name("\n\n".join(p.text for p in archived.pages), source)
                    doc_id = sha256_of(original)
                    batches.append(
                        split_pages(archived.pages, self._splitter, doc_id=doc_id, source=source, topic=topic)
                    )
                except Exception as exc:
                    errors.append({"archive": self._rel(original), "error": f"{type(exc).__name__}: {exc}"})

            if errors:
                raise ReindexError(errors)

            chunks = self._store.rebuild(batches)
            logger.info("Reindexed %d document(s), skipped %d old version(s)", len(batches), len(skipped))
            return ReindexResult(documents=len(batches), chunks=chunks, skipped_old_versions=skipped)

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self._s.data_dir))
