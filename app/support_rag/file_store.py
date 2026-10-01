"""Safe file lifecycle: inbox -> processing -> processed | failed.

Data-safety rules enforced here:
  * Source files are never deleted, only moved.
  * Every move is an atomic rename inside one filesystem (/data), so after a
    crash a file sits in exactly one known folder.
  * Existing files are never overwritten: a clashing name gets a timestamp.
  * Files left in processing/ by a crash are returned to inbox/ on startup.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS, Settings

logger = logging.getLogger(__name__)


class FileStoreError(Exception):
    """Base class for file lifecycle errors."""


class InvalidFileNameError(FileStoreError):
    pass


class UnsupportedFileError(FileStoreError):
    pass


class FileTooLargeError(FileStoreError):
    pass


class FileBusyError(FileStoreError):
    """The file disappeared from inbox/ (most likely claimed by a parallel request)."""


class FileStore:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.inbox = settings.inbox_dir
        self.processing = settings.processing_dir
        self.processed = settings.processed_dir
        self.failed = settings.failed_dir

    def ensure_dirs(self) -> None:
        for directory in (self.inbox, self.processing, self.processed, self.failed):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_inbox_file(self, filename: str) -> Path:
        """Validate a client-supplied file name and return its path in inbox/."""
        # Only bare file names are accepted: this blocks path traversal like "../../etc/passwd".
        name = Path(filename).name
        if not name or name != filename or name.startswith("."):
            raise InvalidFileNameError(f"Invalid file name: {filename!r}")

        path = self.inbox / name
        if not path.is_file():
            raise FileNotFoundError(f"File not found in inbox: {name}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(f"Unsupported file type: {path.suffix}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self._s.max_file_size_mb:
            raise FileTooLargeError(f"{name} is {size_mb:.1f} MB, limit is {self._s.max_file_size_mb} MB")
        return path

    def list_inbox(self) -> list[str]:
        return sorted(
            p.name
            for p in self.inbox.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def list_archives(self) -> list[tuple[Path, Path]]:
        """Return (original file, text archive) pairs from processed/, oldest first.

        Day folders sort chronologically by name; within a day, by modification time.
        """
        pairs = []
        for text_path in self.processed.glob("*/*.txt"):
            original = text_path.with_name(text_path.name.removesuffix(".txt"))
            if original.is_file():
                pairs.append((original, text_path))
        return sorted(pairs, key=lambda pair: (pair[0].parent.name, pair[1].stat().st_mtime))

    def wait_until_stable(self, path: Path) -> None:
        """Block until the file size stops changing.

        The folder watcher fires as soon as a file appears, often while a scanner
        or network copy is still writing it. Reading a half-written PDF would
        produce a corrupted result, so we wait for the size to settle first.
        """
        deadline = time.monotonic() + self._s.file_stable_timeout
        last_size = -1
        stable_since = time.monotonic()
        while True:
            try:
                size = path.stat().st_size
            except FileNotFoundError as exc:
                raise FileBusyError(f"{path.name} disappeared while waiting") from exc

            now = time.monotonic()
            if size != last_size:
                last_size, stable_since = size, now
            elif size > 0 and now - stable_since >= self._s.file_stable_seconds:
                return
            if now > deadline:
                raise TimeoutError(f"{path.name} is still being written after {self._s.file_stable_timeout}s")
            time.sleep(0.5)

    def claim(self, path: Path) -> Path:
        """Atomically move a file from inbox/ to processing/.

        The rename doubles as a lock: if two requests race for the same file,
        only one rename succeeds and the other gets FileBusyError.
        """
        target = self._unique_path(self.processing / path.name)
        try:
            path.rename(target)
        except FileNotFoundError as exc:
            raise FileBusyError(f"{path.name} is already being processed") from exc
        logger.info("Claimed %s for processing", path.name)
        return target

    def mark_processed(self, path: Path, extracted_text: str | None) -> Path:
        """Archive the original file and, when available, its OCR text next to it.

        Keeping the text means the index can be rebuilt later (for example with
        a new embedding model) without re-running the slow OCR step.
        """
        day_dir = self.processed / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        target = self._unique_path(day_dir / path.name)
        if extracted_text is not None:
            self._atomic_write(target.with_name(target.name + ".txt"), extracted_text)
        path.rename(target)
        logger.info("Archived %s -> %s", path.name, target.relative_to(self._s.data_dir))
        return target

    def mark_failed(self, path: Path, error: str) -> Path:
        """Move a file to failed/ together with a JSON report for manual review."""
        target = self._unique_path(self.failed / path.name)
        report = {
            "file": path.name,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        self._atomic_write(target.with_name(target.name + ".error.json"), json.dumps(report, indent=2))
        path.rename(target)
        logger.warning("Moved %s to failed/", path.name)
        return target

    def recover_stale(self) -> list[str]:
        """Return files abandoned in processing/ (e.g. after a crash) to inbox/."""
        recovered: list[str] = []
        for path in self.processing.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            target = self._unique_path(self.inbox / path.name)
            path.rename(target)
            recovered.append(target.name)
            logger.warning("Recovered stale file %s back to inbox/", path.name)
        return recovered

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return path.with_name(f"{path.stem}__{stamp}{path.suffix}")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        # Write to a temp file, fsync, then rename: readers never see a partial file.
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
