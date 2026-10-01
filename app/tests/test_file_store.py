import json

import pytest

from support_rag.file_store import (
    FileBusyError,
    FileStore,
    InvalidFileNameError,
    UnsupportedFileError,
)


@pytest.fixture
def store(settings) -> FileStore:
    return FileStore(settings)


@pytest.mark.parametrize("name", ["../secret.pdf", "sub/dir.pdf", ".hidden.pdf", ""])
def test_rejects_unsafe_names(store, name):
    with pytest.raises(InvalidFileNameError):
        store.resolve_inbox_file(name)


def test_rejects_unsupported_extension(store, settings):
    (settings.inbox_dir / "notes.doc").write_bytes(b"x")
    with pytest.raises(UnsupportedFileError):
        store.resolve_inbox_file("notes.doc")


def test_missing_file(store):
    with pytest.raises(FileNotFoundError):
        store.resolve_inbox_file("nope.pdf")


def test_claim_is_exclusive(store, settings):
    path = settings.inbox_dir / "doc.pdf"
    path.write_bytes(b"%PDF")
    claimed = store.claim(path)
    assert claimed.parent == settings.processing_dir
    with pytest.raises(FileBusyError):
        store.claim(path)


def test_processed_keeps_original_and_text(store, settings):
    path = settings.processing_dir / "doc.pdf"
    path.write_bytes(b"%PDF")
    archived = store.mark_processed(path, extracted_text="hello")
    assert archived.read_bytes() == b"%PDF"
    assert archived.with_name("doc.pdf.txt").read_text() == "hello"
    assert not path.exists()


def test_failed_writes_report_and_never_overwrites(store, settings):
    (settings.failed_dir / "doc.pdf").write_bytes(b"older")
    path = settings.processing_dir / "doc.pdf"
    path.write_bytes(b"newer")

    failed = store.mark_failed(path, "boom")

    assert failed.name != "doc.pdf"  # clashing name got a unique suffix
    assert (settings.failed_dir / "doc.pdf").read_bytes() == b"older"
    report = json.loads(failed.with_name(failed.name + ".error.json").read_text())
    assert report["error"] == "boom"


def test_recover_stale_returns_files_to_inbox(store, settings):
    (settings.processing_dir / "doc.pdf").write_bytes(b"%PDF")
    assert store.recover_stale() == ["doc.pdf"]
    assert (settings.inbox_dir / "doc.pdf").exists()


def test_wait_until_stable_returns_for_complete_file(store, settings):
    path = settings.inbox_dir / "doc.pdf"
    path.write_bytes(b"%PDF")
    store.wait_until_stable(path)  # must not raise or hang
