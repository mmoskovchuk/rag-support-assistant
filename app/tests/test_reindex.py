import pytest
from langchain_core.language_models import FakeListChatModel

from support_rag.archive import format_archive_text, parse_archive_text
from support_rag.extraction import PageText
from support_rag.file_store import FileStore
from support_rag.ingestion import IngestionPipeline, ReindexError, sha256_of
from support_rag.topics import TopicNamer


class FakeStore:
    def __init__(self):
        self.rebuilt = None

    def rebuild(self, batches):
        self.rebuilt = list(batches)
        return sum(len(ids) for _, ids in self.rebuilt)


def archive(settings, day, name, text, content=b"original"):
    folder = settings.processed_dir / day
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(content)
    (folder / f"{name}.txt").write_text(text, encoding="utf-8")
    return folder / name


def make_pipeline(settings, topic_reply="New topic"):
    store = FakeStore()
    topics = TopicNamer(FakeListChatModel(responses=[topic_reply] * 10), max_chars=100)
    return IngestionPipeline(settings, FileStore(settings), store, topics), store


def test_archive_round_trip():
    pages = [PageText(1, "First page.", "text"), PageText(2, "Second page.", "ocr")]
    parsed = parse_archive_text(format_archive_text("policy.pdf", "Leave policy", pages), "fallback.pdf")
    assert parsed.source == "policy.pdf"
    assert parsed.topic == "Leave policy"
    assert parsed.pages == pages


def test_parses_older_archives_without_header():
    parsed = parse_archive_text("--- page 1 ---\nOld text.", "scan.jpg")
    assert parsed.source == "scan.jpg"
    assert parsed.topic is None
    assert parsed.pages == [PageText(1, "Old text.", "ocr")]


def test_reindex_keeps_only_newest_version_of_a_file(settings):
    page = [PageText(1, "Leave is 24 days.", "text")]
    archive(settings, "2026-09-30", "policy.pdf", format_archive_text("policy.pdf", "Old", page), b"v1")
    newest = archive(settings, "2026-10-01", "policy.pdf", format_archive_text("policy.pdf", "New", page), b"v2")
    archive(settings, "2026-10-01", "trips.docx", format_archive_text("trips.docx", "Trips", page))
    pipeline, store = make_pipeline(settings)

    result = pipeline.reindex()

    assert result.documents == 2
    assert result.skipped_old_versions == ["processed/2026-09-30/policy.pdf"]
    by_source = {docs[0].metadata["source"]: docs for docs, _ in store.rebuilt}
    assert by_source["policy.pdf"][0].metadata["topic"] == "New"
    assert by_source["policy.pdf"][0].metadata["doc_id"] == sha256_of(newest)


def test_reindex_can_regenerate_topics(settings):
    archive(settings, "2026-10-01", "a.pdf", format_archive_text("a.pdf", "old lowercase", [PageText(1, "Text.")]))
    pipeline, store = make_pipeline(settings, topic_reply="перелік обладнання")

    pipeline.reindex(regenerate_topics=True)

    assert store.rebuilt[0][0][0].metadata["topic"] == "Перелік обладнання"


def test_broken_archive_leaves_index_untouched(settings):
    archive(settings, "2026-10-01", "good.pdf", format_archive_text("good.pdf", "Good", [PageText(1, "Text.")]))
    archive(settings, "2026-10-01", "empty.pdf", "Source: empty.pdf\nTopic: Empty\n\n")
    pipeline, store = make_pipeline(settings)

    with pytest.raises(ReindexError) as exc:
        pipeline.reindex()

    assert exc.value.errors[0]["archive"] == "processed/2026-10-01/empty.pdf.txt"
    assert store.rebuilt is None  # all or nothing
