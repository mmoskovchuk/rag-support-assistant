from support_rag.chunking import build_splitter, split_pages
from support_rag.extraction import PageText
from support_rag.ocr import normalise_text


def test_chunks_keep_page_numbers_and_deterministic_ids():
    pages = [PageText(1, "Alpha. " * 100), PageText(2, "Beta. " * 100)]
    docs, ids = split_pages(pages, build_splitter(200, 20), doc_id="abc", source="policy.pdf", topic="Policy")

    assert len(docs) == len(ids) > 2
    assert {d.metadata["page"] for d in docs} == {1, 2}
    assert all(d.metadata["source"] == "policy.pdf" for d in docs)
    assert all(d.metadata["topic"] == "Policy" for d in docs)
    assert ids[0] == "abc:00000"
    assert len(set(ids)) == len(ids)

    _, ids_again = split_pages(pages, build_splitter(200, 20), doc_id="abc", source="policy.pdf", topic="Policy")
    assert ids == ids_again  # same content -> same ids -> upsert instead of duplicates


def test_normalise_text_fixes_ocr_artifacts():
    raw = "Від-\nпустка   надається\n\n\n\nна 24 дні.  "
    assert normalise_text(raw) == "Відпустка надається\n\nна 24 дні."
