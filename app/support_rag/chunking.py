"""Split extracted text into overlapping chunks with traceable metadata.

Pages are split independently, so every chunk keeps an exact page number
that can be cited in answers.
"""

from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .extraction import PageText


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer paragraph, then line, then sentence boundaries before cutting words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_pages(
    pages: list[PageText],
    splitter: RecursiveCharacterTextSplitter,
    doc_id: str,
    source: str,
    topic: str,
) -> tuple[list[Document], list[str]]:
    """Return chunks and their deterministic ids.

    Ids are derived from the content hash ("<doc_id>:<index>"), so indexing
    the same document twice overwrites the same records instead of creating
    duplicates.
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []
    for page in pages:
        for text in splitter.split_text(page.text):
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        "topic": topic,
                        "page": page.page_number,
                        "chunk_index": len(documents),
                        "ingested_at": ingested_at,
                    },
                )
            )
    ids = [f"{doc_id}:{i:05d}" for i in range(len(documents))]
    return documents, ids
