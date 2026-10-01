"""Format of the text archive saved next to every processed file ("<file>.txt").

The archive is both human-readable (open it to judge OCR quality) and
machine-readable: /reindex rebuilds the whole index from it without
running OCR again.

    Source: vacation-policy.pdf
    Topic: Annual leave policy

    --- page 1 (text) ---
    ...
    --- page 2 (ocr) ---
    ...
"""

import re
from dataclasses import dataclass

from .extraction import PageText

PAGE_HEADER = re.compile(r"^--- page (\d+)(?: \((text|ocr)\))? ---$", re.MULTILINE)


@dataclass(frozen=True)
class ArchivedText:
    source: str
    topic: str | None
    pages: list[PageText]


def format_archive_text(source: str, topic: str, pages: list[PageText]) -> str:
    header = f"Source: {source}\nTopic: {topic}\n\n"
    return header + "\n\n".join(f"--- page {p.page_number} ({p.method}) ---\n{p.text}" for p in pages)


def parse_archive_text(text: str, fallback_source: str) -> ArchivedText:
    """Parse an archive; tolerates older archives without the Source/Topic header."""
    first_page = PAGE_HEADER.search(text)
    header = text[: first_page.start()] if first_page else text
    fields = dict(re.findall(r"^(Source|Topic): (.+)$", header, re.MULTILINE))

    matches = list(PAGE_HEADER.finditer(text))
    pages = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            pages.append(PageText(int(match.group(1)), body, match.group(2) or "ocr"))

    return ArchivedText(fields.get("Source", fallback_source).strip(), fields.get("Topic"), pages)
