"""Text extraction for every supported format: digital PDF, scanned PDF, images and Word.

Digital text is always preferred over OCR: it is exact and hundreds of times
faster. OCR is used only where there is no text layer, decided page by page,
so a PDF that mixes typed pages and scanned attachments is handled correctly.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import Settings
from .ocr import OCRError, normalise_text, ocr_image_file, ocr_pdf_page

logger = logging.getLogger(__name__)

# Word documents have no fixed pages (layout depends on the printer and font),
# so their text is stored as a single "page" with this number and cited without a page.
NO_PAGE = 0


class ExtractionError(Exception):
    pass


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    method: Literal["text", "ocr"] = "text"


def extract_pages(path: Path, settings: Settings) -> list[PageText]:
    """Return the text of every non-empty page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw_pages = _extract_pdf(path, settings)
    elif suffix == ".docx":
        raw_pages = _extract_docx(path)
    else:
        raw_pages = [PageText(n, text, "ocr") for n, text in ocr_image_file(path, settings)]

    pages: list[PageText] = []
    for page in raw_pages:
        if page.text.strip():
            pages.append(page)
        else:
            logger.warning("%s: page %d produced no text", path.name, page.page_number)

    if not pages:
        raise ExtractionError(f"No text found in {path.name}")
    ocr_count = sum(p.method == "ocr" for p in pages)
    logger.info("%s: %d page(s) with text, %d of them via OCR", path.name, len(pages), ocr_count)
    return pages


def _extract_pdf(path: Path, settings: Settings) -> list[PageText]:
    # Imported here so unit tests that never touch PDFs do not need the package.
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(path)
        total = len(reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ExtractionError(f"Cannot read PDF {path.name}: {exc}") from exc

    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = normalise_text(page.extract_text() or "")
        except Exception:  # broken text layers raise all kinds of errors; OCR is the fallback
            logger.warning("%s: text layer of page %d is unreadable", path.name, number, exc_info=True)
            text = ""

        if len(text) >= settings.pdf_min_text_chars:
            pages.append(PageText(number, text, "text"))
            continue

        # A scanned page is just an image: little or no text layer.
        logger.debug("%s: page %d/%d has no text layer, running OCR", path.name, number, total)
        try:
            pages.append(PageText(number, ocr_pdf_page(path, number, settings), "ocr"))
        except OCRError as exc:
            raise ExtractionError(str(exc)) from exc
    return pages


def _extract_docx(path: Path) -> list[PageText]:
    from docx import Document as load_docx
    from docx.opc.exceptions import PackageNotFoundError
    from docx.table import Table

    try:
        document = load_docx(str(path))
    except (PackageNotFoundError, KeyError, ValueError) as exc:
        raise ExtractionError(f"Cannot read Word document {path.name}: {exc}") from exc

    blocks: list[str] = []
    # iter_inner_content() keeps paragraphs and tables in document order.
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                # Merged cells are repeated by python-docx; drop consecutive duplicates.
                unique = [c for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]
                if unique:
                    blocks.append(" | ".join(unique))
        elif block.text.strip():
            blocks.append(block.text.strip())

    return [PageText(NO_PAGE, normalise_text("\n\n".join(blocks)), "text")]
