"""OCR for scanned, machine-printed pages (PDF pages and raster images).

PDF pages are rendered one at a time, so memory use stays flat even for
documents with hundreds of pages.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from .config import Settings


class OCRError(Exception):
    pass


def ocr_pdf_page(path: Path, number: int, settings: Settings) -> str:
    """Render one PDF page to an image and recognise its text."""
    try:
        images = convert_from_path(str(path), dpi=settings.ocr_dpi, first_page=number, last_page=number, grayscale=True)
    except Exception as exc:  # pdf2image wraps poppler failures in many exception types
        raise OCRError(f"Cannot render page {number} of {path.name}: {exc}") from exc
    return "\n\n".join(_ocr_image(image, settings) for image in images)


def ocr_image_file(path: Path, settings: Settings) -> Iterator[tuple[int, str]]:
    """Yield (page number, text) for every frame of an image file."""
    try:
        with Image.open(path) as img:
            # Multi-page TIFF is a common scanner output; each frame is a page.
            for number, frame in enumerate(ImageSequence.Iterator(img), start=1):
                yield number, _ocr_image(frame.copy(), settings)
    except (UnidentifiedImageError, OSError) as exc:
        raise OCRError(f"Cannot open image {path.name}: {exc}") from exc


def _preprocess(image: Image.Image) -> Image.Image:
    # Respect camera/scanner rotation, convert to grayscale and stretch contrast.
    # Cheap steps that noticeably improve Tesseract accuracy on faded scans.
    image = ImageOps.exif_transpose(image)
    return ImageOps.autocontrast(image.convert("L"))


def _ocr_image(image: Image.Image, settings: Settings) -> str:
    try:
        raw = pytesseract.image_to_string(
            _preprocess(image),
            lang=settings.ocr_languages,
            config=settings.tesseract_config,
            timeout=settings.ocr_page_timeout,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError("Tesseract binary is not installed") from exc
    except (pytesseract.TesseractError, RuntimeError) as exc:  # RuntimeError = timeout
        raise OCRError(f"Tesseract failed: {exc}") from exc
    return normalise_text(raw)


def normalise_text(text: str) -> str:
    """Clean typical OCR and PDF extraction artefacts without changing the meaning of the text."""
    text = text.replace("\u00ad", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split by line-end hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
