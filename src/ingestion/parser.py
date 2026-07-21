"""
Veritas RAG — Document Parser

Primary text extraction using PyMuPDF (fitz).
Falls back to Tesseract OCR for pages with low character yield
(below OCR_FALLBACK_CHAR_THRESHOLD from config).

Image files (.png, .jpg, .tiff) go straight to Tesseract.

Returns per-page: text, page_number, extraction_method, char_count,
ingest_confidence.

Usage:
    from src.ingestion.parser import parse_document
    pages = parse_document("data/test_corpus/agents_guide.pdf")
    for p in pages:
        print(f"Page {p.page_number}: {p.char_count} chars via {p.extraction_method}")
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from src.config import settings
from src.observability.logger import get_logger
from src.schemas import RawPage

log = get_logger("ingestion.parser", stage="ingestion")

# Supported file extensions
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

# Tesseract path for Windows — set if not in PATH
_TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)


def _configure_tesseract() -> bool:
    """
    Configure pytesseract with the Tesseract binary path.
    Returns True if Tesseract is available, False otherwise.
    """
    try:
        import pytesseract

        if os.path.isfile(_TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
            return True
        # Try without explicit path (maybe it's in PATH)
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        log.warning(
            "tesseract_not_available",
            path=_TESSERACT_CMD,
            message="OCR fallback disabled — Tesseract not found",
        )
        return False


def _ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL Image. Returns extracted text."""
    import pytesseract

    if os.path.isfile(_TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    return pytesseract.image_to_string(image)


def _calculate_ocr_confidence(text: str, image_area: float) -> float:
    """
    Estimate OCR confidence based on character density.

    Higher char density relative to image area = higher confidence.
    Returns 0.3 (low) to 0.8 (high) for OCR'd text.
    """
    if not text or not text.strip():
        return 0.1

    char_count = len(text.strip())

    # Heuristic: chars per 1000 pixels of image area
    if image_area > 0:
        density = char_count / (image_area / 1000)
    else:
        density = 0

    # Map density to confidence range [0.3, 0.8]
    # < 0.5 density → 0.3 confidence
    # > 5.0 density → 0.8 confidence
    confidence = min(0.8, max(0.3, 0.3 + (density - 0.5) * 0.1))
    return round(confidence, 2)


def _parse_pdf(file_path: Path) -> list[RawPage]:
    """
    Extract text from a PDF using PyMuPDF.
    Falls back to Tesseract OCR for pages with too few characters.
    """
    pages: list[RawPage] = []
    tesseract_available = _configure_tesseract()

    doc = fitz.open(str(file_path))
    total_pages = len(doc)

    log.info(
        "pdf_opened",
        file=file_path.name,
        total_pages=total_pages,
    )

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").strip()
        char_count = len(text)
        method = "pymupdf"
        confidence = 1.0

        # Check if PyMuPDF extraction is sufficient
        if char_count < settings.OCR_FALLBACK_CHAR_THRESHOLD:
            if tesseract_available:
                log.info(
                    "ocr_fallback_triggered",
                    page=page_num + 1,
                    pymupdf_chars=char_count,
                    threshold=settings.OCR_FALLBACK_CHAR_THRESHOLD,
                )

                # Render page to image for OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                ocr_text = _ocr_image(img)
                if len(ocr_text.strip()) > char_count:
                    text = ocr_text.strip()
                    char_count = len(text)
                    method = "tesseract"
                    confidence = _calculate_ocr_confidence(
                        text, float(pix.width * pix.height)
                    )
            else:
                # No Tesseract — mark as low confidence
                confidence = 0.2
                log.warning(
                    "ocr_skipped_no_tesseract",
                    page=page_num + 1,
                    chars=char_count,
                )

        log.debug(
            "page_extracted",
            page=page_num + 1,
            method=method,
            chars=char_count,
            confidence=confidence,
        )

        pages.append(
            RawPage(
                text=text,
                page_number=page_num + 1,
                extraction_method=method,
                char_count=char_count,
                ingest_confidence=confidence,
            )
        )

    doc.close()

    log.info(
        "pdf_parsed",
        file=file_path.name,
        total_pages=total_pages,
        pages_extracted=len(pages),
        ocr_pages=sum(1 for p in pages if p.extraction_method == "tesseract"),
        avg_confidence=round(
            sum(p.ingest_confidence for p in pages) / max(len(pages), 1), 2
        ),
    )

    return pages


def _parse_image(file_path: Path) -> list[RawPage]:
    """
    Extract text from a standalone image file using Tesseract OCR.
    Returns a single-page list.
    """
    if not _configure_tesseract():
        log.error(
            "image_parse_failed",
            file=file_path.name,
            reason="Tesseract not available",
        )
        # Return empty page with very low confidence rather than crashing
        return [
            RawPage(
                text="[OCR UNAVAILABLE]",
                page_number=1,
                extraction_method="tesseract",
                char_count=0,
                ingest_confidence=0.1,
            )
        ]

    log.info("image_ocr_start", file=file_path.name)

    img = Image.open(str(file_path))
    text = _ocr_image(img)
    char_count = len(text.strip())
    image_area = float(img.width * img.height)
    confidence = _calculate_ocr_confidence(text, image_area)

    log.info(
        "image_parsed",
        file=file_path.name,
        chars=char_count,
        confidence=confidence,
        image_size=f"{img.width}x{img.height}",
    )

    return [
        RawPage(
            text=text.strip() if text.strip() else "[NO TEXT EXTRACTED]",
            page_number=1,
            extraction_method="tesseract",
            char_count=char_count,
            ingest_confidence=confidence,
        )
    ]


def parse_document(file_path: str | Path) -> list[RawPage]:
    """
    Parse a document file and extract text from all pages.

    Supports:
        - PDF files (.pdf): PyMuPDF with Tesseract OCR fallback
        - Image files (.png, .jpg, .tiff): Tesseract OCR

    Args:
        file_path: Path to the document file.

    Returns:
        List of RawPage objects, one per page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()

    if ext in PDF_EXTENSIONS:
        return _parse_pdf(path)
    elif ext in IMAGE_EXTENSIONS:
        return _parse_image(path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {PDF_EXTENSIONS | IMAGE_EXTENSIONS}"
        )
