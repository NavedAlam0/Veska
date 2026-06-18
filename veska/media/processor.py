"""
Multi-modal attachment processor.

Handles both simple path (strings) and explicit path (Image/PDF objects).
Converts everything into provider-agnostic content blocks.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Union

from veska.media.types import Audio, Image, PDF

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Supported document extensions
PDF_EXTENSIONS = {".pdf"}

# Mime type mapping for images
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def process_attachments(
    attachments: list[Union[str, Image, PDF, Audio]],
) -> list[dict[str, Any]]:
    """
    Process a list of attachments into content blocks.

    Accepts:
      - Strings (auto-detect from extension or URL)
      - Image/PDF/Audio objects (explicit control)

    Returns list of content blocks:
      [{"type": "image", "source": ..., "detail": ...}, ...]
    """
    blocks = []

    for attachment in attachments:
        if isinstance(attachment, str):
            blocks.extend(_process_string(attachment))
        elif isinstance(attachment, Image):
            blocks.extend(_process_image(attachment))
        elif isinstance(attachment, PDF):
            blocks.extend(_process_pdf(attachment))
        elif isinstance(attachment, Audio):
            blocks.extend(_process_audio(attachment))

    return blocks


def _process_string(source: str) -> list[dict]:
    """Auto-detect file type from path/URL and process."""
    if _is_url(source):
        # URL — detect type from extension in URL
        ext = _url_extension(source)
        if ext in IMAGE_EXTENSIONS:
            return _process_image(Image(source))
        # Default: treat as image URL
        return _process_image(Image(source))

    # Local file
    path = Path(source)
    ext = path.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return _process_image(Image(source))
    elif ext in PDF_EXTENSIONS:
        return _process_pdf(PDF(source))
    else:
        # Unknown type — try as image
        return _process_image(Image(source))


def _process_image(image: Image) -> list[dict]:
    """Process an image into a content block."""
    source = image.source

    if _is_url(source):
        return [{
            "type": "image",
            "source_type": "url",
            "url": source,
            "media_type": _guess_mime(source),
            "detail": image.detail,
        }]

    # Local file — base64 encode
    path = Path(source)
    if not path.exists():
        return [{"type": "text", "text": f"[Error: File not found: {source}]"}]

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    media_type = MIME_TYPES.get(path.suffix.lower(), "image/png")

    return [{
        "type": "image",
        "source_type": "base64",
        "data": data,
        "media_type": media_type,
        "detail": image.detail,
    }]


def _process_pdf(pdf: PDF) -> list[dict]:
    """Process a PDF into text content blocks.

    Extracts text from the PDF. If specific pages are requested,
    only those pages are extracted.
    """
    source = pdf.source

    if _is_url(source):
        return [{"type": "text", "text": f"[PDF from URL: {source}]"}]

    path = Path(source)
    if not path.exists():
        return [{"type": "text", "text": f"[Error: File not found: {source}]"}]

    # Try to extract text from PDF
    text = _extract_pdf_text(path, pdf.pages)
    if text:
        return [{"type": "text", "text": f"[PDF Content from {path.name}]:\n{text}"}]

    # Fallback: send as base64 document (some providers support this)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return [{
        "type": "document",
        "source_type": "base64",
        "data": data,
        "media_type": "application/pdf",
    }]


def _process_audio(audio: Audio) -> list[dict]:
    """Process audio — placeholder for V3."""
    return [{"type": "text", "text": f"[Audio file: {audio.source} (not yet supported)]"}]


def _extract_pdf_text(path: Path, pages: list[int] | None = None) -> str | None:
    """Try to extract text from a PDF file."""
    # Try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        texts = []
        for i, page in enumerate(reader.pages):
            if pages and (i + 1) not in pages:
                continue
            text = page.extract_text()
            if text:
                texts.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(texts) if texts else None
    except ImportError:
        pass

    # Try pymupdf
    try:
        import fitz
        doc = fitz.open(str(path))
        texts = []
        for i, page in enumerate(doc):
            if pages and (i + 1) not in pages:
                continue
            text = page.get_text()
            if text:
                texts.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(texts) if texts else None
    except ImportError:
        pass

    return None


def _is_url(source: str) -> bool:
    """Check if a source string is a URL."""
    return source.startswith(("http://", "https://"))


def _url_extension(url: str) -> str:
    """Extract file extension from a URL."""
    path = url.split("?")[0].split("#")[0]
    if "." in path.split("/")[-1]:
        return "." + path.split(".")[-1].lower()
    return ""


def _guess_mime(source: str) -> str:
    """Guess MIME type from source path/URL."""
    ext = Path(source).suffix.lower() if not _is_url(source) else _url_extension(source)
    return MIME_TYPES.get(ext, "image/png")
