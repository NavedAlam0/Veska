"""Multi-modal support for Veska."""

from veska.media.types import Image, PDF, Audio
from veska.media.processor import process_attachments

__all__ = [
    "Image",
    "PDF",
    "Audio",
    "process_attachments",
]
