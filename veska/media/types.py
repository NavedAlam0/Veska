"""
Multi-modal content types.

Explicit path — for power users who want fine-grained control.
Simple path (just strings) is handled by the processor.
"""

from __future__ import annotations

from typing import Optional


class Image:
    """Image attachment with optional detail control."""

    def __init__(
        self,
        source: str,
        detail: str = "auto",  # "auto", "high", "low"
    ) -> None:
        self.source = source
        self.detail = detail
        self.type = "image"


class PDF:
    """PDF attachment with optional page selection."""

    def __init__(
        self,
        source: str,
        pages: Optional[list[int]] = None,
    ) -> None:
        self.source = source
        self.pages = pages
        self.type = "pdf"


class Audio:
    """Audio attachment with optional language hint. (V3 — placeholder)"""

    def __init__(
        self,
        source: str,
        language: Optional[str] = None,
    ) -> None:
        self.source = source
        self.language = language
        self.type = "audio"
