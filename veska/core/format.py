"""
Markdown formatting cleanup for Veska.

Strips markdown syntax so output reads cleanly in terminals and plain text.
"""

from __future__ import annotations

import re


def strip_markdown(text: str) -> str:
    """Strip common markdown formatting from text."""
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)

    # Italic: *text* or _text_ (but not inside words like file_name)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)

    # Inline code: `text`
    text = re.sub(r'`(.+?)`', r'\1', text)

    # Headers: # text, ## text, etc.
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Bullet points: - text or * text (at line start)
    text = re.sub(r'^[\-\*]\s+', '• ', text, flags=re.MULTILINE)

    return text
