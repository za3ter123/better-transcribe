"""Input classification and YouTube ID/URL extraction.

Kept dependency-free so it can be unit-tested without faster-whisper, yt-dlp, or
any network access.
"""

from __future__ import annotations

import re
from pathlib import Path

# Canonical YouTube URL shapes (watch, youtu.be, embed, shorts, live, ...).
_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|"
    r"youtube\.com/v/|youtube\.com/shorts/|youtube\.com/live/)([a-zA-Z0-9_-]{11})"
)
# A bare 11-char ID.
_RAW_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")
# Any URL carrying a ?v=/&v= param (covers third-party transcript mirrors).
_VPARAM_RE = re.compile(r"[?&]v=([a-zA-Z0-9_-]{11})")


def extract_youtube_id(url_or_id: str) -> str | None:
    """Return the 11-char YouTube ID from a URL or raw ID, or None if not YouTube."""
    text = url_or_id.strip()
    if _RAW_ID_RE.match(text):
        return text
    for pattern in (_VIDEO_ID_RE, _VPARAM_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def is_url(source: str) -> bool:
    """True if the source looks like an http(s) URL."""
    return source.strip().lower().startswith(("http://", "https://"))


def is_local_file(source: str) -> bool:
    """True if the source resolves to an existing local file."""
    try:
        return Path(source).expanduser().is_file()
    except OSError:
        return False


def classify_source(source: str) -> str:
    """Classify an input into 'file', 'youtube', or 'url'.

    A bare 11-char token is treated as a YouTube ID. An existing path is a
    'file'. Anything else that is a URL is 'url'; if it is a YouTube URL it is
    'youtube' (so the captions fallback can apply).
    """
    if is_local_file(source):
        return "file"
    if extract_youtube_id(source) is not None:
        return "youtube"
    if is_url(source):
        return "url"
    # Last resort: a non-existent path / unknown token. Treat as file so the
    # caller raises a clear "no such file" rather than guessing it is a URL.
    return "file"
