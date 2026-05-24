"""better-transcribe: local speech-to-text that beats YouTube's auto-captions.

A cross-platform CLI + library around faster-whisper (large-v3 by default) with
hallucination guards, word-level timestamps, optional speaker diarization, and
multiple output formats. Input can be a local audio/video file or any URL that
yt-dlp can fetch; for YouTube videos that are blocked, it falls back to a
captions relay (clearly labelled as lower quality).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
