"""Resolve an input source to local audio, with a captions fallback.

- Local file -> used as-is (ffmpeg decodes any audio/video container).
- URL / YouTube ID -> downloaded with yt-dlp (client rotation + optional browser
  cookies + optional PO-token provider for YouTube's bot-gate).
- If a YouTube download is fully blocked, fall back to a server-side captions
  relay (kome.ai). Captions are clearly labelled as lower quality (they re-serve
  YouTube's own track) and carry no real timestamps.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .ids import classify_source, extract_youtube_id

# Player clients that often slip past YouTube's bot wall, best-first.
DEFAULT_CLIENTS = ("android", "tv_embedded", "web_safari", "mweb", "ios", "web")
KOME_ENDPOINT = "https://kome.ai/api/transcript"
_DEFAULT_POT_SCRIPT = Path(
    os.environ.get(
        "BGUTIL_POT_SCRIPT",
        Path.home() / "tools" / "bgutil-pot-provider" / "server" / "build"
        / "generate_once.js",
    )
)


@dataclass
class FetchResult:
    """Outcome of resolving an input. kind == 'audio' | 'captions'."""

    kind: str
    audio_path: Path | None = None
    captions_text: str | None = None
    source_label: str = ""


def _ytdlp_base(cookies_from_browser: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist"]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    return cmd


def download_audio(url: str, tmp: Path, cookies_from_browser: str | None = None,
                   clients: tuple[str, ...] = DEFAULT_CLIENTS) -> Path:
    """Download best audio with yt-dlp, trying each YouTube client until one works.

    Non-YouTube URLs ignore the client rotation (a single attempt). Returns the
    downloaded file path; raises RuntimeError with the last error if all fail.
    """
    is_youtube = extract_youtube_id(url) is not None
    attempt_clients = clients if is_youtube else (None,)
    last_err = ""
    for client in attempt_clients:
        for stale in tmp.iterdir():
            if stale.is_file():
                stale.unlink()
        cmd = _ytdlp_base(cookies_from_browser) + [
            "-f", "bestaudio/best", "--js-runtimes", "node",
            "-o", str(tmp / "%(id)s.%(ext)s"),
        ]
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
            if _DEFAULT_POT_SCRIPT.is_file():
                cmd += ["--extractor-args",
                        f"youtubepot-bgutilscript:script_path={_DEFAULT_POT_SCRIPT}"]
        cmd.append(url)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        files = [p for p in tmp.iterdir() if p.is_file()]
        if files:
            return max(files, key=lambda p: p.stat().st_size)
        last_err = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(f"yt-dlp could not download audio.\n{last_err}")


def fetch_captions_relay(video_id: str, attempts: int = 3) -> str:
    """Fetch YouTube captions via the kome.ai server-side relay (bypasses IP bans)."""
    payload = json.dumps({"video_id": video_id, "format": True}).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            KOME_ENDPOINT, data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = (body.get("transcript") or "").strip()
            if text:
                return text
            raise ValueError("empty transcript in relay response")
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_err = exc
            if attempt < attempts:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"captions relay failed after {attempts} attempts: {last_err}")


def resolve(source: str, tmp: Path, cookies_from_browser: str | None = None,
            captions_fallback: bool = True,
            clients: tuple[str, ...] = DEFAULT_CLIENTS) -> FetchResult:
    """Resolve `source` to a FetchResult (local audio, downloaded audio, or captions)."""
    kind = classify_source(source)

    if kind == "file":
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {source}")
        return FetchResult("audio", audio_path=path, source_label=f"file:{path.name}")

    video_id = extract_youtube_id(source)
    url = (f"https://www.youtube.com/watch?v={video_id}" if video_id else source)
    try:
        audio = download_audio(url, tmp, cookies_from_browser, clients)
        return FetchResult("audio", audio_path=audio, source_label=url)
    except RuntimeError as exc:
        if video_id and captions_fallback:
            print(f"Audio download blocked ({exc}). Falling back to captions "
                  f"relay (lower quality than real ASR).", file=sys.stderr)
            text = fetch_captions_relay(video_id)
            return FetchResult("captions", captions_text=text,
                               source_label=f"captions:{video_id}")
        raise
