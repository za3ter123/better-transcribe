"""Render transcript segments into txt / srt / vtt / json.

Pure functions over segment dicts ({"start", "end", "text", optional
"speaker"}); no model or I/O dependency.
"""

from __future__ import annotations

import json

from .postprocess import Segment, reflow_paragraphs


def _clock(seconds: float, long_video: bool) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if long_video or h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _srt_clock(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _vtt_clock(seconds: float) -> str:
    return _srt_clock(seconds).replace(",", ".")


def _label(seg: Segment) -> str:
    speaker = seg.get("speaker")
    text = seg.get("text", "").strip()
    return f"{speaker}: {text}" if speaker else text


def render_txt(segments: list[Segment], timestamps: bool = False,
               reflow: bool = True) -> str:
    """Plain text. With timestamps -> one line per segment; else reflowed paragraphs."""
    if timestamps:
        long_video = bool(segments) and segments[-1].get("end", 0) >= 3600
        return "\n".join(
            f"[{_clock(s.get('start', 0), long_video)}] {_label(s)}"
            for s in segments if s.get("text", "").strip()
        )
    if any(s.get("speaker") for s in segments):
        # Speaker labels are meaningful per-line; don't reflow across speakers.
        return "\n".join(_label(s) for s in segments if s.get("text", "").strip())
    return reflow_paragraphs(segments) if reflow else "\n".join(
        s.get("text", "").strip() for s in segments if s.get("text", "").strip()
    )


def render_srt(segments: list[Segment]) -> str:
    blocks = []
    for idx, seg in enumerate((s for s in segments if s.get("text", "").strip()), 1):
        start = _srt_clock(seg.get("start", 0))
        end = _srt_clock(seg.get("end", seg.get("start", 0)))
        blocks.append(f"{idx}\n{start} --> {end}\n{_label(seg)}\n")
    return "\n".join(blocks)


def render_vtt(segments: list[Segment]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in (s for s in segments if s.get("text", "").strip()):
        start = _vtt_clock(seg.get("start", 0))
        end = _vtt_clock(seg.get("end", seg.get("start", 0)))
        blocks.append(f"{start} --> {end}\n{_label(seg)}\n")
    return "\n".join(blocks)


def render_json(segments: list[Segment], meta: dict | None = None) -> str:
    payload = {
        "meta": meta or {},
        "segment_count": sum(1 for s in segments if s.get("text", "").strip()),
        "segments": [s for s in segments if s.get("text", "").strip()],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render(segments: list[Segment], fmt: str, timestamps: bool = False,
           reflow: bool = True, meta: dict | None = None) -> str:
    """Dispatch to the requested format renderer."""
    if fmt == "srt":
        return render_srt(segments)
    if fmt == "vtt":
        return render_vtt(segments)
    if fmt == "json":
        return render_json(segments, meta)
    return render_txt(segments, timestamps=timestamps, reflow=reflow)
