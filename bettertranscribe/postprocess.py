"""Transcript clean-up: hallucination/repetition removal, paragraph reflow, WER.

All functions are pure and dependency-free so they can be unit-tested without a
model. A "segment" is a dict: {"start": float, "end": float, "text": str,
optionally "words": list}.
"""

from __future__ import annotations

import re

Segment = dict


def _norm(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace -- for comparisons only."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedup_within(text: str, max_run: int = 3) -> str:
    """Collapse an immediately-repeated token/phrase loop inside one segment.

    Whisper sometimes loops on music or silence: "okay okay okay okay ...".
    Keep at most `max_run` consecutive identical words.
    """
    words = text.split()
    if not words:
        return text
    out: list[str] = []
    run = 0
    for word in words:
        if out and _norm(word) == _norm(out[-1]):
            run += 1
            if run >= max_run:
                continue
        else:
            run = 0
        out.append(word)
    return " ".join(out)


def collapse_repetitions(segments: list[Segment], max_repeat: int = 2) -> list[Segment]:
    """Drop consecutive segments whose normalized text repeats (loop hallucination).

    Keeps at most `max_repeat` identical-in-a-row segments, then de-loops the
    text inside each remaining segment.
    """
    cleaned: list[Segment] = []
    prev_norm: str | None = None
    repeat = 0
    for seg in segments:
        norm = _norm(seg.get("text", ""))
        if not norm:
            continue
        if norm == prev_norm:
            repeat += 1
            if repeat >= max_repeat:
                continue
        else:
            repeat = 0
            prev_norm = norm
        new_seg = dict(seg)
        new_seg["text"] = dedup_within(seg.get("text", "").strip())
        cleaned.append(new_seg)
    return cleaned


def reflow_paragraphs(segments: list[Segment], max_chars: int = 600) -> str:
    """Join segment texts into readable paragraphs broken at sentence ends.

    Never alters words -- only joins lines and starts a new paragraph after a
    sentence-ending punctuation once a paragraph grows past `max_chars`.
    """
    sentence_end = re.compile(r"[.!?][\"')\]]?$")
    paragraphs: list[str] = []
    current: list[str] = []
    length = 0
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        current.append(text)
        length += len(text) + 1
        if length >= max_chars and sentence_end.search(text):
            paragraphs.append(" ".join(current))
            current = []
            length = 0
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance between two token lists (word-level)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate of `hypothesis` against `reference` (0.0 = identical).

    Both strings are normalized (case/punctuation-insensitive) first. Returns
    edit_distance / len(reference_words); an empty reference yields 0.0 when the
    hypothesis is also empty, else 1.0.
    """
    ref = _norm(reference).split()
    hyp = _norm(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)
