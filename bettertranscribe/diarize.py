"""Optional speaker diarization (pyannote.audio).

The model call is gated behind an importable pyannote + a HuggingFace token, so
the main ASR path never hard-depends on it. The speaker-assignment logic is a
pure function (`assign_speakers`) and is unit-tested without any model.
"""

from __future__ import annotations

import os

from .postprocess import Segment


class DiarizationUnavailable(RuntimeError):
    """Raised when diarization is requested but its deps/token are missing."""


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the time overlap between [a_start,a_end] and [b_start,b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segments: list[Segment], turns: list[dict]) -> list[Segment]:
    """Label each segment with the speaker whose turn it overlaps most.

    `turns` is a list of {"start", "end", "speaker"}. Pure function -- given the
    same inputs it always returns the same labelled segments. Segments with no
    overlapping turn are left unlabelled.
    """
    labelled: list[Segment] = []
    for seg in segments:
        s_start = seg.get("start", 0.0)
        s_end = seg.get("end", s_start)
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            ov = _overlap(s_start, s_end, turn["start"], turn["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn["speaker"]
        new_seg = dict(seg)
        if best_speaker is not None:
            new_seg["speaker"] = best_speaker
        labelled.append(new_seg)
    return labelled


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Run pyannote speaker diarization. Returns [{"start","end","speaker"}].

    Raises DiarizationUnavailable with actionable guidance if pyannote is not
    installed or no HuggingFace token is available.
    """
    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationUnavailable(
            "Diarization needs pyannote.audio. Install with:\n"
            "    pip install 'better-transcribe[diarize]'\n"
            "and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        ) from exc
    if not token:
        raise DiarizationUnavailable(
            "Diarization needs a HuggingFace token. Set HUGGINGFACE_TOKEN or pass "
            "--hf-token. Get one at https://huggingface.co/settings/tokens after "
            "accepting the pyannote/speaker-diarization-3.1 model terms."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    diarization = pipeline(audio_path)
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
