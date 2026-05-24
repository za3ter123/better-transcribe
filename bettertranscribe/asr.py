"""Speech-to-text via faster-whisper (default) or mlx-whisper (Apple Silicon).

The faster-whisper call is tuned to beat YouTube auto-captions: large-v3 by
default, beam search, VAD, and explicit hallucination guards
(no_speech/log_prob/compression-ratio thresholds + hallucination-silence
trimming). Word-level timestamps are produced so SRT/VTT segmentation and
diarization alignment are accurate.
"""

from __future__ import annotations

import sys

from .device import Backend, resolve_backend
from .postprocess import Segment


def load_model(model_name: str, backend: Backend):
    """Load a faster-whisper WhisperModel, falling back to CPU on any GPU error."""
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model_name, device=backend.device,
                            compute_type=backend.compute_type)
    except Exception as exc:  # noqa: BLE001 - GPU init can fail many ways
        if backend.device == "cpu":
            raise
        print(f"[{backend.device}] unavailable ({exc}); falling back to CPU.",
              file=sys.stderr)
        return WhisperModel(model_name, device="cpu", compute_type="int8")


def _transcribe_faster(audio_path: str, model_name: str, lang: str | None,
                       backend: Backend, beam_size: int) -> tuple[list[Segment], dict]:
    model = load_model(model_name, backend)
    segments_iter, info = model.transcribe(
        audio_path,
        language=lang,            # None => auto-detect
        beam_size=beam_size,
        vad_filter=True,                       # skip silence: faster, fewer loops
        word_timestamps=True,
        condition_on_previous_text=True,
        no_speech_threshold=0.6,               # treat very-low-speech as silence
        log_prob_threshold=-1.0,               # drop low-confidence decodes
        compression_ratio_threshold=2.4,       # reject gibberish/repetition
        hallucination_silence_threshold=2.0,   # skip text over long silences
    )
    segments: list[Segment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": float(seg.start), "end": float(seg.end),
                         "text": text})
    meta = {
        "backend": backend.name,
        "device": backend.device,
        "model": model_name,
        "language": getattr(info, "language", lang),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
    }
    return segments, meta


def _transcribe_mlx(audio_path: str, model_name: str, lang: str | None
                    ) -> tuple[list[Segment], dict]:
    """Apple-Silicon path via mlx-whisper. UNTESTED on CI (no Mac available)."""
    import mlx_whisper

    repo = f"mlx-community/whisper-{model_name}"
    result = mlx_whisper.transcribe(
        audio_path, path_or_hf_repo=repo, language=lang, word_timestamps=True
    )
    segments: list[Segment] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            segments.append({"start": float(seg.get("start", 0.0)),
                             "end": float(seg.get("end", 0.0)), "text": text})
    meta = {"backend": "mlx", "device": "mps", "model": model_name,
            "language": result.get("language", lang)}
    return segments, meta


def transcribe(audio_path: str, model_name: str = "large-v3",
               lang: str | None = "en", device: str = "auto",
               beam_size: int = 5) -> tuple[list[Segment], dict]:
    """Transcribe an audio file. Returns (segments, meta).

    `lang=None` auto-detects the language. Backend/device are resolved from
    `device` ("auto"/"cuda"/"cpu"/"mps").
    """
    backend = resolve_backend(device)
    if backend.name == "mlx":
        return _transcribe_mlx(audio_path, model_name, lang)
    return _transcribe_faster(audio_path, model_name, lang, backend, beam_size)
