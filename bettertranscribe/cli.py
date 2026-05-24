"""Command-line entry point for better-transcribe."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__
from .asr import transcribe
from .fetch import resolve
from .formats import render
from .postprocess import collapse_repetitions

# Windows consoles default to cp1252 and choke on em-dashes / non-Latin text.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="betterscribe",
        description="Local speech-to-text that beats YouTube auto-captions.",
    )
    p.add_argument("source", help="Local audio/video file, a URL, or a YouTube ID/URL")
    p.add_argument("-o", "--out", help="Write to FILE instead of stdout")
    p.add_argument("-f", "--format", default="text",
                   choices=["text", "srt", "vtt", "json"], help="Output format")
    p.add_argument("-t", "--timestamps", action="store_true",
                   help="Prefix each line with a timestamp (text format)")
    p.add_argument("--model", default="large-v3",
                   help="Whisper model (default large-v3; smaller=faster: medium/small/base/tiny)")
    p.add_argument("--lang", default="en",
                   help="Spoken language (default en; use 'auto' to detect)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "mps"], help="Compute device")
    p.add_argument("--beam-size", type=int, default=5, help="Beam search width")
    p.add_argument("--no-reflow", action="store_true",
                   help="Don't reflow text into paragraphs (one line per segment)")
    p.add_argument("--diarize", action="store_true",
                   help="Label speakers (needs the [diarize] extra + a HF token)")
    p.add_argument("--hf-token", help="HuggingFace token for diarization")
    p.add_argument("--cookies-from-browser",
                   help="Browser to pull cookies from for gated URLs (chrome/edge/firefox/...)")
    p.add_argument("--no-captions-fallback", action="store_true",
                   help="Don't fall back to the captions relay when YT audio is blocked")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _emit(text: str, out: str | None, what: str) -> None:
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        print(f"Saved {what} to {out}", file=sys.stderr)
    else:
        print(text)


def _handle_captions(text: str, args: argparse.Namespace, label: str) -> None:
    """Captions fallback has no real timestamps -- emit text or wrap as JSON."""
    if args.format in ("srt", "vtt"):
        print("Captions fallback has no timestamps; emitting plain text instead.",
              file=sys.stderr)
    if args.format == "json":
        payload = json.dumps({"meta": {"source": label, "kind": "captions"},
                              "captions_text": text}, ensure_ascii=False, indent=2)
        _emit(payload, args.out, "captions JSON")
    else:
        _emit(text, args.out, "captions transcript")


def run(args: argparse.Namespace) -> int:
    lang = None if args.lang.lower() == "auto" else args.lang
    with tempfile.TemporaryDirectory(prefix="betterscribe-") as td:
        result = resolve(
            args.source, Path(td),
            cookies_from_browser=args.cookies_from_browser,
            captions_fallback=not args.no_captions_fallback,
        )
        if result.kind == "captions":
            _handle_captions(result.captions_text or "", args, result.source_label)
            return 0

        audio_path = str(result.audio_path)
        print(f"Transcribing {result.source_label} ...", file=sys.stderr)
        segments, meta = transcribe(
            audio_path, model_name=args.model, lang=lang,
            device=args.device, beam_size=args.beam_size,
        )
        segments = collapse_repetitions(segments)

        if args.diarize:
            from .diarize import DiarizationUnavailable, assign_speakers, diarize_audio
            try:
                turns = diarize_audio(audio_path, hf_token=args.hf_token)
                segments = assign_speakers(segments, turns)
                meta["diarized"] = True
            except DiarizationUnavailable as exc:
                print(f"Diarization skipped: {exc}", file=sys.stderr)

    if not any(s.get("text", "").strip() for s in segments):
        print("Error: no speech recognized (silent/empty audio?).", file=sys.stderr)
        return 1

    text = render(segments, args.format, timestamps=args.timestamps,
                  reflow=not args.no_reflow, meta=meta)
    _emit(text, args.out, f"{args.format} transcript ({len(segments)} segments)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
