"""Command-line entry point for better-transcribe."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__, library

# Heavy imports (faster-whisper via .asr) are deferred into run() so that the
# `recall` / `library` subcommands work without the ASR dependencies installed.

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
    lib = p.add_argument_group("transcript library")
    lib.add_argument("--save", action="store_true",
                     help="Remember this transcript in the searchable library (recall it later)")
    lib.add_argument("--title", help="Title for the saved transcript (default: the source)")
    lib.add_argument("--tags", help="Comma-separated tags for the saved transcript")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _save_to_library(text: str, args: argparse.Namespace, source_label: str) -> None:
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    path = library.save(
        text, source=source_label, title=args.title,
        model=getattr(args, "model", ""), lang=getattr(args, "lang", ""), tags=tags,
    )
    print(f"Saved to library: {path}", file=sys.stderr)


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
    from .asr import transcribe
    from .fetch import resolve
    from .formats import render
    from .postprocess import collapse_repetitions

    lang = None if args.lang.lower() == "auto" else args.lang
    with tempfile.TemporaryDirectory(prefix="betterscribe-") as td:
        result = resolve(
            args.source, Path(td),
            cookies_from_browser=args.cookies_from_browser,
            captions_fallback=not args.no_captions_fallback,
        )
        if result.kind == "captions":
            _handle_captions(result.captions_text or "", args, result.source_label)
            if args.save and result.captions_text:
                _save_to_library(result.captions_text, args, result.source_label)
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
    if args.save:
        lib_text = text if args.format == "text" else render(
            segments, "text", timestamps=False, reflow=True, meta=meta)
        _save_to_library(lib_text, args, result.source_label)
    _emit(text, args.out, f"{args.format} transcript ({len(segments)} segments)")
    return 0


def _cmd_recall(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="betterscribe recall",
        description="Search your saved transcripts and pull just the ones you need.",
    )
    p.add_argument("query", nargs="?", default="", help="Words to search for")
    p.add_argument("--tag", help="Only transcripts with this tag (comma-separated)")
    p.add_argument("--source", help="Only transcripts whose source contains this text")
    p.add_argument("--since", help="Only transcripts created on/after YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=8, help="Max results")
    p.add_argument("--full", action="store_true", help="Print the full text of the top match")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    a = p.parse_args(argv)
    results = library.search(a.query, tag=a.tag, source=a.source, since=a.since, limit=a.limit)
    if a.json:
        print(json.dumps(
            [{"id": n.id, "title": n.title, "source": n.source, "created": n.created,
              "score": s, "path": str(n.path)} for n, s in results],
            ensure_ascii=False, indent=2))
        return 0
    if not results:
        print("No matching transcripts. (Transcribe with --save to build the library.)")
        return 0
    if a.full:
        print(results[0][0].path.read_text(encoding="utf-8"))
        return 0
    for note, score in results:
        tags = " ".join("#" + t for t in note.tags)
        print(f"\n  {note.title}  ({note.created}{'  ' + tags if tags else ''})  [score {score}]")
        print(f"  {note.path}")
        print(f"    {library.snippet(note, a.query)}")
    print(f"\n{len(results)} transcript(s). Open a path or re-run with --full — pull, don't auto-load.")
    return 0


def _cmd_library(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="betterscribe library",
        description="Show what's in your transcript library.",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    a = p.parse_args(argv)
    s = library.stats()
    notes = s["notes"]  # type: ignore[index]
    if a.json:
        print(json.dumps(
            {"dir": s["dir"], "count": s["count"], "tags": s["tags"],
             "notes": [{"id": n.id, "title": n.title, "created": n.created} for n in notes]},
            ensure_ascii=False, indent=2))
        return 0
    print(f"library: {s['dir']}")
    print(f"  transcripts: {s['count']}")
    top = list(s["tags"].items())[:12]  # type: ignore[union-attr]
    print(f"  tags: {len(s['tags'])}  ({'  '.join(f'#{t}:{c}' for t, c in top) or '—'})")  # type: ignore[arg-type]
    for n in notes[-15:]:
        print(f"    {n.created}  {n.title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "recall":
        return _cmd_recall(argv[1:])
    if argv and argv[0] in ("library", "lib"):
        return _cmd_library(argv[1:])
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
