# better-transcribe

**Local speech-to-text that beats YouTube's auto-captions.**

YouTube's auto-captions have no punctuation, mangle proper nouns, and miss
words. `better-transcribe` listens to the actual audio with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`large-v3` by
default) and returns a clean, punctuated, correctly-capitalized transcript —
nothing leaves your machine and there's no per-use cost.

```
YouTube auto-caption :  cloud code is a tool by anthropic for d-chrome work
better-transcribe    :  Claude Code is a tool by Anthropic for --Chrome work.
```

## What it does

- **Real ASR, not caption scraping.** Transcribes the audio with Whisper
  `large-v3`, tuned to beat auto-captions: beam search, voice-activity
  detection, and explicit hallucination guards (no-speech / log-prob /
  compression-ratio thresholds + hallucination-silence trimming).
- **Any input.** A local audio/video file, a direct media URL, or a YouTube
  URL/ID. For YouTube videos whose audio is blocked, it falls back to a
  server-side captions relay (clearly labelled as lower quality).
- **Any output.** Plain text (reflowed into paragraphs), timestamped text,
  `.srt`, `.vtt`, or `.json`.
- **Optional speaker labels.** `--diarize` adds `SPEAKER_00:` style labels via
  pyannote (opt-in extra).
- **Cross-platform.** CPU everywhere; CUDA on NVIDIA (Windows/Linux); an
  Apple-Silicon backend via `mlx-whisper` when installed.

## Install

> Requires **Python 3.9+** and **ffmpeg** on your PATH.
> ([ffmpeg downloads](https://ffmpeg.org/download.html) — `winget install
> Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS,
> `apt install ffmpeg` on Debian/Ubuntu.)

```bash
git clone https://github.com/za3ter123/better-transcribe.git
cd better-transcribe
pip install .
```

Optional extras:

```bash
pip install ".[cuda]"      # NVIDIA GPU acceleration (Windows/Linux)
pip install ".[mlx]"       # Apple-Silicon acceleration
pip install ".[diarize]"   # speaker labels (also needs a HuggingFace token)
```

The first run downloads the chosen Whisper model once (~1.5 GB for `large-v3`,
~75 MB for `tiny`), then caches it.

## Usage

```bash
betterscribe path/to/audio.mp3                    # clean paragraphs to stdout
betterscribe lecture.m4a -o lecture.txt           # save to a file
betterscribe "https://youtu.be/VIDEO_ID"          # YouTube (audio -> Whisper)
betterscribe clip.wav -f srt -o clip.srt          # subtitles
betterscribe interview.wav --diarize              # speaker labels
betterscribe talk.mp4 --model tiny --device cpu   # fast, lower accuracy
betterscribe foreign.mp3 --lang auto              # auto-detect language
```

| Option | Effect |
|--------|--------|
| `-o, --out FILE` | Write to a file instead of stdout |
| `-f, --format` | `text` (default) / `srt` / `vtt` / `json` |
| `-t, --timestamps` | Prefix each line with a timestamp (text format) |
| `--model` | `large-v3` (default), `medium`, `small`, `base`, `tiny` |
| `--lang` | Spoken language (default `en`; `auto` to detect) |
| `--device` | `auto` (default) / `cuda` / `cpu` / `mps` |
| `--beam-size` | Beam search width (default 5) |
| `--no-reflow` | One line per segment instead of paragraphs |
| `--diarize` | Speaker labels (needs `[diarize]` extra + HF token) |
| `--hf-token` | HuggingFace token for diarization |
| `--cookies-from-browser` | Use a browser's cookies for gated URLs (`chrome`/`edge`/`firefox`/…) |
| `--no-captions-fallback` | Don't relay captions when YouTube audio is blocked |

## Speed

`large-v3` is ~5–10× faster than real time on an NVIDIA GPU, and slower than
real time on CPU — use `--model small` or `--model base` on CPU-only machines.
The Apple-Silicon `mlx` backend is much faster than CPU on M-series Macs.

## How it compares

| | YouTube auto-captions | caption scrapers | **better-transcribe** |
|---|---|---|---|
| Punctuation & casing | ✗ | ✗ (re-serves YT) | ✓ |
| Fixes mis-heard words | ✗ | ✗ | ✓ (large-v3) |
| Works on any file/URL | ✗ | ✗ | ✓ |
| Speaker labels | ✗ | ✗ | ✓ (`--diarize`) |
| Runs locally / private | ✗ | ✗ | ✓ |

## Development

```bash
pip install ".[dev]"
pytest                      # fast pure-logic suite (no GPU/network)
RUN_ASR=1 pytest -s         # also runs a real end-to-end ASR check via OS TTS
```

The end-to-end test synthesizes a known sentence with your OS's text-to-speech
(SAPI on Windows, `say` on macOS, `espeak-ng` on Linux), transcribes it, and
asserts the words come back — an honest proof the pipeline works on your
machine.

## License

MIT — see [LICENSE](LICENSE).
