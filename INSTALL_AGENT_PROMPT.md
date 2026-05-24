# Install with your AI coding agent

Don't want to install by hand? Paste the prompt below into **Claude Code**,
**Cursor**, **Codex**, or any terminal-capable AI agent. It will clone, install,
verify, and report back.

---

## Copy-paste prompt

```
Install the "better-transcribe" CLI for me and verify it works. Steps:

1. Check prerequisites and tell me if any are missing before continuing:
   - Python 3.9+  (`python --version`)
   - ffmpeg on PATH  (`ffmpeg -version`). If missing, install it:
       Windows: `winget install Gyan.FFmpeg`
       macOS:   `brew install ffmpeg`
       Linux:   `sudo apt-get install -y ffmpeg`

2. Clone and install:
       git clone https://github.com/za3ter123/better-transcribe.git
       cd better-transcribe
       pip install .

3. Pick the right acceleration extra for THIS machine and install it too:
   - NVIDIA GPU (check `nvidia-smi`):           pip install ".[cuda]"
   - Apple Silicon (macOS arm64):               pip install ".[mlx]"
   - CPU only:                                  nothing extra needed

4. Verify the install with the project's own end-to-end test, which
   synthesizes speech with the OS text-to-speech engine and transcribes it:
       pip install ".[dev]"
       python -m pytest -q                 # pure-logic suite, must pass
       RUN_ASR=1 python -m pytest tests/test_e2e_asr.py -s   # real ASR proof

5. Run a real transcription to confirm the CLI entry point works. If I gave you
   an audio file or URL, use it; otherwise transcribe any short local audio
   file you can find, e.g.:
       betterscribe <file-or-url> --model tiny --device cpu

6. Report: Python/ffmpeg versions, which acceleration extra you installed, the
   test results, and the transcript you produced. If anything failed, show me
   the exact error and your fix.

Notes:
- Default model is large-v3 (best quality, ~1.5 GB download, needs a GPU or
  patience on CPU). Use --model small or --model base on CPU-only machines.
- For YouTube URLs that are blocked, the tool falls back to a captions relay
  automatically; for gated videos pass --cookies-from-browser chrome.
- Speaker labels need: pip install ".[diarize]" plus a HuggingFace token
  (HUGGINGFACE_TOKEN) after accepting the pyannote/speaker-diarization-3.1
  model terms on huggingface.co.
```

---

## What the agent will end up running (manual equivalent)

```bash
git clone https://github.com/za3ter123/better-transcribe.git
cd better-transcribe
pip install .                      # + ".[cuda]" or ".[mlx]" for acceleration
pip install ".[dev]" && pytest -q  # verify
betterscribe path/to/audio.mp3     # use it
```
