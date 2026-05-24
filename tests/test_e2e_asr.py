"""Real end-to-end ASR test: synthesize known speech with the OS TTS, transcribe
it with the actual Whisper pipeline, and check the words come back.

Opt-in: set RUN_ASR=1 (it downloads a small model on first run and needs CPU
time). Skips cleanly when TTS or faster-whisper is unavailable. This is the
honest proof the pipeline works on whatever machine runs it.

    RUN_ASR=1 pytest tests/test_e2e_asr.py -s
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

SENTENCE = "the quick brown fox jumps over the lazy dog"
RUN = os.environ.get("RUN_ASR") == "1"
pytestmark = pytest.mark.skipif(not RUN, reason="set RUN_ASR=1 to run real ASR")


def _synthesize(text: str, out_wav: Path) -> bool:
    """Render `text` to a WAV via the OS TTS. Returns True on success."""
    system = platform.system()
    try:
        if system == "Windows":
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SetOutputToWaveFile('{out_wav}'); $s.Speak('{text}'); $s.Dispose()"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           check=True, capture_output=True)
        elif system == "Darwin":
            subprocess.run(["say", "-o", str(out_wav), "--file-format=WAVE",
                            "--data-format=LEI16@22050", text],
                           check=True, capture_output=True)
        else:  # Linux
            subprocess.run(["espeak-ng", "-w", str(out_wav), text],
                           check=True, capture_output=True)
        return out_wav.is_file() and out_wav.stat().st_size > 1000
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def test_pipeline_recovers_known_sentence(tmp_path):
    pytest.importorskip("faster_whisper")
    from bettertranscribe.asr import transcribe
    from bettertranscribe.formats import render_txt
    from bettertranscribe.postprocess import collapse_repetitions, wer

    wav = tmp_path / "speech.wav"
    if not _synthesize(SENTENCE, wav):
        pytest.skip("no OS TTS available to generate test audio")

    # tiny model on CPU keeps the test fast; force cpu so it never needs a GPU.
    segments, meta = transcribe(str(wav), model_name="tiny", lang="en", device="cpu")
    segments = collapse_repetitions(segments)
    text = render_txt(segments, reflow=False)
    print(f"\n[e2e] backend={meta['backend']} device={meta['device']} -> {text!r}",
          file=sys.stderr)

    score = wer(SENTENCE, text)
    assert score < 0.5, f"ASR too far from reference (WER={score:.2f}): {text!r}"
