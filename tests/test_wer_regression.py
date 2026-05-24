"""Regression guard on the WER metric using a real Whisper-vs-captions excerpt.

The fixtures are aligned openings of the same video: `whisper_excerpt.txt` (our
ASR) and `captions_excerpt.txt` (YouTube's own caption track). They cover the
same speech, so their WER should be small but non-zero (punctuation/casing are
normalized away; the residual is real word differences). This locks the metric
against regressions without needing a GPU or network.
"""

from pathlib import Path

from bettertranscribe.postprocess import wer

FIX = Path(__file__).parent / "fixtures"


def test_excerpts_exist_and_nonempty():
    for name in ("whisper_excerpt.txt", "captions_excerpt.txt"):
        text = (FIX / name).read_text(encoding="utf-8")
        assert len(text.split()) > 50


def test_whisper_vs_captions_wer_is_small():
    whisper = (FIX / "whisper_excerpt.txt").read_text(encoding="utf-8")
    captions = (FIX / "captions_excerpt.txt").read_text(encoding="utf-8")
    score = wer(whisper, captions)
    # Same speech: should agree closely. If this ever blows past 0.5 the metric
    # or the fixtures broke.
    assert 0.0 <= score < 0.5


def test_wer_is_symmetric_enough():
    a = (FIX / "whisper_excerpt.txt").read_text(encoding="utf-8")
    b = (FIX / "captions_excerpt.txt").read_text(encoding="utf-8")
    # Word edit distance is symmetric; normalization by ref length makes the two
    # directions differ only by denominator, so they stay in the same ballpark.
    assert abs(wer(a, b) - wer(b, a)) < 0.2
