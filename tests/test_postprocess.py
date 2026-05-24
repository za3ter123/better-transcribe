from bettertranscribe.postprocess import (collapse_repetitions, dedup_within,
                                          reflow_paragraphs, wer)


def test_dedup_within_collapses_runs():
    # default max_run=3 keeps at most 3 of a long identical run
    assert dedup_within("okay okay okay okay okay") == "okay okay okay"
    # a tighter cap collapses harder
    assert dedup_within("okay okay okay okay okay", max_run=2) == "okay okay"
    assert dedup_within("the the cat") == "the the cat"  # short run kept


def test_dedup_within_keeps_normal_text():
    s = "this is a normal sentence with no loops"
    assert dedup_within(s) == s


def test_collapse_repetitions_drops_repeated_segments():
    segs = [
        {"start": 0, "end": 1, "text": "Thank you."},
        {"start": 1, "end": 2, "text": "Thank you."},
        {"start": 2, "end": 3, "text": "Thank you."},
        {"start": 3, "end": 4, "text": "Now the real content."},
    ]
    out = collapse_repetitions(segs, max_repeat=2)
    texts = [s["text"] for s in out]
    assert texts.count("Thank you.") == 2
    assert "Now the real content." in texts


def test_collapse_skips_empty():
    segs = [{"start": 0, "end": 1, "text": "  "}, {"start": 1, "end": 2, "text": "hi"}]
    out = collapse_repetitions(segs)
    assert [s["text"] for s in out] == ["hi"]


def test_reflow_paragraphs_breaks_on_sentences():
    segs = [{"start": i, "end": i + 1, "text": f"Sentence number {i}."}
            for i in range(40)]
    text = reflow_paragraphs(segs, max_chars=100)
    assert "\n\n" in text
    # every word preserved
    assert text.replace("\n\n", " ").split() == " ".join(
        s["text"] for s in segs).split()


def test_wer_identical_is_zero():
    assert wer("Hello world", "hello, world!") == 0.0


def test_wer_all_wrong_is_one():
    assert wer("alpha beta gamma", "x y z") == 1.0


def test_wer_partial():
    # one substitution out of four words
    assert abs(wer("the quick brown fox", "the quick brown dog") - 0.25) < 1e-9


def test_wer_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "stuff") == 1.0
