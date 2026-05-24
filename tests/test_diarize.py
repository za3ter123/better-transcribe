from bettertranscribe.diarize import assign_speakers


def test_assign_by_max_overlap():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "first"},
        {"start": 3.0, "end": 5.0, "text": "second"},
    ]
    turns = [
        {"start": 0.0, "end": 2.5, "speaker": "A"},
        {"start": 2.5, "end": 6.0, "speaker": "B"},
    ]
    out = assign_speakers(segs, turns)
    assert out[0]["speaker"] == "A"
    assert out[1]["speaker"] == "B"


def test_unlabelled_when_no_overlap():
    segs = [{"start": 10.0, "end": 11.0, "text": "lonely"}]
    turns = [{"start": 0.0, "end": 5.0, "speaker": "A"}]
    out = assign_speakers(segs, turns)
    assert "speaker" not in out[0]


def test_picks_dominant_overlap():
    segs = [{"start": 0.0, "end": 10.0, "text": "spans both"}]
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": "A"},
        {"start": 3.0, "end": 10.0, "speaker": "B"},  # 7s > 3s
    ]
    assert assign_speakers(segs, turns)[0]["speaker"] == "B"


def test_does_not_mutate_input():
    segs = [{"start": 0.0, "end": 2.0, "text": "x"}]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}]
    assign_speakers(segs, turns)
    assert "speaker" not in segs[0]
