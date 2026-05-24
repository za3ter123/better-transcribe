import json

from bettertranscribe.formats import (render, render_json, render_srt,
                                       render_txt, render_vtt)

SEGS = [
    {"start": 0.0, "end": 2.5, "text": "Hello there."},
    {"start": 2.5, "end": 5.0, "text": "General Kenobi."},
]


def test_render_txt_reflow():
    out = render_txt(SEGS)
    assert "Hello there. General Kenobi." in out


def test_render_txt_timestamps():
    out = render_txt(SEGS, timestamps=True)
    assert out.startswith("[00:00] Hello there.")
    assert "[00:02] General Kenobi." in out


def test_render_srt_structure():
    out = render_srt(SEGS)
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello there." in out
    assert "2\n00:00:02,500 --> 00:00:05,000\nGeneral Kenobi." in out


def test_render_vtt_header_and_dots():
    out = render_vtt(SEGS)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in out


def test_render_json_roundtrip():
    out = render_json(SEGS, meta={"model": "tiny"})
    data = json.loads(out)
    assert data["segment_count"] == 2
    assert data["meta"]["model"] == "tiny"
    assert data["segments"][0]["text"] == "Hello there."


def test_speaker_labels_in_txt():
    segs = [{"start": 0, "end": 1, "text": "Hi", "speaker": "SPEAKER_00"},
            {"start": 1, "end": 2, "text": "Yo", "speaker": "SPEAKER_01"}]
    out = render_txt(segs)
    assert "SPEAKER_00: Hi" in out
    assert "SPEAKER_01: Yo" in out


def test_render_dispatch():
    assert render(SEGS, "srt").startswith("1\n")
    assert render(SEGS, "vtt").startswith("WEBVTT")
    assert json.loads(render(SEGS, "json"))["segment_count"] == 2
