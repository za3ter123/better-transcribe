from bettertranscribe.ids import (classify_source, extract_youtube_id, is_url,
                                  is_local_file)


def test_extract_from_watch_url():
    assert extract_youtube_id("https://www.youtube.com/watch?v=qGAuqbbj1Ls") == "qGAuqbbj1Ls"


def test_extract_from_short_url():
    assert extract_youtube_id("https://youtu.be/qGAuqbbj1Ls?t=10") == "qGAuqbbj1Ls"


def test_extract_from_shorts_and_live():
    assert extract_youtube_id("https://youtube.com/shorts/qGAuqbbj1Ls") == "qGAuqbbj1Ls"
    assert extract_youtube_id("https://youtube.com/live/qGAuqbbj1Ls") == "qGAuqbbj1Ls"


def test_extract_raw_id():
    assert extract_youtube_id("qGAuqbbj1Ls") == "qGAuqbbj1Ls"


def test_extract_none_for_non_youtube():
    assert extract_youtube_id("https://example.com/audio.mp3") is None
    assert extract_youtube_id("not a video") is None


def test_is_url():
    assert is_url("https://x.com/a")
    assert is_url("http://x.com")
    assert not is_url("file.mp3")


def test_is_local_file(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert is_local_file(str(f))
    assert not is_local_file(str(tmp_path / "missing.wav"))


def test_classify(tmp_path):
    f = tmp_path / "clip.mp3"
    f.write_bytes(b"x")
    assert classify_source(str(f)) == "file"
    assert classify_source("https://www.youtube.com/watch?v=qGAuqbbj1Ls") == "youtube"
    assert classify_source("qGAuqbbj1Ls") == "youtube"
    assert classify_source("https://example.com/a.mp3") == "url"
