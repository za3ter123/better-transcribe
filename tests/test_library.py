"""Tests for the transcript library (save / search / metadata filter)."""

from __future__ import annotations

from bettertranscribe import library


def test_save_writes_markdown_with_frontmatter(tmp_path):
    path = library.save(
        "Level one is a folder of markdown files.",
        source="youtube:abc123",
        title="Four levels of memory",
        model="large-v3",
        lang="en",
        tags=["memory", "rag"],
        directory=tmp_path,
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "source: youtube:abc123" in text
    assert "tags: [memory, rag]" in text
    assert "Level one is a folder of markdown files." in text


def test_search_ranks_keyword_matches(tmp_path):
    library.save("Metadata filtering narrows a huge corpus down.",
                 source="vid:1", title="Metadata filtering", tags=["memory"], directory=tmp_path)
    library.save("Subagents run in parallel for large refactors.",
                 source="vid:2", title="Subagents", tags=["agents"], directory=tmp_path)

    results = library.search("metadata corpus", directory=tmp_path)
    assert results, "expected at least one match"
    assert results[0][0].title == "Metadata filtering"


def test_metadata_filter_by_tag(tmp_path):
    library.save("alpha content", source="vid:1", title="A", tags=["memory"], directory=tmp_path)
    library.save("beta content", source="vid:2", title="B", tags=["agents"], directory=tmp_path)

    results = library.search("content", tag="agents", directory=tmp_path)
    assert len(results) == 1
    assert results[0][0].title == "B"


def test_filter_by_source_substring(tmp_path):
    library.save("x", source="youtube:keepme", title="Keep", directory=tmp_path)
    library.save("x", source="local:drop.mp3", title="Drop", directory=tmp_path)

    results = library.search("x", source="youtube", directory=tmp_path)
    assert [n.title for n, _ in results] == ["Keep"]


def test_stats_counts_notes_and_tags(tmp_path):
    library.save("one", source="v1", title="One", tags=["a", "b"], directory=tmp_path)
    library.save("two", source="v2", title="Two", tags=["a"], directory=tmp_path)

    stats = library.stats(directory=tmp_path)
    assert stats["count"] == 2
    assert stats["tags"]["a"] == 2
    assert stats["tags"]["b"] == 1


def test_empty_library_returns_no_results(tmp_path):
    assert library.search("anything", directory=tmp_path) == []
    assert library.stats(directory=tmp_path)["count"] == 0
