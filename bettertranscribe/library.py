"""Transcript library — a searchable memory of everything you transcribe.

Saving a transcript writes one markdown note (frontmatter + text). You pull notes
back later with keyword + metadata search — just in time, when you actually need
them, never auto-injected. Plain files, zero extra dependencies.

    betterscribe <source> --save        # transcribe and remember it
    betterscribe recall "rag is dead"    # find it again later
    betterscribe library                 # what's in the library
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", re.S)
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "be", "in", "on",
    "for", "with", "how", "what", "why", "this", "that", "your", "you", "it",
    "as", "at", "by", "we", "i", "so", "if", "but",
}


def library_dir() -> Path:
    """Where transcripts are stored (override with $BETTERTRANSCRIBE_LIBRARY)."""
    env = os.environ.get("BETTERTRANSCRIBE_LIBRARY")
    base = Path(env) if env else Path.home() / ".better-transcribe" / "library"
    return base


@dataclass(frozen=True)
class Note:
    path: Path
    id: str
    title: str
    source: str
    created: str
    model: str
    lang: str
    tags: list[str] = field(default_factory=list)
    body: str = ""


def _slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "transcript"


def _parse_frontmatter(raw: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, object] = {}
    for line in match.group(1).splitlines():
        kv = re.match(r"^\s*([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not kv:
            continue
        key, value = kv.group(1), kv.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value.strip("\"'")
    return meta, match.group(2)


def _serialize(meta: dict[str, object], body: str) -> str:
    lines = ["---"]
    for key, value in meta.items():
        rendered = f"[{', '.join(value)}]" if isinstance(value, list) else value
        lines.append(f"{key}: {rendered}")
    lines += ["---", "", body.strip(), ""]
    return "\n".join(lines)


def save(
    text: str,
    *,
    source: str,
    title: str | None = None,
    model: str = "",
    lang: str = "",
    tags: list[str] | None = None,
    directory: Path | None = None,
) -> Path:
    """Write a transcript to the library as a markdown note. Returns its path."""
    directory = directory or library_dir()
    directory.mkdir(parents=True, exist_ok=True)
    title = title or source
    created = date.today().isoformat()
    note_id = f"{created}-{_slugify(title)}"
    meta: dict[str, object] = {
        "id": note_id,
        "title": title,
        "source": source,
        "created": created,
        "model": model,
        "lang": lang,
        "tags": tags or [],
    }
    path = directory / f"{note_id}.md"
    path.write_text(_serialize(meta, f"# {title}\n\n{text.strip()}\n"), encoding="utf-8")
    return path


def _load(directory: Path | None = None) -> list[Note]:
    directory = directory or library_dir()
    notes: list[Note] = []
    if not directory.exists():
        return notes
    for path in sorted(directory.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        title_match = re.search(r"^#\s+(.+)$", body, re.M)
        notes.append(
            Note(
                path=path,
                id=str(meta.get("id") or path.stem),
                title=str(meta.get("title") or (title_match.group(1) if title_match else path.stem)),
                source=str(meta.get("source") or ""),
                created=str(meta.get("created") or ""),
                model=str(meta.get("model") or ""),
                lang=str(meta.get("lang") or ""),
                tags=[str(t).lower() for t in tags],
                body=body,
            )
        )
    return notes


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower()) if t not in _STOP]


def search(
    query: str,
    *,
    tag: str | None = None,
    source: str | None = None,
    since: str | None = None,
    limit: int = 10,
    directory: Path | None = None,
) -> list[tuple[Note, int]]:
    """Keyword search + metadata filter over the library. Returns (note, score)."""
    notes = _load(directory)
    if tag:
        wanted = {t.strip().lstrip("#").lower() for t in tag.split(",")}
        notes = [n for n in notes if wanted & set(n.tags)]
    if source:
        notes = [n for n in notes if source.lower() in n.source.lower()]
    if since:
        notes = [n for n in notes if n.created and n.created >= since]

    terms = _tokenize(query)
    scored: list[tuple[Note, int]] = []
    for note in notes:
        haystack = f"{note.title} {note.title} {' '.join(note.tags)} {note.source} {note.body}".lower()
        score = sum(len(re.findall(r"\b" + re.escape(t), haystack)) for t in terms) if terms else 1
        if score > 0:
            scored.append((note, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


def snippet(note: Note, query: str, width: int = 160) -> str:
    body = re.sub(r"\s+", " ", note.body).strip()
    terms = _tokenize(query)
    if terms:
        idx = body.lower().find(terms[0])
        if idx > -1:
            start = max(0, idx - 40)
            return "…" + body[start:idx + width - 40].strip() + "…"
    return body[:width].strip() + ("…" if len(body) > width else "")


def stats(directory: Path | None = None) -> dict[str, object]:
    notes = _load(directory)
    tags: dict[str, int] = {}
    for note in notes:
        for tag in note.tags:
            tags[tag] = tags.get(tag, 0) + 1
    return {
        "dir": str(directory or library_dir()),
        "count": len(notes),
        "tags": dict(sorted(tags.items(), key=lambda kv: kv[1], reverse=True)),
        "notes": notes,
    }
