"""Minimal SRT reader and writer."""

import os
from pathlib import Path
import re
from typing import NamedTuple

# Cues are separated by one or more blank (or whitespace-only) lines.
CUE_SEPARATOR = re.compile(r"\n(?:[ \t]*\n)+")


class Cue(NamedTuple):
    index: str
    timestamp: str
    text: str


def read_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        raise ValueError(f"Empty subtitle file: {path}")
    cues = []
    for num, block in enumerate(CUE_SEPARATOR.split(raw), 1):
        lines = block.split("\n")
        # A cue may legitimately have no text: index and timestamp only.
        if len(lines) < 2 or not lines[0].strip().isdigit() or "-->" not in lines[1]:
            raise ValueError(f"Invalid SRT block {num} in {path}: {block[:100]!r}")
        cues.append(Cue(lines[0].strip(), lines[1], "\n".join(lines[2:])))
    return cues


def write_srt(path: Path, original_cues: list[Cue], translations: list[str]) -> None:
    """Write cues with original indexes and timestamps, atomically and durably replacing `path`."""
    data = (
        "\n\n".join(
            f"{cue.index}\n{cue.timestamp}\n{text}" for cue, text in zip(original_cues, translations, strict=False)
        )
        + "\n"
    )
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
