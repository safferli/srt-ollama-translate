"""Batch translation of SRT files: filename rules, planning, checkpoints and resume."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import NamedTuple

from .languages import Language, language_from_tag
from .ollama import OllamaSettings, translate_cue
from .srt import read_srt, write_srt


class Job(NamedTuple):
    path: Path
    basename: str
    source: Language


@dataclass(frozen=True)
class TranslationOptions:
    target: Language
    ollama: OllamaSettings
    context: int = 5
    max_cues: int | None = None


def source_language_from_filename(path: Path) -> tuple[str, Language]:
    """Return (basename without language + .srt, validated source language)."""
    if path.suffix.lower() != ".srt":
        raise ValueError(f"Not an SRT: {path.name}")
    parts = path.name.rsplit(".", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"Missing language in {path.name!r}; rename it to e.g. 'episode.fr.srt'")
    name, language_tag, _srt = parts
    try:
        source = language_from_tag(language_tag)
    except ValueError as exc:
        raise ValueError(f"Invalid source language in {path.name!r}: {exc}") from exc
    return name, source


def output_path(job: Job, target: Language) -> Path:
    return job.path.with_name(f"{job.basename}.{target.tag}.srt")


def plan_jobs(paths: list[Path], target: Language, source: Language | None = None) -> tuple[list[Job], list[str]]:
    """Pick one source file per episode; return (jobs, skip notes).

    Every filename is validated up front so no Ollama request is sent for a bad batch.
    An episode (files sharing a basename) with several candidate sources needs `source`
    to pick one, since e.g. ep.fr.srt and an earlier ep.en.srt output both map to ep.de.srt.
    """
    episodes: dict[Path, list[Job]] = {}
    for path in paths:
        basename, language = source_language_from_filename(path)
        job = Job(path, basename, language)
        episodes.setdefault(output_path(job, target), []).append(job)

    jobs: list[Job] = []
    notes: list[str] = []
    for destination, candidates in episodes.items():
        candidates = [job for job in candidates if job.source.identity != target.identity]
        if not candidates:
            continue
        if destination.exists():
            notes.append(f"Skipping {destination.name}: already exists")
            continue
        if source is not None:
            candidates = [job for job in candidates if job.source.identity == source.identity]
            if not candidates:
                notes.append(f"Skipping {destination.name}: no {source.name} source")
                continue
        if len(candidates) > 1:
            names = ", ".join(job.path.name for job in candidates)
            hint = "" if source is not None else "; pass --from to pick one"
            raise ValueError(f"Several sources for {destination.name}: {names}{hint}")
        jobs.append(candidates[0])
    return jobs, notes


def check_checkpoint_settings(settings_path: Path, settings: OllamaSettings) -> None:
    """Refuse to resume a checkpoint made with a different model or style."""
    expected = {"model": settings.model, "style": settings.style}
    if not settings_path.exists():
        settings_path.write_text(json.dumps(expected), encoding="utf-8")
        return
    recorded = json.loads(settings_path.read_text(encoding="utf-8"))
    if recorded != expected:
        raise ValueError(
            f"Checkpoint was made with {recorded}, not {expected}; "
            f"rerun with the same --model/--style or delete {settings_path.name} and its .partial"
        )


def translate_file(job: Job, options: TranslationOptions) -> None:
    target = options.target
    output = output_path(job, target)
    if output.exists():
        print(f"Skipping {job.path.name}: {output.name} already exists", flush=True)
        return

    checkpoint = output.with_name(output.name + ".partial")
    settings_path = checkpoint.with_name(checkpoint.name + ".json")
    cues = read_srt(job.path)
    translated: list[str] = []
    if checkpoint.exists():
        previous = read_srt(checkpoint)
        if len(previous) > len(cues) or any(a[:2] != b[:2] for a, b in zip(previous, cues, strict=False)):
            raise ValueError(f"Checkpoint {checkpoint.name} does not match {job.path.name}")
        translated = [cue.text for cue in previous]
        print(
            f"Resuming {job.path.name} ({job.source.name} -> {target.name}) at {len(translated)}/{len(cues)}",
            flush=True,
        )
    else:
        settings_path.unlink(missing_ok=True)
        print(f"Translating {job.path.name} ({job.source.name} -> {target.name}; {len(cues)} cues)", flush=True)
    check_checkpoint_settings(settings_path, options.ollama)

    stop = len(cues) if options.max_cues is None else min(len(cues), options.max_cues)
    for i in range(len(translated), stop):
        if not cues[i].text.strip():
            translated.append("")
        else:
            start = max(0, i - options.context)
            context = [
                (cue.text, done)
                for cue, done in zip(cues[start:i], translated[start:i], strict=True)
                if cue.text.strip()
            ]
            translated.append(
                translate_cue(
                    cues[i].text,
                    [text for text, _ in context],
                    [done for _, done in context],
                    job.source,
                    target,
                    options.ollama,
                )
            )
        write_srt(checkpoint, cues, translated)
        if (i + 1) % 10 == 0 or i + 1 == stop:
            print(f"  {i + 1}/{len(cues)}", flush=True)

    if len(translated) == len(cues):
        os.replace(checkpoint, output)
        settings_path.unlink(missing_ok=True)
        print(f"Saved {output.name}", flush=True)
    else:
        print(f"Checkpoint: {checkpoint.name}; rerun to resume", flush=True)
