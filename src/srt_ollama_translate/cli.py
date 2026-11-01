"""Command-line entry point: `translate-srt`."""

import argparse
import glob
from pathlib import Path
import sys
from urllib.error import URLError

from .languages import language_from_tag
from .ollama import DEFAULT_MODEL, DEFAULT_URL, STYLES, OllamaSettings
from .translator import TranslationOptions, plan_jobs, translate_file

DESCRIPTION = """\
Translate language-tagged SRT subtitle files through local Ollama.

By default, scans ALL *.srt files in the current working directory, translates
those with a valid ISO language suffix to English, and skips completed outputs.
For example, episode.fr.srt -> episode.en.srt. It does not scan subfolders.

Filename rules (strict):
  episode.fr.srt       French subtitle file (valid)
  episode.fra.srt      French subtitle file (valid)
  episode.pt-BR.srt    Brazilian Portuguese (valid)
  episode.qsm.srt      ERROR: unknown ISO language code
  episode.srt          ERROR: missing language suffix

Model's 11 most extensively trained languages (author's model card):
  Tier 1: English (en), Russian (ru), Spanish (es), German (de), Japanese (ja)
  Tier 2: French (fr), Portuguese (pt), Chinese (zh), Dutch (nl),
          Italian (it), Korean (ko)

These are the model's training tiers, NOT a restriction on accepted filenames:
any valid ISO 639 language code is allowed, but other languages are less
thoroughly trained/evaluated and translation quality may vary.

Several sources for one episode?
  If episode.fr.srt and episode.en.srt both exist, --to de is ambiguous: pass
  --from fr to pick the source. With --from, episodes without a source in that
  language are skipped. Episodes with a single source never need --from.

What counts as a cue?
  One numbered subtitle block with a timestamp and one or more lines of text.
  --max-cues 20 translates AT MOST the first 20 cues PER FILE for a trial run.
  Without --max-cues, translates every cue. Incomplete work is saved to
  *.partial checkpoints; rerunning without the limit resumes automatically.

Examples (run in the folder containing your SRT files):
  translate-srt                    # all *.srt -> English
  translate-srt --to de            # all *.srt -> German
  translate-srt --from fr --to de  # only French sources -> German
  translate-srt --max-cues 20      # 20 cues PER file
  translate-srt episode.fr.srt --max-cues 20
  translate-srt 'show*.fr.srt' --to ja
  translate-srt -h                 # this help

Uses the context-aware prompt format for 17slever17/translate-gemma-4-sub-e4b
via the local Ollama API. Original subtitle files are never overwritten.
"""


def expand(pattern: str) -> list[Path]:
    """Take existing paths literally, so names like 'Show [1080p].fr.srt' aren't read as globs."""
    if Path(pattern).exists():
        return [Path(pattern)]
    return [Path(filename) for filename in glob.glob(pattern)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate-srt",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="SRT paths or glob patterns; default: all *.srt in cwd (non-recursive). "
        "Every matched filename must have a valid language suffix.",
    )
    parser.add_argument(
        "--to",
        "--target",
        default="en",
        dest="target",
        help="Target ISO language code (default: en; e.g. de, ja, pt-BR)",
    )
    parser.add_argument(
        "--from",
        "--source",
        dest="source",
        help="Source ISO language code; picks the source when an episode has several (e.g. fr)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model ID")
    parser.add_argument("--url", default=DEFAULT_URL, help="Ollama chat API endpoint")
    parser.add_argument(
        "--context", type=int, default=5, help="Previous source/translation pairs to provide as context (default: 5)"
    )
    parser.add_argument("--style", choices=STYLES, default="neutral")
    parser.add_argument(
        "--max-cues",
        type=int,
        help="Stop after N subtitle cues PER FILE; retain .partial checkpoint to resume later (default: all cues)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context < 0 or (args.max_cues is not None and args.max_cues < 1):
        parser.error("--context must be >= 0 and --max-cues must be >= 1")
    try:
        target = language_from_tag(args.target)
    except ValueError as exc:
        parser.error(f"Invalid target language: {exc}")

    source = None
    if args.source is not None:
        try:
            source = language_from_tag(args.source)
        except ValueError as exc:
            parser.error(f"Invalid source language: {exc}")
        if source.identity == target.identity:
            parser.error("--from and --to are the same language")

    matched: set[Path] = set()
    for pattern in args.files or ["*.srt"]:
        matches = expand(pattern)
        if args.files and not matches:
            parser.error(f"No file matches {pattern!r}")
        matched.update(matches)
    paths = sorted(path for path in matched if path.is_file() and path.suffix.lower() == ".srt")
    if not paths:
        parser.error("No matching SRT files")

    try:
        jobs, notes = plan_jobs(paths, target, source)
    except ValueError as exc:
        parser.error(str(exc))
    for note in notes:
        print(note, flush=True)

    options = TranslationOptions(
        target=target,
        ollama=OllamaSettings(model=args.model, url=args.url, style=args.style),
        context=args.context,
        max_cues=args.max_cues,
    )
    for job in jobs:
        try:
            translate_file(job, options)
        except (OSError, ValueError, URLError, RuntimeError, KeyError) as exc:
            print(f"ERROR translating {job.path.name}: {exc}", file=sys.stderr, flush=True)
            print("Completed cues, if any, are retained in the .partial checkpoint.", file=sys.stderr)
            return 1
    return 0
