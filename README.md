# srt-ollama-translate

Translate language-tagged SRT subtitle files with a local [Ollama](https://ollama.com) model.

By default, `translate-srt` scans every `*.srt` file in the current folder (not subfolders),
translates those with a valid ISO 639 language suffix into English, and skips finished outputs.
Original subtitle files are never overwritten.

```
episode.fr.srt  ->  episode.en.srt
episode.pt-BR.srt  ->  episode.en.srt
```

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally (default `http://localhost:11434`) with the translation model pulled:

  ```bash
  ollama pull hf.co/17slever17/translate-gemma-4-sub-e4b-GGUF:Q4_K_XL
  ```

## Installation

From the repository root:

```bash
uv sync
```

To put `translate-srt` on your PATH so you can run it from any subtitle folder:

```bash
uv tool install .
```

## Usage

Run these in the folder that contains your SRT files:

```bash
translate-srt                       # all *.srt -> English
translate-srt --to de               # all *.srt -> German
translate-srt --from fr --to de     # only French sources -> German
translate-srt --max-cues 20         # first 20 cues per file (trial run)
translate-srt episode.fr.srt --max-cues 20
translate-srt 'show*.fr.srt' --to ja
translate-srt -h                    # full help
```

If you have not installed the tool, run it from the repository without changing folders:

```bash
uv run --project /path/to/srt-ollama-translate translate-srt --to de
```

### Filename rules

The language suffix is required and must be a valid ISO 639 code, optionally with a script or region:

| File | Result |
| --- | --- |
| `episode.fr.srt` | French (valid) |
| `episode.fra.srt` | French (valid) |
| `episode.pt-BR.srt` | Brazilian Portuguese (valid) |
| `episode.qsm.srt` | Error: unknown ISO language code |
| `episode.srt` | Error: missing language suffix |

All filenames are validated before any request is sent to Ollama. Existing paths are taken
literally, so names with brackets like `Show [1080p].fr.srt` work without escaping.

### Several sources for one episode

Files that share a basename are one episode. If an episode has more than one source (for example
`episode.fr.srt` and an earlier `episode.en.srt` output), `--to de` is ambiguous and the run stops
before translating anything. Pass `--from fr` to choose the source. With `--from`, episodes that have
no source in that language are skipped. Episodes with a single source never need `--from`.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--to`, `--target` | `en` | Target ISO language code (e.g. `de`, `ja`, `pt-BR`) |
| `--from`, `--source` | none | Source language; picks the source when an episode has several |
| `--model` | `hf.co/17slever17/translate-gemma-4-sub-e4b-GGUF:Q4_K_XL` | Ollama model ID |
| `--url` | `http://localhost:11434/api/chat` | Ollama chat API endpoint |
| `--context` | `10` | Previous source/translation pairs passed as context |
| `--num-ctx` | `4096` | Ollama context window in tokens; sent explicitly so the server default can't truncate the prompt |
| `--style` | `neutral` | One of `neutral`, `friendly`, `official` |
| `--max-cues` | all | Stop after N cues per file and keep a `.partial` checkpoint |

### Resuming

Translation writes progress to `<output>.partial` after every cue. Interrupted or
`--max-cues`-limited runs can be resumed by running the same command again without the limit.
The output file appears only once every cue is translated. The model and style used are recorded
in `<output>.partial.json`; resuming with a different `--model` or `--style` is refused, so one file
never mixes two settings.

## Model languages

The translation model's 11 most extensively trained languages (per the author's model card):

- **Tier 1:** English (en), Russian (ru), Spanish (es), German (de), Japanese (ja)
- **Tier 2:** French (fr), Portuguese (pt), Chinese (zh), Dutch (nl), Italian (it), Korean (ko)

These are training tiers, not a restriction: any valid ISO 639 code is accepted, but quality may
vary for languages outside these tiers.

## Development

```bash
uv sync
uv run pytest
ruff check && ruff format --check
```

The tests include an in-process fake Ollama server, so they do not need the real model.

## Project layout

```
src/srt_ollama_translate/
  cli.py        argument parsing and the translate-srt entry point
  translator.py filename rules, job planning, checkpoints and resume
  ollama.py     one-cue translation request to the Ollama chat API
  srt.py        SRT reading and atomic writing
  languages.py  ISO 639 / script / region tag validation (pycountry)
tests/
samples/        real-world SRT files for manual trial runs
```
