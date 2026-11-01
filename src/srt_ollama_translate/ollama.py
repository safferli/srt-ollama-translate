"""Translate one subtitle cue through a local Ollama chat endpoint."""

from dataclasses import dataclass
import json
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .languages import Language

DEFAULT_MODEL = "hf.co/17slever17/translate-gemma-4-sub-e4b-GGUF:Q4_K_XL"
DEFAULT_URL = "http://localhost:11434/api/chat"
STYLES = ("neutral", "friendly", "official")
MAX_TOKENS = 256


@dataclass(frozen=True)
class OllamaSettings:
    model: str = DEFAULT_MODEL
    url: str = DEFAULT_URL
    style: str = "neutral"


def translate_cue(
    source_text: str,
    prior_source: list[str],
    prior_translated: list[str],
    source_language: Language,
    target_language: Language,
    settings: OllamaSettings,
    timeout: float = 600,
) -> str:
    system = (
        f"TASK: Translate {source_language.name} subtitles into {target_language.name}.\n"
        f"STYLE: {settings.style}.\n"
        "Translate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only. "
        "Preserve meaning, tone, slang, profanity, uncertainty, repetitions and incomplete speech. "
        "Keep the translation concise enough for the existing subtitle timing where possible. "
        "Return only the final translation without labels or commentary."
    )
    prompt_parts = []
    if prior_source:
        prompt_parts.extend(
            [
                "[PREVIOUS_SOURCE]\n" + "\n".join(prior_source),
                "[PREVIOUS_TRANSLATION]\n" + "\n".join(prior_translated),
            ]
        )
    prompt_parts.append("[CURRENT_SOURCE]\n" + source_text)
    body = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(prompt_parts)},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": MAX_TOKENS},
    }
    request = Request(
        settings.url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    answer = data.get("message", {}).get("content", "").strip()
    if data.get("done_reason") == "length":
        raise RuntimeError(f"Model output truncated at {MAX_TOKENS} tokens: {answer[:160]!r}")
    if not answer or "[CURRENT_SOURCE]" in answer:
        raise ValueError(f"Unexpected model response: {answer[:160]!r}")
    # A blank line inside a cue would end the SRT block early and corrupt the file.
    return re.sub(r"\n\s*\n", "\n", answer)
