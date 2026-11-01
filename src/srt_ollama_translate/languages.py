"""ISO 639 language tag validation with optional script and region qualifiers."""

from dataclasses import dataclass
import re

import pycountry

LANGUAGE_TAG = re.compile(r"^([a-zA-Z]{2,3})(?:-([a-zA-Z]{4}))?(?:-([a-zA-Z]{2}|\d{3}))?$")


@dataclass(frozen=True)
class Language:
    tag: str
    name: str
    identity: tuple[str, str | None, str | None]


def language_from_tag(tag: str) -> Language:
    """Validate ISO 639 code with optional BCP-47 script/region qualifiers."""
    match = LANGUAGE_TAG.fullmatch(tag)
    if not match:
        raise ValueError(f"Unknown language code {tag!r}. Expected an ISO 639 code (e.g. fr, fra, de, pt-BR).")
    base, script, region = match.groups()
    base = base.lower()
    language = (
        pycountry.languages.get(alpha_2=base)
        or pycountry.languages.get(alpha_3=base)
        or pycountry.languages.get(bibliographic=base)
    )
    if language is None:
        raise ValueError(f"Unknown ISO 639 language code {tag!r}.")
    # Special codes such as und, mul and zxx are valid ISO 639 but name no language.
    if getattr(language, "type", None) == "S":
        raise ValueError(f"{tag!r} ({language.name}) is not a specific language.")

    parts = [getattr(language, "alpha_2", language.alpha_3)]
    label = language.name
    if script:
        script_entry = pycountry.scripts.get(alpha_4=script.title())
        if script_entry is None:
            raise ValueError(f"Unknown language script {script!r} in {tag!r}")
        script = script.title()
        parts.append(script)
        label += f" ({script_entry.name})"
    if region:
        region = region.upper()
        country = (
            pycountry.countries.get(alpha_2=region) if len(region) == 2 else pycountry.countries.get(numeric=region)
        )
        if country is None:
            raise ValueError(f"Unknown region {region!r} in language tag {tag!r}")
        parts.append(region)
        label += f" ({country.name})"
    return Language("-".join(parts), label, (language.alpha_3, script, region))
