import json
from pathlib import Path
import re

_MAPPING_PATH = Path(__file__).resolve().parents[2] / "avatar_mapping.json"

def load_mapping():
    try:
        return json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"phrases": {}, "signs": {}}

def map_text_to_animations(text: str):
    """Return only explicitly supported animation IDs; unsupported words stay text-only."""
    mapping = load_mapping()
    phrases = mapping.get("phrases", {})
    signs = mapping.get("signs", {})
    normalized = re.sub(r"[^A-Z0-9 ]+", " ", text.upper())
    normalized = " ".join(normalized.split())

    animations = []
    remaining = normalized

    # Prefer longest matching predefined phrases.
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase in remaining:
            animations.append(phrases[phrase])
            remaining = remaining.replace(phrase, " ", 1)

    for word in remaining.split():
        if word in signs:
            animations.append(signs[word])

    # Preserve honest text fallback; do not claim unsupported sentences are signed.
    supported_ids = set(phrases.values()) | set(signs.values())
    animations = [item for item in animations if item in supported_ids]
    covered = set()
    for phrase, anim in phrases.items():
        if anim in animations:
            covered.update(phrase.split())
    for word in normalized.split():
        if word in signs and signs[word] in animations:
            covered.add(word)
    text_only = [word for word in normalized.split() if word not in covered]
    return animations, text_only
