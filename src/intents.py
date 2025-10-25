from typing import Optional
from .utils import norm_text, phrase_to_regex

INTENTS = {
    "mirror": ["mirror mode", "start mirroring", "mirror"],
    "idle":   ["idle mode", "go idle", "stop", "sleep", "standby", "cancel"],
}

INTENT_PATTERNS = {k: [phrase_to_regex(p) for p in vs] for k, vs in INTENTS.items()}
INTENT_PRIORITY  = ["idle", "mirror"] + [k for k in INTENTS if k not in ("idle","mirror")]
INTENTS_NORM_MAP = {norm_text(p): k for k, vs in INTENTS.items() for p in vs}

def match_intent_keyword(nt: str) -> Optional[str]:
    found = {label for label, pats in INTENT_PATTERNS.items() if any(p.search(nt) for p in pats)}
    if not found: return None
    for label in INTENT_PRIORITY:
        if label in found: return label
    return next(iter(found))
