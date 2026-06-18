"""Small, dependency-free text helpers shared by judges and synthetic bots."""
from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "for", "on", "in", "at", "by", "with", "and", "or", "but", "if",
    "this", "that", "these", "those", "it", "its", "as", "from", "what", "which",
    "who", "whom", "when", "where", "why", "how", "do", "does", "did", "can",
    "could", "will", "would", "should", "may", "might", "i", "you", "we", "they",
    "your", "our", "their", "not", "no", "yes",
}

_WORD = re.compile(r"[a-z0-9]+")

REFUSAL_MARKERS = (
    "not in sources",
    "not in the sources",
    "not found in sources",
    "not found in the sources",
    "no information",
    "does not contain",
    "doesn't contain",
    "not covered",
    "cannot find",
    "can't find",
    "i don't know",
    "i do not know",
    "unable to answer",
    "no relevant information",
)


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower()) if t not in _STOPWORDS]


def content_token_overlap(answer: str, context: str) -> float:
    """Fraction of an answer's content tokens that also appear in the context.

    A cheap, transparent proxy for groundedness: a hallucinated answer invents
    content that is absent from the supporting context, so its overlap is low.
    Swap this for an NLI entailment model (see judges/groundedness.py) for the
    real thing -- the interface is identical.
    """
    ans = set(tokens(answer))
    if not ans:
        return 0.0
    ctx = set(tokens(context or ""))
    if not ctx:
        return 0.0
    return len(ans & ctx) / len(ans)


def looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in REFUSAL_MARKERS)


def split_explanation(output: str) -> str:
    """Return the explanation portion of a response (text before 'Answer:')."""
    if not output:
        return ""
    m = re.search(r"answer\s*[:\-]", output, flags=re.IGNORECASE)
    return output[: m.start()] if m else output


def last_number(text: str) -> str | None:
    """The last numeric token in a string, or None. Used for consistency checks."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


def normalize_answer(text: str) -> str:
    """Pull a short final answer out of a response and normalize it.

    Scope: short-form answers -- numbers, single words, booleans (yes/no/true/
    false), and multiple-choice letters. Strategy:

      * if an explicit ``Answer: X`` marker is present, take the FIRST token of
        X (so "Answer: B because 2+2=4" -> "b", not "4");
      * otherwise fall back to the last numeric token, then the last word.

    It does NOT score long free-form or symbolic answers -- that needs a
    semantic-equivalence judge (documented limitation).
    """
    if text is None:
        return ""
    m = re.search(r"answer\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        first_line = m.group(1).strip().splitlines()[0] if m.group(1).strip() else ""
        first_token = first_line.split()[0] if first_line.split() else ""
        cleaned = re.sub(r"[^a-z0-9.\-]", "", first_token.lower())
        return cleaned
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    words = _WORD.findall(text.lower())
    return words[-1] if words else ""
