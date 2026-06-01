from __future__ import annotations

import re
from typing import TypedDict


class Recommendation(TypedDict, total=False):
    title: str
    insight: str
    evidence: str
    action: str


_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "been",
        "being",
        "their",
        "from",
        "with",
        "that",
        "this",
        "than",
        "into",
        "such",
        "also",
        "may",
        "its",
        "more",
        "some",
        "what",
        "which",
        "while",
        "each",
        "other",
        "about",
        "there",
        "these",
        "they",
        "only",
        "same",
        "both",
        "very",
        "when",
    }
)


def _content_tokens(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]{3,}", text.lower())
        if w not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _entity_like_strings(text: str) -> set[str]:
    """Proper nouns and acronyms from original casing (report-agnostic)."""
    out: set[str] = set()
    for m in re.finditer(r"\b[A-Z]{2,}\b", text):
        out.add(m.group(0).lower())
    for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        out.add(m.group(0).lower())
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        w = m.group(0).lower()
        if w not in _STOPWORDS:
            out.add(w)
    return out


def _recommendation_text(rec: Recommendation) -> str:
    parts = [
        (rec.get("title") or "").strip(),
        (rec.get("insight") or "").strip(),
        (rec.get("evidence") or "").strip(),
        (rec.get("action") or "").strip(),
    ]
    return " ".join(p for p in parts if p)


def dedupe_recommendations(
    recs: list[Recommendation],
    *,
    high_jaccard: float = 0.42,
    entity_jaccard: float = 0.22,
) -> list[Recommendation]:
    if len(recs) <= 1:
        return recs
    kept: list[Recommendation] = []
    for rec in recs:
        combined = _recommendation_text(rec)
        tokens = _content_tokens(combined)
        entities = _entity_like_strings(combined)
        duplicate = False
        for prev in kept:
            prev_combined = _recommendation_text(prev)
            prev_tok = _content_tokens(prev_combined)
            prev_ent = _entity_like_strings(prev_combined)
            jac = _jaccard(tokens, prev_tok)
            if jac >= high_jaccard:
                duplicate = True
                break
            if entities & prev_ent and jac >= entity_jaccard:
                duplicate = True
                break
        if not duplicate:
            kept.append(rec)
    return kept


_FIELD_PATTERNS = {
    "title": re.compile(
        r"[\*\s]*Title[\*\s]*:?\s*(.+?)(?=\n[\*\s]*(?:Insight|Evidence|Action)[\*\s]*:?|\Z)",
        re.I | re.DOTALL,
    ),
    "insight": re.compile(
        r"[\*\s]*Insight[\*\s]*:?\s*(.+?)(?=\n[\*\s]*(?:Evidence|Action)[\*\s]*:?|\Z)",
        re.I | re.DOTALL,
    ),
    "evidence": re.compile(
        r"[\*\s]*Evidence[\*\s]*:?\s*(.+?)(?=\n[\*\s]*Action[\*\s]*:?|\Z)",
        re.I | re.DOTALL,
    ),
    "action": re.compile(r"[\*\s]*Action[\*\s]*:?\s*(.+)\Z", re.I | re.DOTALL),
}


def _clean_field_value(value: str) -> str:
    cleaned = value.strip()
    # Gemini sometimes leaves markdown marker residue like "** " after labels.
    cleaned = re.sub(r"^[\*\s:-]+", "", cleaned)
    return cleaned.strip()


def parse_strategy_markdown(raw: str) -> list[Recommendation]:
    text = (raw or "").strip()
    if not text:
        return []

    blocks = re.split(r"(?im)^#+\s*Recommendation.*$", text)
    if len(blocks) <= 1:
        # Fallback: if Gemini skips header formatting, split right before 'Title:'
        blocks = re.split(r"(?im)(?=^[\*\s]*Title[\*\s]*:)", text)

    recs: list[Recommendation] = []
    for block in blocks:
        chunk = block.strip()
        if not chunk:
            continue
        entry: Recommendation = {}
        for key, pat in _FIELD_PATTERNS.items():
            m = pat.search(chunk)
            if m:
                entry[key] = _clean_field_value(m.group(1))  # type: ignore[literal-required]
        if any(entry.get(k) for k in ("title", "insight", "evidence", "action")):
            recs.append(entry)
    return recs


def filter_grounded_recommendations(
    recs: list[Recommendation],
) -> list[Recommendation]:
    return [
        r
        for r in recs
        if (r.get("title") or "").strip()
        and (r.get("insight") or "").strip()
        and (r.get("evidence") or "").strip()
    ]
