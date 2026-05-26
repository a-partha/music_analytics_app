from __future__ import annotations

from rapidfuzz import fuzz

CATEGORY_DTC = "DTC"
CATEGORY_IP = "IP"
CATEGORY_OTHER = "OTHER"

# Mirrors src.pipelines.analysis_pipeline tuples (duplicated to avoid import cycles).
DTC_SECTION_REFERENCES = (
    "Engagement Horizon",
)
IP_SECTION_REFERENCES = (
    "Import / Export",
    "Streaming Atlas",
)
OTHER_SECTION_REFERENCES = (
    "Midyear Metrics",
    "Artist Spectrum",
    "Future in Focus",
    "Midyear Charts",
)

DTC_ALIASES = (
    "engagement horizon",
    "fan engagement",
    "superfan",
    "superfans",
    "direct to consumer",
    "dtc",
    "short-form video",
    "tiktok",
    "reels",
    "shorts",
    "discord",
    "roblox",
    "fortnite",
    "gaming music",
    "podcasts",
    "community platforms",
    "fan tiers",
)

IP_ALIASES = (
    "import export",
    "import / export",
    "streaming atlas",
    "atlas",
    "catalog",
    "deep catalog",
    "export power",
    "cross-border",
    "local vs foreign",
    "territory",
    "chart performance",
    "country export",
)

OTHER_ALIASES = (
    "midyear metrics",
    "artist spectrum",
    "future in focus",
    "midyear charts",
    "charts",
    "metrics",
    "outlook",
)


def _pool_score(display_title: str, references: tuple[str, ...]) -> float:
    title = display_title.strip()
    if not title:
        return 0.0
    best = 0.0
    for ref in references:
        best = max(best, float(fuzz.WRatio(title, ref)))
    return best


# Minimum best-pool score to assign a non-OTHER category; below → OTHER.
DEFAULT_CATEGORY_MIN_SCORE = 55.0


def assign_category(
    display_title: str,
    *,
    min_score: float = DEFAULT_CATEGORY_MIN_SCORE,
) -> str:
    """
    Map a document section title to DTC, IP, or OTHER using fuzzy match
    against known section names and short aliases.
    """
    title = display_title.strip()
    if not title:
        return CATEGORY_OTHER

    pools: tuple[tuple[str, tuple[str, ...]], ...] = (
        (CATEGORY_DTC, DTC_SECTION_REFERENCES + DTC_ALIASES),
        (CATEGORY_IP, IP_SECTION_REFERENCES + IP_ALIASES),
        (CATEGORY_OTHER, OTHER_SECTION_REFERENCES + OTHER_ALIASES),
    )
    scored = [(cat, _pool_score(title, refs)) for cat, refs in pools]
    best_cat, best_score = max(scored, key=lambda x: x[1])
    if best_score < min_score:
        return CATEGORY_OTHER
    return best_cat
