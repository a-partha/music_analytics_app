"""Unit tests for strategy markdown parsing (four fields)."""

from __future__ import annotations

from src.agents.strategy.postprocess import parse_strategy_markdown


def test_parse_four_fields() -> None:
    raw = """
## Recommendation 1
**Title:** Superfan surge
**Insight:** Gaming platforms over-index for genre X.
**Evidence:** Engagement Horizon notes superfans in gaming.
**Action:** Prioritize Discord activations in markets A and B.

## Recommendation 2
**Title:** Export mix shift
**Insight:** Local share rose in key territories.
**Evidence:** Import / Export describes cross-border flows.
**Action:** Rebalance catalog marketing toward export-first titles.
"""
    recs = parse_strategy_markdown(raw)
    assert len(recs) == 2
    assert recs[0]["title"] == "Superfan surge"
    assert "Gaming" in recs[0]["insight"]
    assert "Engagement Horizon" in recs[0]["evidence"]
    assert "Discord" in recs[0]["action"]
