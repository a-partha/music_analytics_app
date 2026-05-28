from __future__ import annotations

from enum import Enum


class AnalysisMode(str, Enum):
    DTC_ONLY = "dtc_only"
    IP_ONLY = "ip_only"
    BOTH = "both"


def resolve_analysis_mode(
    value: AnalysisMode | str | None,
) -> AnalysisMode | None:
    if value is None:
        return None
    if isinstance(value, AnalysisMode):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in ("dtc", "dtc_only", "dtc only"):
        return AnalysisMode.DTC_ONLY
    if raw in ("ip", "ip_only", "ip only"):
        return AnalysisMode.IP_ONLY
    if raw in ("both", "dtc_ip", "dtc+ip"):
        return AnalysisMode.BOTH
    return None

