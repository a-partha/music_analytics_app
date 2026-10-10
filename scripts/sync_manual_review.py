"""One-time fill of manual_review.json from testing_report.json answers.

Hashes are computed from the saved answer strings. The correct flag and note
are the manual check of each answer against that item's ground truth.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "src" / "validation" / "evals" / "testing_report.json"
REVIEW_PATH = ROOT / "src" / "validation" / "evals" / "manual_review.json"

# id -> (correct, note). False means the answer refused for lack of context
# or stated a figure that contradicts the ground truth.
JUDGMENTS = {
    "consumption_metrics_01": (
        True,
        "Matches ground truth: global On-Demand Audio change was +9.6%.",
    ),
    "consumption_metrics_02": (
        False,
        "Retrieval failed: answer says the context lacks the ex-U.S. +11.6% figure.",
    ),
    "consumption_metrics_03": (
        True,
        "Matches ground truth: U.S. total album consumption was 1.138 billion.",
    ),
    "consumption_metrics_04": (
        False,
        "Retrieval failed: answer says the context lacks Canada's 3.8 million physical sales.",
    ),
    "vital_stats_01": (
        True,
        "Matches ground truth: 106,000 ISRCs per day is the same figure as 106K.",
    ),
    "vital_stats_02": (
        True,
        "Matches ground truth: 88% of tracks had 1K streams or less.",
    ),
    "vital_stats_03": (
        False,
        "Retrieval failed: answer says the context lacks the 47.9% figure for 2020s releases.",
    ),
    "vital_stats_04": (
        True,
        "Matches ground truth: U.S. vinyl sales rose for the 19th straight year.",
    ),
    "premium_pricing_01": (
        True,
        "Matches ground truth: the four territories are the U.S., Mexico, Brazil, and Germany.",
    ),
    "premium_pricing_02": (
        False,
        "Retrieval failed: answer says the context lacks the LATAM conversion lead.",
    ),
    "premium_pricing_03": (
        False,
        "Retrieval failed: answer says the context lacks the 42% Japanese Gen Z figure.",
    ),
    "premium_pricing_04": (
        False,
        "Retrieval failed: answer says the context lacks the Apple Music over-index.",
    ),
    "welcome_to_transmedia_01": (
        True,
        "Matches ground truth: Becoming Led Zeppelin ranked first in minutes watched.",
    ),
    "welcome_to_transmedia_02": (
        True,
        "Matches ground truth: the Taylor Swift movie grossed $34M at the U.S. box office.",
    ),
    "welcome_to_transmedia_03": (
        False,
        "Retrieval failed: answer says the context lacks the KPop Demon Hunters soundtrack.",
    ),
    "welcome_to_transmedia_04": (
        False,
        "Retrieval failed: answer says the context lacks the 1.1 million Fortnite streams.",
    ),
    "evolving_fandom_01": (
        True,
        "Matches ground truth: Brazil moved up on Latin music exports.",
    ),
    "evolving_fandom_02": (
        False,
        "Retrieval failed: answer says the context lacks Brazil's 75.2% local-artist share.",
    ),
    "evolving_fandom_03": (
        True,
        "Matches ground truth: 20% of U.S. music listeners are superfans.",
    ),
    "ai_artists_01": (
        True,
        "Matches ground truth: the Xania Monet creator received a $3 million advance.",
    ),
    "ai_artists_02": (
        True,
        "Matches ground truth: Xania Monet was the only AI artist in the top 97th percentile.",
    ),
    "ai_artists_03": (
        False,
        "Contradicts ground truth: answer says 29%, ground truth says 44%.",
    ),
    "year_end_charts_01": (
        True,
        "Matches ground truth: Die With A Smile by Lady Gaga and Bruno Mars ranked first.",
    ),
    "year_end_charts_02": (
        True,
        "Matches ground truth: The Life of a Showgirl by Taylor Swift had the highest U.S. sales.",
    ),
    "year_end_charts_03": (
        False,
        "Retrieval failed: answer says the context lacks the top U.S. radio song.",
    ),
}


def main():
    with open(REPORT_PATH, encoding="utf-8") as report_file:
        report = json.load(report_file)
    results = report["results"]
    ids = [item["id"] for item in results]
    missing = [item_id for item_id in ids if item_id not in JUDGMENTS]
    extra = [item_id for item_id in JUDGMENTS if item_id not in ids]
    if missing or extra:
        raise SystemExit(f"Judgment keys do not match results. missing={missing} extra={extra}")

    review = {}
    for item in results:
        answer = item["answer"]
        correct, note = JUDGMENTS[item["id"]]
        review[item["id"]] = {
            "correct": correct,
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "note": note,
        }

    with open(REVIEW_PATH, "w", encoding="utf-8") as review_file:
        json.dump(review, review_file, indent=2)
        review_file.write("\n")
    print(f"Wrote {len(review)} reviews to {REVIEW_PATH.name}.")


if __name__ == "__main__":
    main()
