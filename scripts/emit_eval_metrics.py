#!/usr/bin/env python3
"""Emit evaluation metrics JSON from SetResults.csv for CI artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "SetResults.csv"
OUT = ROOT / "artifacts" / "eval_metrics.json"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}; run RavensProject.py then RavensGrader.grade()")

    rows = []
    with SRC.open(newline="") as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            correct = int(row["Correct"])
            incorrect = int(row["Incorrect"])
            skipped = int(row.get("Skipped", 0) or 0)
            total = correct + incorrect + skipped
            rows.append(
                {
                    "set": row["Set"],
                    "correct": correct,
                    "incorrect": incorrect,
                    "skipped": skipped,
                    "accuracy": (correct / total) if total else None,
                }
            )

    basic = [r for r in rows if r["set"].startswith("Basic")]
    challenge = [r for r in rows if r["set"].startswith("Challenge")]

    def agg(group: list[dict]) -> dict:
        c = sum(r["correct"] for r in group)
        i = sum(r["incorrect"] for r in group)
        s = sum(r["skipped"] for r in group)
        t = c + i + s
        return {
            "correct": c,
            "incorrect": i,
            "skipped": s,
            "accuracy": (c / t) if t else None,
        }

    payload = {
        "schema_version": 1,
        "source": "SetResults.csv",
        "sets": rows,
        "basic": agg(basic),
        "challenge": agg(challenge),
        "overall": agg(rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
