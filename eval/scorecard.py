"""Build comparative bake-off scorecard JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "bakeoff_scorecard.json"


def build_scorecard(
    methods: dict[str, dict[str, Any]],
    *,
    subset: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Comparative dict for classical / torch / llm (and any extras).

    Each method value should include at least status + optional summary/rows.
    """
    return {
        "schema_version": 1,
        "subset": subset or {},
        "methods": methods,
        "notes": notes or [],
    }


def write_scorecard(
    scorecard: dict[str, Any],
    path: Path | None = None,
) -> Path:
    out = path or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard, indent=2) + "\n")
    return out
