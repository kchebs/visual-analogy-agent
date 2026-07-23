"""Score answer options and normalize rankings."""
from __future__ import annotations

from typing import Any

from rpm_agent import _impl


def best_as_one(scores: list[float]) -> list[float]:
    return list(_impl.scr_arr(scores))


def normalize_scores(scores: list[Any]) -> list[Any]:
    return list(_impl.norm_scrs(scores))
