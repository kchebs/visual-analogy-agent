"""Optional VLM/LLM solver stub — never fails CI when unset."""
from __future__ import annotations

import os
from typing import Any, Literal

Mode = Literal["skipped", "stub_ready", "live"]

_PROVIDER_ENV = "VISUAL_ANALOGY_LLM_PROVIDER"
_KEY_ENVS = (
    "VISUAL_ANALOGY_LLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


def mode_status() -> Mode:
    """Return how the LLM baseline should run given current env."""
    provider = (os.environ.get(_PROVIDER_ENV) or "").strip()
    if not provider:
        return "skipped"
    if not any(os.environ.get(k) for k in _KEY_ENVS):
        return "skipped"
    # Keys present: still a stub unless a real provider client is wired later.
    return "stub_ready"


def _fill_ratio(fig: Any) -> float:
    """Cheap visual 'feature': fraction of dark pixels (no network)."""
    from PIL import Image
    import numpy as np

    path = fig.visualFilename if hasattr(fig, "visualFilename") else fig["visualFilename"]
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    # Dark ink on light background (RPM PNGs).
    return float((arr < 128).mean())


def describe_problem_features(problem: Any) -> dict[str, Any]:
    """Structured fill-ratio features for a future VLM/LLM prompt."""
    cells = {}
    for name, fig in problem.figures.items():
        try:
            cells[name] = {"fill_ratio": _fill_ratio(fig)}
        except OSError:
            cells[name] = {"fill_ratio": None}
    return {
        "name": getattr(problem, "name", None),
        "problem_type": getattr(problem, "problemType", None),
        "cells": cells,
        "provider": (os.environ.get(_PROVIDER_ENV) or "").strip() or None,
    }


def _stub_pick_from_features(features: dict[str, Any]) -> int:
    """
    Placeholder policy (no network): pick answer whose fill_ratio is closest
    to C + (B - A) fill arithmetic. Documents the interface only.
    """
    cells = features.get("cells") or {}
    try:
        a = cells["A"]["fill_ratio"]
        b = cells["B"]["fill_ratio"]
        c = cells["C"]["fill_ratio"]
    except (KeyError, TypeError):
        return 1
    if None in (a, b, c):
        return 1
    target = c + (b - a)
    best_idx, best_dist = 1, float("inf")
    for i in range(1, 9):
        key = str(i)
        if key not in cells or cells[key].get("fill_ratio") is None:
            continue
        dist = abs(cells[key]["fill_ratio"] - target)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return int(best_idx)


def solve(problem: Any) -> int | None:
    """
    Match Agent.Solve when active; return None when mode is skipped.

    Never calls an external network unless a provider *and* API key are set.
    Even then, this repo ships a local fill-ratio stub only — wire a real
    client behind mode ``live`` later without changing the harness contract.
    """
    mode = mode_status()
    if mode == "skipped":
        return None

    features = describe_problem_features(problem)
    # Future: if mode == "live", call provider with features / images.
    # Guard remains: no network unless key present (already required above).
    return _stub_pick_from_features(features)
