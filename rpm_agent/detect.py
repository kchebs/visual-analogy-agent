"""Transform detection between matrix cells (reflection, rotation, fill)."""
from __future__ import annotations

from typing import Any

from rpm_agent import _impl


def reflect_lr(image_one: Any, image_two: Any) -> float:
    return float(_impl.refLR(image_one, image_two))


def reflect_tb(image_one: Any, image_two: Any) -> float:
    return float(_impl.refTB(image_one, image_two))


def rotate_90(image_one: Any, image_two: Any) -> float:
    return float(_impl.rot90(image_one, image_two))


def rotate_270(image_one: Any, image_two: Any) -> float:
    return float(_impl.rot270(image_one, image_two))


def difference(image_one: Any, image_two: Any) -> float:
    return float(_impl.diff_finder(image_one, image_two))


def detect_pairwise_transform(fig1: Any, fig2: Any) -> Any:
    """Legacy semantic-net edge builder between two figures."""
    return _impl.trnsfrmtn(fig1, fig2)
