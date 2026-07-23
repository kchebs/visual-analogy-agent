"""Pixel-level image comparisons used by the visual analogy pipeline."""
from __future__ import annotations

from typing import Any

from rpm_agent import _impl


def rms(img1: Any, img2: Any) -> float:
    """Root-mean-square pixel difference between two images."""
    return float(_impl.rms(img1, img2))


def equal_imgs(image_one: Any, image_two: Any) -> bool:
    return bool(_impl.equal_imgs(image_one, image_two))


def dark_pixel_pct(image_one_arr: Any, image_two_arr: Any) -> float:
    return float(_impl.drk_pxl_perc(image_one_arr, image_two_arr))


def load_figure_array(problem: Any, name: str) -> Any:
    return _impl.get_imgs(problem, name)
