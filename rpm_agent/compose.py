"""3x3 composition solvers (diagonal, row/column, generate-and-test)."""
from __future__ import annotations

from typing import Any

from rpm_agent import _impl


def unpack_figures(problem: Any) -> tuple[list[Any], list[Any]]:
    return _impl.begin(problem)


def score_diagonal(scores: list[Any], figures: list[Any], solutions: list[Any], problem: Any) -> list[Any]:
    return list(_impl.compare_diag(scores, figures, solutions, problem))


def score_top_bottom(scores: list[Any], figures: list[Any], solutions: list[Any]) -> list[Any]:
    return list(_impl.top_comp_bttm(scores, figures, solutions))


def score_image_ops(figures: list[Any], solutions: list[Any]) -> list[Any]:
    return list(_impl.img_op_solver(figures, solutions))


def score_generate_and_test(
    init_net: Any, scores: list[Any], figures: list[Any], solutions: list[Any], problem: Any
) -> list[Any]:
    return list(_impl.gn_tst(init_net, scores, figures, solutions, problem))
