"""Typed facade over the visual analogy solver."""
from __future__ import annotations

from typing import Any

from rpm_agent import _impl


class Agent:
    """Solves 2x2 and 3x3 Raven-style visual analogy problems."""

    def __init__(self) -> None:
        self._impl = _impl.Agent()

    def Solve(self, problem: Any) -> int:
        """Return the best answer option index (1-based)."""
        answer = self._impl.Solve(problem)
        return int(answer)
