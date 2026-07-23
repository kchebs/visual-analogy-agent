"""Load problems via ProblemSet, run solvers, grade vs ProblemAnswer.txt."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]

SolverFn = Callable[[Any], int | None]


def load_problem_set(set_name: str, *, root: Path | None = None) -> list[Any]:
    """Load problems for a named set (cwd / root must see Problems/)."""
    prev = os.getcwd()
    base = root or ROOT
    try:
        os.chdir(base)
        from ProblemSet import ProblemSet

        return list(ProblemSet(set_name).problems)
    finally:
        os.chdir(prev)


def ground_truth(set_name: str, problem_name: str, *, root: Path | None = None) -> int:
    base = root or ROOT
    path = base / "Problems" / set_name / problem_name / "ProblemAnswer.txt"
    return int(path.read_text().strip())


def run_solver(
    solver: SolverFn,
    problems: Iterable[Any],
    *,
    set_name: str,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a solver on problems; grade each answer against ProblemAnswer.txt."""
    rows: list[dict[str, Any]] = []
    for problem in problems:
        truth = ground_truth(set_name, problem.name, root=root)
        try:
            pred = solver(problem)
        except Exception as exc:  # keep bake-off resilient
            rows.append(
                {
                    "problem": problem.name,
                    "prediction": None,
                    "truth": truth,
                    "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if pred is None:
            outcome = "skipped"
        elif int(pred) < 0:
            outcome = "skipped"
        elif int(pred) == truth:
            outcome = "correct"
        else:
            outcome = "incorrect"

        rows.append(
            {
                "problem": problem.name,
                "prediction": None if pred is None else int(pred),
                "truth": truth,
                "outcome": outcome,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for r in rows if r["outcome"] == "correct")
    incorrect = sum(1 for r in rows if r["outcome"] == "incorrect")
    skipped = sum(1 for r in rows if r["outcome"] == "skipped")
    errors = sum(1 for r in rows if r["outcome"] == "error")
    graded = correct + incorrect
    return {
        "n": len(rows),
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "errors": errors,
        "accuracy": (correct / graded) if graded else None,
    }
