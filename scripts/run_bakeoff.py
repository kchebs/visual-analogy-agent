#!/usr/bin/env python3
"""Run classical + torch (+ optional LLM) on a small smoke subset."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from eval.harness import load_problem_set, run_solver, summarize
from eval.scorecard import build_scorecard, write_scorecard

SET_NAME = "Basic Problems B"
SUBSET_N = 3


def _classical_solver(problem):
    from Agent import Agent

    return Agent().Solve(problem)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> None:
    problems = load_problem_set(SET_NAME, root=ROOT)[:SUBSET_N]
    subset_meta = {
        "set": SET_NAME,
        "n": len(problems),
        "problems": [p.name for p in problems],
        "caveat": "smoke subset — not full eval",
    }

    methods: dict = {}

    # Classical (always)
    classical_rows = run_solver(_classical_solver, problems, set_name=SET_NAME, root=ROOT)
    methods["classical"] = {
        "status": "ok",
        "summary": summarize(classical_rows),
        "rows": classical_rows,
    }

    # Torch (optional dependency)
    if not _torch_available():
        methods["torch"] = {
            "status": "skipped_no_torch",
            "summary": None,
            "rows": [],
        }
    else:
        from baselines.torch.solver import solve as torch_solve

        torch_rows = run_solver(torch_solve, problems, set_name=SET_NAME, root=ROOT)
        methods["torch"] = {
            "status": "ok",
            "summary": summarize(torch_rows),
            "rows": torch_rows,
        }

    # LLM only if env provider set (and key for non-skipped mode)
    from baselines.llm.solver import mode_status, solve as llm_solve

    llm_mode = mode_status()
    if llm_mode == "skipped":
        methods["llm"] = {
            "status": "skipped",
            "mode": llm_mode,
            "summary": None,
            "rows": [],
            "reason": "VISUAL_ANALOGY_LLM_PROVIDER unset or no API key",
        }
    else:
        llm_rows = run_solver(llm_solve, problems, set_name=SET_NAME, root=ROOT)
        methods["llm"] = {
            "status": "ok",
            "mode": llm_mode,
            "summary": summarize(llm_rows),
            "rows": llm_rows,
        }

    scorecard = build_scorecard(
        methods,
        subset=subset_meta,
        notes=[
            "Smoke subset only (Basic Problems B, first 3).",
            "Torch uses committed smoke_weights.pt (zero training).",
            "LLM is env-gated and does not call the network without a key.",
        ],
    )
    out = write_scorecard(scorecard)
    print(f"wrote {out}")
    for name, payload in methods.items():
        summary = payload.get("summary")
        status = payload.get("status")
        if summary and summary.get("accuracy") is not None:
            print(
                f"  {name}: status={status} "
                f"correct={summary['correct']}/{summary['correct'] + summary['incorrect']} "
                f"acc={summary['accuracy']:.0%}"
            )
        else:
            print(f"  {name}: status={status}")


if __name__ == "__main__":
    main()
