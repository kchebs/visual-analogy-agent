"""Unit tests for rpm_agent facades and torch baseline smoke."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _chdir_repo_root():
    prev = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(prev)


def _load_problem(set_name: str, problem_name: str):
    from ProblemSet import ProblemSet

    ps = ProblemSet(set_name)
    for problem in ps.problems:
        if problem.name == problem_name:
            return problem
    raise AssertionError(f"{problem_name} not found in {set_name}")


def test_detect_reflect_lr_returns_float():
    from PIL import Image

    from rpm_agent import detect

    problem = _load_problem("Basic Problems B", "Basic Problem B-01")
    a = Image.open(problem.figures["A"].visualFilename).convert("L")
    b = Image.open(problem.figures["B"].visualFilename).convert("L")
    value = detect.reflect_lr(a, b)
    assert isinstance(value, float)
    assert value >= 0.0


def test_score_best_as_one_and_normalize():
    from rpm_agent import score

    # scr_arr expects ranked indices; marks best (first) as one-hot over 8 slots
    ranked = score.best_as_one([2, 0, 1])
    assert isinstance(ranked, list)
    assert len(ranked) == 8
    assert ranked[2] == 1
    assert sum(ranked) == 1

    normalized = score.normalize_scores([0, 0, 5, 0, 0, 0, 0, 0])
    assert isinstance(normalized, list)
    assert len(normalized) == 8
    assert normalized[2] == pytest.approx(1.0)


def test_torch_solver_constructs_and_returns_int():
    torch = pytest.importorskip("torch")
    from baselines.torch.model import TinyCNNEncoder
    from baselines.torch.solver import solve

    model = TinyCNNEncoder()
    assert model.embed_dim == 64

    problem = _load_problem("Basic Problems B", "Basic Problem B-01")
    answer = solve(problem)
    assert isinstance(answer, int)
    assert 1 <= answer <= 6


def test_llm_solver_skipped_without_env(monkeypatch):
    monkeypatch.delenv("VISUAL_ANALOGY_LLM_PROVIDER", raising=False)
    from baselines.llm.solver import mode_status, solve

    assert mode_status() == "skipped"
    problem = _load_problem("Basic Problems B", "Basic Problem B-01")
    assert solve(problem) is None
