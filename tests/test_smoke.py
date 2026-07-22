"""Smoke tests: agent solves a few Basic problems without crashing."""

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


@pytest.mark.parametrize(
    "set_name,problem_name,expected",
    [
        ("Basic Problems B", "Basic Problem B-01", 2),
        ("Basic Problems B", "Basic Problem B-02", 5),
        ("Basic Problems D", "Basic Problem D-01", 3),
        ("Basic Problems D", "Basic Problem D-02", 1),
    ],
)
def test_basic_problem_answer(set_name, problem_name, expected):
    from Agent import Agent

    agent = Agent()
    problem = _load_problem(set_name, problem_name)
    answer = agent.Solve(problem)
    assert isinstance(answer, int)
    assert 1 <= answer <= 8
    assert answer == expected


def test_agent_constructs():
    from Agent import Agent

    assert Agent() is not None
