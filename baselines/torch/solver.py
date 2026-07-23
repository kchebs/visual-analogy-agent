"""Embedding analogy solver for 2×2 Raven problems (offline smoke weights)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_FIGURE_SIZE = 64
_WEIGHTS_PATH = Path(__file__).resolve().parent / "smoke_weights.pt"
_MODEL = None


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when torch absent
        raise ImportError(
            "baselines.torch requires the optional [torch] extra "
            "(pip install '.[torch]')"
        ) from exc
    return torch


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    torch = _require_torch()
    from baselines.torch.model import TinyCNNEncoder

    model = TinyCNNEncoder(embed_dim=64)
    if _WEIGHTS_PATH.is_file():
        state = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        # Deterministic random-init fallback (zero training); still usable offline.
        torch.manual_seed(0)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    model.eval()
    _MODEL = model
    return model


def figure_to_tensor(fig: Any):
    """Load a Raven figure PNG as a (1, 1, H, W) float tensor in [0, 1]."""
    torch = _require_torch()
    path = fig.visualFilename if hasattr(fig, "visualFilename") else fig["visualFilename"]
    img = Image.open(path).convert("L").resize((_FIGURE_SIZE, _FIGURE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _embed(model, fig: Any):
    torch = _require_torch()
    with torch.no_grad():
        return model(figure_to_tensor(fig)).squeeze(0)


def _answer_keys(problem: Any) -> list[str]:
    keys = []
    for i in range(1, 9):
        key = str(i)
        if key in problem.figures:
            keys.append(key)
    return keys


def _cosine(a, b) -> float:
    torch = _require_torch()
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float(torch.dot(a, b))


def solve(problem: Any) -> int:
    """Return best answer option index (1-based), matching Agent.Solve."""
    model = _load_model()
    answer_keys = _answer_keys(problem)
    if not answer_keys:
        return -1

    if problem.problemType == "2x2":
        emb_a = _embed(model, problem.figures["A"])
        emb_b = _embed(model, problem.figures["B"])
        emb_c = _embed(model, problem.figures["C"])
        # Classic analogy arithmetic: A:B :: C:?  →  C + (B - A)
        predicted = emb_c + (emb_b - emb_a)
        transform = emb_b - emb_a
    else:
        # Lightweight 3×3 fallback: treat G:H :: C:? style row completion.
        emb_g = _embed(model, problem.figures["G"])
        emb_h = _embed(model, problem.figures["H"])
        emb_c = _embed(model, problem.figures["C"])
        predicted = emb_c + (emb_h - emb_g)
        transform = emb_h - emb_g

    best_idx = 1
    best_score = float("-inf")
    for key in answer_keys:
        emb = _embed(model, problem.figures[key])
        # Blend arithmetic match with transform nearest-neighbor.
        arith = _cosine(predicted, emb)
        # Nearest neighbor of transform: compare (ans - C) to (B - A) for 2x2.
        if problem.problemType == "2x2":
            cand_t = emb - _embed(model, problem.figures["C"])
        else:
            cand_t = emb - _embed(model, problem.figures["C"])
        nn_score = _cosine(transform, cand_t)
        score = 0.5 * arith + 0.5 * nn_score
        idx = int(key)
        if score > best_score:
            best_score = score
            best_idx = idx
    return int(best_idx)
