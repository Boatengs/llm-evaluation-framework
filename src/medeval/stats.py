"""Paired statistics for comparing two systems on the same questions."""
from __future__ import annotations

import math

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean of b minus a with a percentile interval from resampling questions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci95": [float(lo), float(hi)], "n": int(len(d))}


def sign_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided sign test on b versus a; ties are dropped."""
    d = np.asarray(b, float) - np.asarray(a, float)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"b_wins": wins, "a_wins": losses, "ties": int(len(d) - n), "p_value": p}


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test for paired accuracy."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_b, only_a = int((b & ~a).sum()), int((a & ~b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_b_correct": only_b, "only_a_correct": only_a, "p_value": p}
