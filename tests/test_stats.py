import numpy as np
import pytest

from medeval.stats import mcnemar, paired_bootstrap, sign_test


def test_sign_test_known_value():
    # 13 wins, 7 losses: two-sided exact p = 0.2632
    a = np.zeros(20)
    b = np.array([1] * 13 + [-1] * 7, float)
    r = sign_test(a, b)
    assert (r["b_wins"], r["a_wins"]) == (13, 7)
    assert r["p_value"] == pytest.approx(0.2632, abs=1e-3)


def test_sign_test_drops_ties():
    r = sign_test(np.zeros(5), np.array([0, 0, 1, 1, 1.0]))
    assert r["ties"] == 2 and r["p_value"] == pytest.approx(0.25)


def test_paired_bootstrap_covers_true_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0.2, 0.1, 400)
    b = a + 0.05 + rng.normal(0, 0.05, 400)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["ci95"][0] < 0.05 < r["ci95"][1]
    assert r["diff"] == pytest.approx(0.05, abs=0.01)


def test_mcnemar_symmetric_is_not_significant():
    a = np.array([1, 0] * 50, bool)
    b = np.array([0, 1] * 50, bool)
    r = mcnemar(a, b)
    assert r["only_a_correct"] == r["only_b_correct"] == 50
    assert r["p_value"] == pytest.approx(1.0)
