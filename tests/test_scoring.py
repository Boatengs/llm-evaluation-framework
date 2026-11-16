import pytest

from medeval.scoring import rouge


def test_rouge_identical_is_one():
    r = rouge("statins inhibit HMG-CoA reductase", "statins inhibit HMG-CoA reductase")
    assert r["rougeL_f"] == pytest.approx(1.0) and r["words"] == 4


def test_long_answer_keeps_recall_but_loses_precision():
    ref = "aspirin inhibits cox enzymes"
    short = rouge("aspirin inhibits cox enzymes", ref)
    long = rouge("aspirin inhibits cox enzymes " + "and many other words about platelets " * 5, ref)
    assert long["rouge1_r"] == pytest.approx(short["rouge1_r"])
    assert long["rouge1_p"] < short["rouge1_p"]
    assert long["rouge1_f"] < short["rouge1_f"]
