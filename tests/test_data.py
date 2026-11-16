from medeval.data import normalise, sample


def test_normalise_ignores_case_and_punctuation():
    assert normalise("What is ACE?") == normalise("what is ace")


def test_sample_is_deterministic_and_ordered():
    rows = [{"id": i} for i in range(100)]
    a, b = sample(rows, 10, 3), sample(rows, 10, 3)
    assert a == b and [r["id"] for r in a] == sorted(r["id"] for r in a)
    assert len(sample(rows, 500, 0)) == 100
