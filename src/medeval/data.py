"""Rebuild the exact split used in training and pick a held-out test set.

The training notebook ran `train_test_split(test_size=0.1, seed=42)` on the
Medical Meadow flashcards, trained on the first 5,000 rows of the train side
and chose its checkpoint on the first 500 rows of the test side. Every other
row of the test side was never seen during training or model selection.
"""
from __future__ import annotations

import re

from . import DATASET, SPLIT_SEED

MMLU_MEDICAL = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def heldout_flashcards(n: int, sample_seed: int = 0):
    """Return (heldout sample, training questions) as lists of dicts / a set."""
    from datasets import load_dataset

    ds = load_dataset(DATASET)["train"]
    split = ds.train_test_split(test_size=0.1, seed=SPLIT_SEED)
    trained = split["train"].select(range(5000))
    held = split["test"].select(range(500, len(split["test"])))
    seen = {normalise(q) for q in trained["input"]}
    rows = [
        {"id": f"mm-{i}", "question": r["input"].strip(), "reference": r["output"].strip()}
        for i, r in enumerate(held)
        if r["input"].strip() and r["output"].strip() and normalise(r["input"]) not in seen
    ]
    return sample(rows, n, sample_seed), seen


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    import random

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), min(n, len(rows))))
    return [rows[i] for i in idx]


def mmlu_medical() -> list[dict]:
    from datasets import load_dataset

    out = []
    for sub in MMLU_MEDICAL:
        for i, r in enumerate(load_dataset("cais/mmlu", sub, split="test")):
            out.append({"id": f"{sub}-{i}", "subject": sub, "question": r["question"],
                        "choices": list(r["choices"]), "answer": int(r["answer"])})
    return out
