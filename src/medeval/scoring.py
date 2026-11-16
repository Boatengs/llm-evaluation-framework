"""Answer scoring against a reference.

ROUGE F1 rewards answers whose length matches the reference, and flashcard
references are short, so precision and recall are kept separate and answer
length is recorded. A sentence-embedding similarity is added as a check that
is less tied to exact wording.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _rouge():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(prediction: str, reference: str) -> dict[str, float]:
    s = _rouge().score(reference, prediction)
    out = {}
    for k in ("rouge1", "rouge2", "rougeL"):
        out[f"{k}_f"] = s[k].fmeasure
        out[f"{k}_p"] = s[k].precision
        out[f"{k}_r"] = s[k].recall
    out["avg_rouge_f"] = (out["rouge1_f"] + out["rouge2_f"] + out["rougeL_f"]) / 3
    out["words"] = len(prediction.split())
    return out


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def embedding_similarity(predictions: list[str], references: list[str]) -> list[float]:
    model = _embedder()
    a = model.encode(predictions, normalize_embeddings=True, batch_size=64)
    b = model.encode(references, normalize_embeddings=True, batch_size=64)
    return [float((x * y).sum()) for x, y in zip(a, b, strict=True)]
