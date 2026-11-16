"""Query a running llama-server for one shard of the work.

    python -m medeval.run generate --shard 3 --shards 12 --out out/gen-3.jsonl
    python -m medeval.run mmlu --shard 0 --shards 8 --out out/mmlu-0.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.request
from pathlib import Path

from .data import heldout_flashcards, mmlu_medical
from .prompts import CONDITIONS, LETTERS, STOP, mmlu_prompt

SERVER = "http://127.0.0.1:8080"
N_FLASHCARDS = 300
MAX_TOKENS = 200


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def generation_items() -> list[dict]:
    items, _ = heldout_flashcards(N_FLASHCARDS)
    for r in items:
        r["set"] = "heldout"
    original = Path(__file__).resolve().parents[2] / "eval_results.csv"
    with open(original, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            items.append({"id": f"orig-{i}", "set": "original20", "category": row["category"],
                          "question": row["question"], "reference": row["reference"]})
    return items


def generate(shard: int, shards: int, out: Path) -> None:
    items = [r for i, r in enumerate(generation_items()) if i % shards == shard]
    with open(out, "w") as f:
        for n, item in enumerate(items):
            for name, template, scale in CONDITIONS:
                res = post("/completion", {
                    "prompt": template.format(question=item["question"]),
                    "n_predict": MAX_TOKENS, "temperature": 0, "top_k": 1, "seed": 0,
                    "stop": STOP, "lora": [{"id": 0, "scale": scale}], "cache_prompt": False,
                })
                f.write(json.dumps({**item, "condition": name, "answer": res["content"].strip(),
                                    "tokens": res["tokens_predicted"],
                                    "truncated": res.get("stop_type") == "limit" or res.get("truncated", False)}) + "\n")
            f.flush()
            print(f"gen {n + 1}/{len(items)}", flush=True)


def letter_logprobs(res: dict) -> list[float]:
    probs = res["completion_probabilities"][0]["top_logprobs"]
    best = {L: -math.inf for L in LETTERS}
    for p in probs:
        tok = p["token"].strip()
        if tok in best:
            best[tok] = max(best[tok], p["logprob"])
    return [best[L] for L in LETTERS]


def mmlu(shard: int, shards: int, out: Path) -> None:
    items = [r for i, r in enumerate(mmlu_medical()) if i % shards == shard]
    with open(out, "w") as f:
        for n, item in enumerate(items):
            for name, scale in [("base", 0.0), ("finetuned", 1.0)]:
                res = post("/completion", {"prompt": mmlu_prompt(item), "n_predict": 1, "temperature": 0,
                                           "n_probs": 20, "lora": [{"id": 0, "scale": scale}], "cache_prompt": False})
                lp = letter_logprobs(res)
                pred = max(range(4), key=lambda k: lp[k])
                f.write(json.dumps({"id": item["id"], "subject": item["subject"], "condition": name,
                                    "answer": item["answer"], "pred": pred, "logprobs": lp,
                                    "letters_found": sum(math.isfinite(x) for x in lp)}) + "\n")
            f.flush()
            print(f"mmlu {n + 1}/{len(items)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["generate", "mmlu"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    (generate if a.task == "generate" else mmlu)(a.shard, a.shards, a.out)


if __name__ == "__main__":
    main()
