"""One-off probe: data access, held-out split, duplicate check, server speed."""
import json, os, re, subprocess, sys, time, urllib.request

from datasets import load_dataset

ds = load_dataset("medalpaca/medical_meadow_medical_flashcards")["train"]
print("rows", len(ds), "cols", ds.column_names)
split = ds.train_test_split(test_size=0.1, seed=42)
train5k = split["train"].select(range(5000))
val500 = split["test"].select(range(500))
held = split["test"].select(range(500, len(split["test"])))
print("train5k", len(train5k), "val500", len(val500), "heldout", len(held))
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
tq = {norm(x) for x in train5k["input"]}
dup = sum(norm(x) in tq for x in held["input"])
print("heldout questions identical to a training question:", dup)
print("instructions", {k: v for k, v in __import__("collections").Counter(held["instruction"]).most_common(3)})
lens = sorted(len(x.split()) for x in held["output"])
print("answer words median", lens[len(lens)//2], "p90", lens[int(len(lens)*.9)])
for sub in ["anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine", "college_biology"]:
    print("mmlu", sub, len(load_dataset("cais/mmlu", sub, split="test")))

def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:8080" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))

print("lora list", json.load(urllib.request.urlopen("http://127.0.0.1:8080/lora-adapters")))
prompt = "### Instruction:\nAnswer this question truthfully\n\n### Input:\nWhat is the mechanism of action of beta blockers?\n\n### Response:\n"
for scale in [0.0, 1.0]:
    t = time.time()
    out = post("/completion", {"prompt": prompt, "n_predict": 120, "temperature": 0, "lora": [{"id": 0, "scale": scale}], "cache_prompt": False})
    print(f"scale {scale}: {out['timings']['predicted_per_second']:.1f} tok/s gen, {out['timings']['prompt_per_second']:.1f} tok/s prompt, {time.time()-t:.1f}s")
    print("  ", out["content"][:300].replace("\n", " | "))
mc = "Question: Which vitamin deficiency causes scurvy?\nA. Vitamin A\nB. Vitamin B12\nC. Vitamin C\nD. Vitamin D\nAnswer:"
out = post("/completion", {"prompt": mc, "n_predict": 1, "temperature": 0, "n_probs": 10, "lora": [{"id": 0, "scale": 0.0}]})
print("mc probs", json.dumps(out.get("completion_probabilities", [])[:1])[:600])
