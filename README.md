# Medical QA LLM Evaluation: Mistral 7B vs its LoRA Fine Tune

An evaluation of [`samurvivor-07/medical-mistral-lora`](https://huggingface.co/samurvivor-07/medical-mistral-lora), a LoRA adapter for Mistral-7B-Instruct-v0.3 trained on 5,000 Medical Meadow flashcards ([training repo](https://github.com/Boatengs/medical-qa-lora)), against the base model it was trained from.

This is version 2. Version 1 compared the two models on 20 hand-written questions with ROUGE and reported a 17.8% gain and a 65% win rate. Version 2 tests on 300 held-out flashcards and 945 multiple choice questions, gives the base model a fair prompt, separates answer style from content, and reports intervals and significance tests.

## Results in brief

| Question | Answer |
|---|---|
| Does the fine tune answer held-out flashcards better? | Yes. Average ROUGE F1 rose from 0.273 to 0.421 (+0.148, 95% interval +0.118 to +0.178) with the same prompt, and embedding similarity rose from 0.802 to 0.848. |
| Is that gain mostly style? | Largely. ROUGE-L precision doubled (0.294 to 0.600) while recall did not move (0.458 to 0.457). The fine tune's median answer is 15.5 words, close to the flashcards' 22.5; the base's is 72.5. |
| Did fine tuning cost general medical knowledge? | Yes, measurably. On 945 MMLU medical questions accuracy fell from 65.2% to 61.2% (−4.0 points, 95% interval −6.6 to −1.6, McNemar p = 0.002). |
| Did the version 1 headline hold up? | Its direction held, but on weak evidence: 13 wins of 20 has a sign test p of 0.26. Rerun with greedy decoding the fine tune won 15 of 20 (p = 0.041). |

## What changed from version 1

| | Version 1 | Version 2 |
|---|---|---|
| Test questions | 20, written in the notebook | 300 held-out Medical Meadow flashcards, plus the original 20 |
| Leakage control | None | Rebuilt the training split (seed 42), excluded the 5,000 training rows and the 500 checkpoint selection rows, removed 36 held-out questions identical to a training question |
| Base model prompt | The fine tune's flashcard format | Both the flashcard format and Mistral's own `[INST]` chat format |
| Decoding | Sampling at temperature 0.1 | Greedy, so every answer can be reproduced |
| Metrics | Average ROUGE F1 | ROUGE F1, precision and recall, answer length, embedding similarity, and MMLU medical accuracy |
| Statistics | Means only | Paired bootstrap intervals, exact sign tests, McNemar |
| Code | One notebook on a Colab GPU | `medeval` package, tests, CI, and a 20 job GitHub Actions run on CPU with llama.cpp |

## Held-out flashcards (300 questions)

| Metric | Base, flashcard prompt | Base, chat prompt | Fine tuned |
|---|---|---|---|
| Average ROUGE F1 | 0.273 | 0.229 | 0.421 |
| ROUGE-L precision | 0.294 | 0.184 | 0.600 |
| ROUGE-L recall | 0.458 | 0.494 | 0.457 |
| Embedding similarity | 0.802 | 0.792 | 0.848 |
| Median answer length (words) | 72.5 | 109.5 | 15.5 |
| Answers cut off at 200 tokens | 17.7% | 56.7% | 1.3% |

Median reference answer: 22.5 words. Against the base in its own chat format the fine tune's gain is larger (average ROUGE +0.192, embedding similarity +0.055), but its ROUGE-L recall is 0.037 lower. The base answers that hit the 200 token limit lose some recall, so that recall gap is if anything understated.

## MMLU medical multiple choice (945 questions)

| Subject | Questions | Base | Fine tuned |
|---|---|---|---|
| Anatomy | 135 | 56.3% | 54.8% |
| Clinical knowledge | 265 | 70.2% | 64.5% |
| College medicine | 173 | 60.1% | 53.2% |
| Medical genetics | 100 | 71.0% | 70.0% |
| Professional medicine | 272 | 65.8% | 62.9% |
| All | 945 | 65.2% | 61.2% |

The fine tune was lower in every subject. It answered 55 questions the base got wrong and missed 93 the base got right.

Full tables, every interval and p value: [`results/RESULTS.md`](results/RESULTS.md). Every answer and score: [`results/raw/`](results/raw/).

## What this shows

The adapter did what its training data asks: short, flashcard style answers that match the reference closely. That is a real gain for that format. It is not evidence of more medical knowledge. Recall of reference content did not improve, and accuracy on standard medical multiple choice went down, which is the forgetting that version 1's write-up said had not happened.

Limits:
- The models ran as an 8 bit GGUF (Q8_0) through llama.cpp on CPU, with the adapter applied at f16. The adapter was trained on a 4 bit NF4 base, so scores can differ slightly from a GPU run.
- ROUGE and embedding similarity measure agreement with one reference answer, not clinical correctness. Nothing here was reviewed by a clinician.
- MMLU is scored zero shot from the probability of the answer letter.

## Reproduce

```bash
pip install -e ".[dev]"
pytest -q
```

The [`evaluate`](.github/workflows/evaluate.yml) workflow converts the adapter to GGUF, downloads Mistral-7B-Instruct-v0.3 (Q8_0), starts `llama-server` (llama.cpp b12445) with the adapter switchable per request, runs 14 generation shards and 6 MMLU shards, then scores everything with `python -m medeval.report` and commits `results/`.

## Layout

```
src/medeval/
  data.py      rebuild the training split, held-out flashcards, MMLU medical subsets
  prompts.py   flashcard and chat prompts, the three conditions, MMLU prompt
  run.py       query llama-server for one shard
  scoring.py   ROUGE precision, recall and F1, length, embedding similarity
  stats.py     paired bootstrap, sign test, McNemar
  report.py    results/summary.json and results/RESULTS.md
tests/         unit tests for each module
version 1 files: llm_evaluation_framework.ipynb, eval_results.csv, eval_summary.json,
                 evaluation_dashboard.html, research_analysis.md, app.py
```

## Data

- Medical Meadow flashcards: [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- MMLU: [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), subsets anatomy, clinical knowledge, college medicine, medical genetics and professional medicine

## Live demo

The [Hugging Face Space](https://huggingface.co/spaces/samurvivor-07/llm-evaluation-framework) still shows the version 1 dashboard.
