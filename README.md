# 🔬 LLM Evaluation Framework — Medical QA

A research-grade evaluation framework comparing Base Mistral 7B vs LoRA Fine-tuned Medical model across 20 medical QA questions in 5 clinical domains.

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace%20Hub-yellow)](https://huggingface.co/samboateng190/medical-mistral-lora)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)

---

## 📊 Key Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg ROUGE |
|-------|---------|---------|---------|-----------|
| Base Mistral 7B | 0.2740 | 0.1014 | 0.2159 | 0.1971 |
| Fine-tuned LoRA | 0.3032 | 0.1262 | 0.2668 | 0.2321 |
| **Improvement** | **+0.0292** | **+0.0248** | **+0.0509** | **+17.8%** |

**Win Rate: Fine-tuned wins 13/20 questions (65%)**

---

## 📁 By Category

| Category | Base | Fine-tuned | Δ |
|----------|------|-----------|---|
| Pharmacology | 0.2280 | 0.2764 | +0.0484 |
| Pathophysiology | 0.1229 | 0.1816 | +0.0587 |
| Anatomy & Physiology | 0.1749 | 0.2271 | +0.0522 |
| Treatment | 0.2727 | 0.2823 | +0.0096 |
| Symptoms & Diagnosis | 0.1868 | 0.1928 | +0.0060 |

---

## 🏗️ Framework Architecture
---

## 📂 Files

| File | Description |
|------|-------------|
| `llm_evaluation_framework.ipynb` | Full evaluation notebook |
| `eval_results.csv` | Per-question results for both models |
| `eval_summary.json` | Aggregated results summary |
| `evaluation_dashboard.html` | Interactive Plotly dashboard |
| `research_analysis.md` | Full research write-up |

---

## 🔍 Key Findings

**Consistent gains across all 5 domains** — Fine-tuned model wins in every category confirming no catastrophic forgetting.

**Pathophysiology strongest (+47.8% relative)** — Disease mechanism questions benefited most from medical domain training.

**ROUGE limitations revealed** — 35% regression rate highlights that lexical overlap metrics don't fully capture clinical accuracy.

**LoRA efficiency** — Only 0.36% of parameters trained (13.6M / 3.7B) yet achieved +17.8% improvement.

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Models | Mistral-7B-Instruct-v0.3 + LoRA adapter |
| Evaluation | ROUGE-1, ROUGE-2, ROUGE-L |
| Visualization | Plotly |
| Framework | HuggingFace Transformers + PEFT |
| Hardware | T4 GPU — Google Colab Free |

---

## 🔗 Links

- [Fine-tuned Model on HuggingFace](https://huggingface.co/samboateng190/medical-mistral-lora)
- [Project 3 — Medical QA LoRA Fine-tune](https://github.com/Boatengs/medical-qa-lora)
- [Project 2 — SPORTZBOT RAG Chatbot](https://github.com/Boatengs/sports-rag-chatbot-)
- [Project 1 — Sentiment Analyzer](https://github.com/Boatengs/sentiment-analyzer)
