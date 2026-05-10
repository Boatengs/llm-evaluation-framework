# 🔬 LLM Evaluation Framework — Medical QA

A research-grade evaluation framework comparing Base Mistral 7B vs LoRA Fine-tuned Medical model across 20 medical QA questions in 5 clinical domains.

## 📊 Key Results

- Base Mistral 7B avg ROUGE: 0.1971
- Fine-tuned LoRA avg ROUGE: 0.2321
- Overall improvement: +17.8%
- Win rate: Fine-tuned wins 13/20 questions (65%)

## 📁 Performance by Category

- Pharmacology: Base 0.2280 → Fine-tuned 0.2764 (+0.0484)
- Pathophysiology: Base 0.1229 → Fine-tuned 0.1816 (+0.0587)
- Anatomy & Physiology: Base 0.1749 → Fine-tuned 0.2271 (+0.0522)
- Treatment: Base 0.2727 → Fine-tuned 0.2823 (+0.0096)
- Symptoms & Diagnosis: Base 0.1868 → Fine-tuned 0.1928 (+0.0060)

## 📂 Files

- llm_evaluation_framework.ipynb — Full evaluation notebook
- eval_results.csv — Per-question results for both models
- eval_summary.json — Aggregated results summary
- evaluation_dashboard.html — Interactive Plotly dashboard
- research_analysis.md — Full research write-up

## 🔍 Key Findings

1. Consistent gains across all 5 domains — Fine-tuned model wins in every category
2. Pathophysiology strongest (+47.8% relative improvement)
3. ROUGE limitations revealed — 35% regression rate highlights lexical overlap limits
4. LoRA efficiency — Only 0.36% of parameters trained yet +17.8% improvement

## 🛠️ Tech Stack

- Models: Mistral-7B-Instruct-v0.3 + LoRA adapter
- Evaluation: ROUGE-1, ROUGE-2, ROUGE-L
- Visualization: Plotly interactive dashboard
- Framework: HuggingFace Transformers + PEFT
- Hardware: T4 GPU — Google Colab Free Tier

## 🔗 Links

- Fine-tuned Model: https://huggingface.co/samboateng190/medical-mistral-lora
- Project 3 Medical QA LoRA: https://github.com/Boatengs/medical-qa-lora
- Project 2 SPORTZBOT RAG: https://github.com/Boatengs/sports-rag-chatbot-
- Project 1 Sentiment Analyzer: https://github.com/Boatengs/sentiment-analyzer
