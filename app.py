"""Version 2 dashboard for the medical LoRA evaluation (Hugging Face Space).

Reads results/summary.json and results/raw/answers_scored.csv. Run locally from
the repository root, or from the Space, which sets APP_REF to a pinned commit
and reads the files from GitHub.
"""
import io
import json
import urllib.request
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

REPO_RAW = "https://raw.githubusercontent.com/Boatengs/llm-evaluation-framework"
REF = globals().get("APP_REF", "main")


def read_text(rel: str) -> str:
    local = Path(rel)
    if local.exists():
        return local.read_text()
    with urllib.request.urlopen(f"{REPO_RAW}/{REF}/{rel}", timeout=60) as r:
        return r.read().decode("utf-8")


summary = json.loads(read_text("results/summary.json"))
answers = pd.read_csv(io.StringIO(read_text("results/raw/answers_scored.csv")))
held = answers[answers["set"] == "heldout"]

CONDS = [("base_flashcard", "Base, flashcard prompt", "#94a3b8"),
         ("base_chat", "Base, chat prompt", "#64748b"),
         ("finetuned", "Fine tuned", "#2563eb")]
H = summary["heldout"]
M = summary["mmlu"]
PAIR = H["pairs"]["base_flashcard__finetuned"]


def metrics_chart():
    metrics = [("avg_rouge_f", "Average ROUGE F1"), ("rougeL_p", "ROUGE-L precision"),
               ("rougeL_r", "ROUGE-L recall"), ("embed_sim", "Embedding similarity")]
    fig = go.Figure()
    for key, label, color in CONDS:
        fig.add_bar(name=label, x=[m[1] for m in metrics], marker_color=color,
                    y=[H["conditions"][key][m[0]] for m in metrics],
                    text=[f"{H['conditions'][key][m[0]]:.3f}" for m in metrics], textposition="outside")
    fig.update_layout(barmode="group", template="plotly_white", height=420, yaxis_range=[0, 1],
                      title=f"Held-out flashcards ({H['n']} questions): precision doubles, recall does not move",
                      legend={"orientation": "h", "y": -0.15}, margin={"t": 60})
    return fig


def length_chart():
    fig = go.Figure()
    for key, label, color in CONDS:
        fig.add_box(y=held[held.condition == key]["words"], name=label, marker_color=color, boxpoints=False)
    fig.add_hline(y=H["median_reference_words"], line_dash="dash", line_color="#16a34a",
                  annotation_text=f"Median reference: {H['median_reference_words']:g} words", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=400, showlegend=False,
                      title="Answer length in words: the fine tune writes flashcard length answers")
    return fig


def mmlu_chart():
    subj = M["by_subject"]
    names = [s.replace("_", " ").capitalize() for s in subj]
    fig = go.Figure()
    fig.add_bar(name="Base", x=names, y=[v["base"] * 100 for v in subj.values()], marker_color="#94a3b8",
                text=[f"{v['base'] * 100:.1f}%" for v in subj.values()], textposition="outside")
    fig.add_bar(name="Fine tuned", x=names, y=[v["finetuned"] * 100 for v in subj.values()], marker_color="#2563eb",
                text=[f"{v['finetuned'] * 100:.1f}%" for v in subj.values()], textposition="outside")
    fig.update_layout(barmode="group", template="plotly_white", height=420, yaxis_title="Accuracy (%)",
                      yaxis_range=[0, 100], legend={"orientation": "h", "y": -0.2},
                      title=f"MMLU medical ({M['n']} questions): lower in every subject after fine tuning")
    return fig


QUESTIONS = held.drop_duplicates("id")[["id", "question"]].reset_index(drop=True)
CHOICES = [f"{i + 1:03d}. {q[:110]}" for i, q in enumerate(QUESTIONS["question"])]


def show(choice):
    if not choice:
        return "", "", "", ""
    qid = QUESTIONS.loc[int(choice.split(".")[0]) - 1, "id"]
    rows = held[held.id == qid].set_index("condition")
    head = f"**Question:** {rows['question'].iloc[0]}\n\n**Reference answer:** {rows['reference'].iloc[0]}"
    out = []
    for key, label, _ in CONDS:
        r = rows.loc[key]
        out.append(f"**{label}** · {int(r['words'])} words · average ROUGE {r['avg_rouge_f']:.3f} · "
                   f"similarity {r['embed_sim']:.3f}\n\n{r['answer']}")
    return head, *out


d = PAIR["avg_rouge_f"]
intro = f"""
# Medical LoRA evaluation, version 2

Mistral-7B-Instruct-v0.3 against its LoRA fine tune ([adapter](https://huggingface.co/samurvivor-07/medical-mistral-lora)),
tested on **{H['n']} held-out Medical Meadow flashcards** and **{M['n']} MMLU medical questions**.

- **Flashcards:** average ROUGE {H['conditions']['base_flashcard']['avg_rouge_f']:.3f} → {H['conditions']['finetuned']['avg_rouge_f']:.3f}
  ({d['diff']:+.3f}, 95% interval {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}). ROUGE-L recall did not change, so most of the gain is concise style.
- **MMLU medical:** accuracy {M['accuracy']['base'] * 100:.1f}% → {M['accuracy']['finetuned'] * 100:.1f}%
  (McNemar p = {M['mcnemar']['p_value']:.3f}): fine tuning cost some general medical knowledge.

Code, every answer and every test: [GitHub](https://github.com/Boatengs/llm-evaluation-framework).
Research evaluation only; nothing here is medical advice.
"""

with gr.Blocks(title="Medical LoRA Evaluation") as demo:
    gr.Markdown(intro)
    with gr.Tab("Results"):
        gr.Plot(metrics_chart())
        gr.Plot(length_chart())
        gr.Plot(mmlu_chart())
    with gr.Tab("Browse the answers"):
        pick = gr.Dropdown(CHOICES, label=f"Pick one of the {H['n']} held-out questions", value=CHOICES[0])
        head = gr.Markdown()
        with gr.Row():
            a, b, c = gr.Markdown(), gr.Markdown(), gr.Markdown()
        pick.change(show, pick, [head, a, b, c])
        demo.load(show, pick, [head, a, b, c])

if __name__ == "__main__" or "APP_REF" in globals():
    demo.launch()
