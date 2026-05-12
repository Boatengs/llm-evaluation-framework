import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load Results ───────────────────────────────────────────────
df = pd.read_csv("eval_results.csv")

# ── Build Dashboard ────────────────────────────────────────────
def build_dashboard():
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Overall ROUGE Score Comparison",
            "Performance by Category",
            "Question-level Improvement (Δ ROUGE)",
            "Win Rate Distribution"
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "pie"}]
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    colors = {"base": "#636EFA", "ft": "#00CC96", "pos": "#00CC96", "neg": "#EF553B"}

    # Plot 1 — Overall
    for name, col, key in [("Base Mistral 7B", colors["base"], "base"), ("Fine-tuned", colors["ft"], "ft")]:
        fig.add_trace(go.Bar(
            name=name,
            x=["ROUGE-1", "ROUGE-2", "ROUGE-L", "Avg"],
            y=[df[f'{key}_rouge1'].mean(), df[f'{key}_rouge2'].mean(),
               df[f'{key}_rougeL'].mean(), df[f'{key}_avg_rouge'].mean()],
            marker_color=col,
        ), row=1, col=1)

    # Plot 2 — Per category
    cat_groups = df.groupby('category').agg(
        base_avg=('base_avg_rouge', 'mean'),
        ft_avg=('ft_avg_rouge', 'mean')
    ).reset_index()

    fig.add_trace(go.Bar(name="Base Mistral 7B", x=cat_groups['category'],
        y=cat_groups['base_avg'], marker_color=colors["base"], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(name="Fine-tuned", x=cat_groups['category'],
        y=cat_groups['ft_avg'], marker_color=colors["ft"], showlegend=False), row=1, col=2)

    # Plot 3 — Per question
    imp_colors = [colors["pos"] if x > 0 else colors["neg"] for x in df['improvement']]
    fig.add_trace(go.Bar(
        x=list(range(1, 21)), y=df['improvement'].tolist(),
        marker_color=imp_colors, showlegend=False,
        hovertext=[q[:50] for q in df['question']], hoverinfo="text+y"
    ), row=2, col=1)

    # Plot 4 — Win rate
    wins   = (df['improvement'] > 0).sum()
    losses = (df['improvement'] <= 0).sum()
    fig.add_trace(go.Pie(
        labels=["Fine-tuned Wins", "Base Wins"],
        values=[wins, losses],
        marker_colors=[colors["pos"], colors["neg"]],
        hole=0.4, textinfo="label+percent"
    ), row=2, col=2)

    fig.update_layout(
        height=650, barmode="group", template="plotly_dark",
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font={"color": "white", "size": 11},
        title={"text": "🔬 Medical LLM Evaluation — Base Mistral 7B vs Fine-tuned LoRA", "x": 0.5}
    )
    fig.update_xaxes(tickangle=-25, row=1, col=2)
    fig.update_xaxes(title_text="Question #", row=2, col=1)

    return fig

def get_details(question_idx):
    if question_idx is None:
        return "", "", ""
    row = df.iloc[int(question_idx)]
    q        = f"**Question:** {row['question']}"
    base_ans = f"**Base Model Answer:**\n\n{row['base_answer']}"
    ft_ans   = f"**Fine-tuned Answer:**\n\n{row['ft_answer']}\n\n---\n**ROUGE Improvement: {row['improvement']:+.4f}**"
    return q, base_ans, ft_ans

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
    --bg-primary:#ffffff;--bg-card:#ffffff;--bg-elevated:#f1f5f9;
    --border:rgba(0,0,0,0.08);--text-primary:#0f172a;--text-secondary:#475569;
    --text-tertiary:#94a3b8;--accent:#6366f1;--accent-dim:rgba(99,102,241,0.1);
    --positive:#10b981;--negative:#ef4444;
    --shadow:0 4px 6px -1px rgba(0,0,0,0.07);--shadow-lg:0 10px 15px -3px rgba(0,0,0,0.08);
    --radius:16px;
}
[data-theme="dark"] {
    --bg-primary:#0f172a;--bg-card:#1e293b;--bg-elevated:#334155;
    --border:rgba(255,255,255,0.08);--text-primary:#f1f5f9;
    --text-secondary:#94a3b8;--text-tertiary:#475569;
    --shadow:0 4px 6px -1px rgba(0,0,0,0.3);--shadow-lg:0 10px 15px -3px rgba(0,0,0,0.4);
}
* { font-family:"Inter",sans-serif !important; box-sizing:border-box; }
body,.gradio-container { background:var(--bg-primary) !important; color:var(--text-primary) !important; transition:all 0.3s ease; }
.gradio-container { max-width:1100px !important; margin:0 auto !important; padding:24px !important; }
.app-header { text-align:center; padding:52px 24px 36px; background:linear-gradient(135deg,var(--accent-dim),transparent); border-radius:var(--radius); border:1px solid var(--border); margin-bottom:28px; }
.app-header h1 { font-size:2.4em !important; font-weight:900 !important; background:linear-gradient(135deg,var(--accent),#8b5cf6,#10b981); -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; margin-bottom:14px !important; }
.app-header p { color:var(--text-secondary) !important; font-size:1em !important; max-width:650px; margin:0 auto !important; line-height:1.8 !important; }
.badges { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:20px; }
.badge { padding:6px 16px; border-radius:20px; font-size:0.75em; font-weight:600; border:1px solid var(--border); background:var(--bg-elevated); color:var(--text-secondary); }
.badge.accent { background:var(--accent-dim); color:var(--accent); border-color:rgba(99,102,241,0.25); }
.badge.positive { background:rgba(16,185,129,0.1); color:#10b981; border-color:rgba(16,185,129,0.25); }
.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }
.stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:20px 16px; text-align:center; box-shadow:var(--shadow); transition:all 0.2s ease; }
.stat-card:hover { transform:translateY(-2px); box-shadow:var(--shadow-lg); }
.stat-value { font-size:1.9em; font-weight:900; color:var(--accent); line-height:1; }
.stat-label { font-size:0.7em; color:var(--text-tertiary); font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-top:6px; }
.card { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:var(--radius) !important; padding:24px !important; box-shadow:var(--shadow) !important; }
.theme-toggle { position:fixed !important; top:20px !important; right:20px !important; background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:50% !important; width:46px !important; height:46px !important; cursor:pointer !important; font-size:1.2em !important; box-shadow:var(--shadow-lg) !important; transition:all 0.2s !important; z-index:1000 !important; }
.theme-toggle:hover { transform:scale(1.12) rotate(15deg) !important; }
label { color:var(--text-secondary) !important; font-weight:700 !important; font-size:0.78em !important; text-transform:uppercase !important; letter-spacing:1px !important; }
.footer { text-align:center; padding:28px; color:var(--text-tertiary); font-size:0.78em; border-top:1px solid var(--border); margin-top:36px; line-height:2; }
"""

js = """
function toggleTheme(){
    const root=document.documentElement;
    const btn=document.getElementById("theme-btn");
    const dark=root.getAttribute("data-theme")==="dark";
    root.setAttribute("data-theme",dark?"light":"dark");
    btn.textContent=dark?"🌙":"☀️";
    localStorage.setItem("eval-theme",dark?"light":"dark");
}
document.addEventListener("DOMContentLoaded",()=>{
    const saved=localStorage.getItem("eval-theme")||"light";
    document.documentElement.setAttribute("data-theme",saved);
    const btn=document.getElementById("theme-btn");
    if(btn)btn.textContent=saved==="dark"?"☀️":"🌙";
});
"""

with gr.Blocks(css=css, title="🔬 LLM Evaluation Framework") as demo:

    gr.HTML(f"""
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">🌙</button>
    <script>{js}</script>
    <div class="app-header">
        <h1>🔬 LLM Evaluation Framework</h1>
        <p>Research-grade evaluation comparing Base Mistral 7B vs LoRA Fine-tuned Medical AI
        across 20 questions in 5 clinical domains. Interactive results dashboard.</p>
        <div class="badges">
            <span class="badge accent">Mistral 7B</span>
            <span class="badge accent">LoRA Fine-tuned</span>
            <span class="badge positive">+17.8% ROUGE</span>
            <span class="badge positive">65% Win Rate</span>
            <span class="badge">5 Medical Domains</span>
            <span class="badge">20 Questions</span>
        </div>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">+17.8%</div>
            <div class="stat-label">ROUGE Improvement</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">65%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">20</div>
            <div class="stat-label">Questions Evaluated</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">5</div>
            <div class="stat-label">Medical Domains</div>
        </div>
    </div>
    """)

    with gr.Column(elem_classes=["card"]):
        gr.Markdown("### 📊 Evaluation Dashboard")
        dashboard_plot = gr.Plot(value=build_dashboard())

    with gr.Column(elem_classes=["card"]):
        gr.Markdown("### 🔍 Question-level Analysis")
        question_slider = gr.Slider(
            minimum=0, maximum=19, step=1, value=0,
            label="Select Question (0-19)"
        )
        with gr.Row():
            question_out  = gr.Markdown()
        with gr.Row():
            base_out = gr.Markdown(label="Base Model")
            ft_out   = gr.Markdown(label="Fine-tuned Model")

    gr.HTML("""
    <div style="margin-top:28px;background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:28px;box-shadow:var(--shadow);">
        <h3 style="color:var(--text-primary);margin-bottom:20px;font-size:1em;font-weight:700;">📋 Evaluation Categories</h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;">
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;text-align:center;">
                <div style="font-size:1.8em;margin-bottom:8px;">💊</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.85em;">Pharmacology</div>
                <div style="color:var(--text-tertiary);font-size:0.75em;margin-top:4px;">4 questions</div>
            </div>
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;text-align:center;">
                <div style="font-size:1.8em;margin-bottom:8px;">🩺</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.85em;">Symptoms & Diagnosis</div>
                <div style="color:var(--text-tertiary);font-size:0.75em;margin-top:4px;">4 questions</div>
            </div>
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;text-align:center;">
                <div style="font-size:1.8em;margin-bottom:8px;">🧬</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.85em;">Pathophysiology</div>
                <div style="color:var(--text-tertiary);font-size:0.75em;margin-top:4px;">4 questions</div>
            </div>
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;text-align:center;">
                <div style="font-size:1.8em;margin-bottom:8px;">💉</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.85em;">Treatment</div>
                <div style="color:var(--text-tertiary);font-size:0.75em;margin-top:4px;">4 questions</div>
            </div>
            <div style="padding:16px;background:var(--bg-elevated);border-radius:12px;text-align:center;">
                <div style="font-size:1.8em;margin-bottom:8px;">🫀</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.85em;">Anatomy & Physiology</div>
                <div style="color:var(--text-tertiary);font-size:0.75em;margin-top:4px;">4 questions</div>
            </div>
        </div>
    </div>

    <div class="footer">
        Base Mistral 7B vs LoRA Fine-tuned • ROUGE-1, ROUGE-2, ROUGE-L Metrics •
        20 Medical QA Questions • 5 Clinical Domains •
        <a href="https://github.com/Boatengs/llm-evaluation-framework" target="_blank" style="color:var(--accent);">GitHub</a> •
        <a href="https://huggingface.co/samboateng190/medical-mistral-lora" target="_blank" style="color:var(--accent);">Fine-tuned Model</a>
    </div>
    """)

    question_slider.change(
        fn=get_details,
        inputs=question_slider,
        outputs=[question_out, base_out, ft_out]
    )

demo.launch()
