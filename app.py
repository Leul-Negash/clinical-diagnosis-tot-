"""Streamlit web UI. Same engine as the CLI - this is just a view that draws the
tree with Graphviz and offers the IO/CoT/ToT comparison.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from tot.env import load_dotenv

load_dotenv()  # pick up GROQ_API_KEY / GEMINI_API_KEY from a local .env file

from tot import (
    EvaluationStrategy,
    GenerationStrategy,
    LLMConfig,
    SearchStrategy,
    ToTConfig,
    ToTEngine,
    build_client,
    compare,
)

st.set_page_config(page_title="Clinical ToT Researcher", page_icon="🧠",
                   layout="wide")

_STATUS_COLOR = {
    "active": ("#e8f0fe", "#1a73e8"),
    "expanded": ("#e6f4ea", "#188038"),
    "solved": ("#ceead6", "#0b8043"),
    "pruned": ("#fce8e6", "#c5221f"),
}


def tree_to_dot(root: dict) -> str:
    """Render the tree-of-thoughts as Graphviz DOT, coloured by node status."""
    lines = [
        "digraph ToT {",
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=10];',
        '  edge [color="#9aa0a6"];',
        '  root [label="Case", fillcolor="#202124", fontcolor="white"];',
    ]

    def esc(s: str) -> str:
        return s.replace('"', "'").replace("\n", " ")[:48]

    def walk(node: dict, parent_id: str):
        for child in node["children"]:
            fill, border = _STATUS_COLOR.get(child["status"], ("#ffffff", "#000"))
            nid = f"n{child['id']}"
            label = f"{esc(child['content'])}\\n{child['score']:.2f} · {child['label'] or '—'}"
            style = "rounded,filled,dashed" if child["status"] == "pruned" else "rounded,filled"
            lines.append(
                f'  {nid} [label="{label}", fillcolor="{fill}", '
                f'color="{border}", style="{style}"];'
            )
            lines.append(f"  {parent_id} -> {nid};")
            walk(child, nid)

    walk(root, "root")
    lines.append("}")
    return "\n".join(lines)


# sidebar: search bounds and strategy controls
st.sidebar.title("🧠 ToT controls")
st.sidebar.caption("Yao et al. 2023 · arXiv:2305.10601")

provider = st.sidebar.selectbox(
    "LLM provider", ["mock", "groq", "gemini", "openai", "ollama"],
    help="'mock' runs fully offline. Live providers read their API key from env.",
)
st.sidebar.markdown("**Search bounds**")
depth = st.sidebar.slider("Max depth T", 1, 5, 2)
breadth = st.sidebar.slider("Breadth b (BFS)", 1, 6, 3)
k = st.sidebar.slider("Branching k", 1, 6, 4)
prune = st.sidebar.slider("Prune threshold v_th", 0.0, 1.0, 0.3, 0.05)
votes = st.sidebar.slider("Evaluator votes", 1, 5, 1)

st.sidebar.markdown("**Strategies**")
search = st.sidebar.radio("Search", ["bfs", "dfs"], horizontal=True)
generate = st.sidebar.radio("Generate", ["propose", "sample"], horizontal=True)
evaluate = st.sidebar.radio("Evaluate", ["value", "vote"], horizontal=True)


def make_cfg() -> ToTConfig:
    llm = LLMConfig().apply_preset(provider)
    return ToTConfig(
        max_depth=depth, breadth=breadth, n_generate=k,
        prune_threshold=prune, n_evaluate_votes=votes,
        search_strategy=SearchStrategy(search),
        generation_strategy=GenerationStrategy(generate),
        evaluation_strategy=EvaluationStrategy(evaluate),
        llm=llm,
    ).validate()


# main panel
st.title("Clinical Differential-Diagnosis Researcher")
st.caption("A deep-research chatbot that reasons by **deliberate tree search** "
           "over competing diagnostic hypotheses — generate, evaluate, prune, "
           "backtrack — instead of answering in one linear pass.")
st.warning("⚕️ Educational research tool — **not medical advice** and not a "
           "diagnostic device. Always consult a qualified clinician.")

case = st.text_area(
    "Describe the case",
    "58-year-old man, 2 hours of crushing central chest pain radiating to the "
    "jaw, diaphoretic, nauseated, history of hypertension and smoking.",
    height=110,
)

col_run, col_cmp = st.columns(2)
run_clicked = col_run.button("🔍 Run Tree-of-Thoughts", use_container_width=True,
                             type="primary")
cmp_clicked = col_cmp.button("⚖️ Compare IO vs CoT vs ToT", use_container_width=True)

cfg = make_cfg()
if provider != "mock" and not cfg.llm.api_key:
    st.error(f"No API key found in `${cfg.llm.api_key_env}`. "
             f"Export it before running, or pick the **mock** provider.")
    st.stop()


def render_result(result):
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Tree of thoughts explored")
        st.graphviz_chart(tree_to_dot(result.render_tree()), use_container_width=True)
        st.caption("Solid green = pursued/solved · dashed red = pruned · "
                   "blue = on the frontier")
    with right:
        st.subheader("Search telemetry")
        st.json(result.telemetry)
        st.metric("LLM calls", result.telemetry["llm_calls"])
        st.metric("Thoughts generated", result.telemetry["thoughts_generated"])
    st.subheader("Differential diagnosis (research summary)")
    st.markdown(result.summary)
    if result.red_flags:
        st.error("🚩 Red flags detected: " + ", ".join(result.red_flags))


if run_clicked and case.strip():
    with st.spinner("Deliberating over a tree of hypotheses…"):
        result = ToTEngine(cfg).run(case.strip())
    render_result(result)

if cmp_clicked and case.strip():
    with st.spinner("Running IO, CoT and ToT on the same case…"):
        client = build_client(cfg.llm)
        cmp = compare(case.strip(), cfg, client=client)
    st.subheader("IO vs CoT vs ToT")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**IO (one shot)**"); c1.info(cmp.io.answer)
    c1.caption(f"{cmp.io.llm_calls} LLM call(s)")
    c2.markdown("**CoT (linear chain)**"); c2.info(cmp.cot.answer)
    c2.caption(f"{cmp.cot.llm_calls} LLM call(s)")
    c3.markdown("**ToT (deliberate search)**"); c3.success(cmp.tot.summary)
    c3.caption(f"{cmp.tot.telemetry['llm_calls']} LLM call(s) · "
               f"{cmp.tot.telemetry['thoughts_generated']} thoughts explored")
    st.divider()
    render_result(cmp.tot)
