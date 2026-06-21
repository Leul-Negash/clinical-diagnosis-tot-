# Clinical Differential-Diagnosis Researcher

A research chatbot that works through a clinical case using **Tree-of-Thoughts
(ToT)** reasoning, based on Yao et al., 2023, *Tree of Thoughts: Deliberate
Problem Solving with Large Language Models* ([arXiv:2305.10601](https://arxiv.org/abs/2305.10601)).

Instead of answering in a single pass, it reasons the way a clinician builds a
differential: it proposes competing diagnoses, weighs each against the case,
rules out the ones that don't fit, backtracks, and then writes up what's left.

> **Educational/research tool — not medical advice and not a diagnostic device.**
> Always consult a qualified clinician.

---

## Contents

- [Why differential diagnosis](#why-differential-diagnosis)
- [How it maps to the paper](#how-it-maps-to-the-paper)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Command-line options](#command-line-options)
- [Project layout](#project-layout)
- [Tests](#tests)

---

## Why differential diagnosis

ToT treats problem solving as search over a tree of intermediate "thoughts",
with generation, evaluation, pruning, and backtracking. Differential diagnosis
is the same loop in clinical form, so the fit is direct:

| ToT | Here |
|-----|------|
| Thought | A candidate diagnosis (or a narrower version of one) |
| State | The case + the hypotheses explored so far |
| Generate | Propose competing / more specific diagnoses |
| Evaluate | Rate each as sure / maybe / impossible |
| Prune | Rule a diagnosis out and stop exploring it |
| Search | BFS or DFS over the hypothesis tree |

The paper's main result is that this kind of search beats one-shot decoding
(Game of 24: chain-of-thought 4% vs ToT 74%). The `--compare` mode lets you see
the same contrast on a clinical case: IO and CoT commit to one line, ToT
explores and prunes.

---

## How it maps to the paper

All four ToT design choices are implemented as separate, swappable parts:

1. **Thought decomposition** — `tot/domain.py` turns the raw case into facts.
2. **Thought generator** — `tot/generator.py`, with `propose` and `sample`.
3. **State evaluator** — `tot/evaluator.py`, with `value` (sure/maybe/impossible)
   and `vote`, each able to average over several samples.
4. **Search** — `tot/search.py`: BFS (Algorithm 1) and DFS with pruning and
   backtracking (Algorithm 2).

The search is always bounded, set in `tot/config.py`:

| Setting | Paper | Default | What it does |
|---------|-------|---------|--------------|
| `max_depth` | T | 2 | how deep the tree can go |
| `breadth` | b | 3 | states kept per level (BFS) |
| `n_generate` | k | 4 | hypotheses generated per state |
| `prune_threshold` | v_th | 0.3 | drop states scoring below this |
| `n_evaluate_votes` | — | 1 | evaluator samples to average |
| `max_total_thoughts` | — | 60 | hard cap on total nodes per query |
| `solved_threshold` | — | 0.9 | stop deepening once this confident |

`ToTConfig.validate()` rejects bad values, so a misconfigured search can't run
away or hang.

---

## Setup

```bash
cd clinical-diagnosis-tot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a live LLM (optional — the mock provider needs none of this), add a key to
a `.env` file:

```bash
cp .env.example .env
# open .env and set GROQ_API_KEY=...   (free, no card: https://console.groq.com/keys)
```

`.env` is loaded automatically. An `export GROQ_API_KEY=...` also works and
takes priority over the file.

---

## How to run

**Offline (no key needed):**

```bash
python cli.py --provider mock
```

**Live on Groq's free tier:**

```bash
python cli.py --provider groq
```

**One case, then exit:**

```bash
python cli.py --provider groq --case "58M, 2h crushing chest pain to the jaw, sweaty, history of smoking"
```

**Compare IO vs CoT vs ToT on the same case:**

```bash
python cli.py --provider groq --compare --case "32F pleuritic chest pain after a long flight"
```

**Web UI (interactive tree + comparison):**

```bash
streamlit run app.py
```

Tip: richer case descriptions give sharper trees — a few symptoms, some history,
and any red flags work better than a single word.

---

## Command-line options

```
--provider   mock | groq | gemini | openai | ollama   (default: mock)
--model      override the provider's default model
--search     bfs | dfs              (default: bfs)
--generate   propose | sample       (default: propose)
--evaluate   value | vote           (default: value)
--depth      max depth T            (default: 2)
--breadth    states per level b     (default: 3)
--k          hypotheses per state   (default: 4)
--prune      prune threshold v_th   (default: 0.3)
--votes      evaluator samples      (default: 1)
--case       run one case and exit
--compare    show IO vs CoT vs ToT
```

Example — deeper DFS with stricter pruning and 3-sample evaluation:

```bash
python cli.py --provider groq --search dfs --depth 3 --prune 0.4 --votes 3 \
    --case "70M, sudden worst-ever headache, vomiting, photophobia"
```

---

## Project layout

```
tot/
  config.py      settings for the LLM and the search
  llm.py         OpenAI-compatible client + offline mock
  thought.py     Node / Tree
  domain.py      Domain interface + clinical diagnosis domain
  prompts.py     the prompts
  generator.py   propose / sample
  evaluator.py   value / vote
  search.py      BFS and DFS
  engine.py      ties it together: decompose -> search -> synthesize
  baselines.py   IO and CoT, for the comparison
  env.py         .env loader
cli.py           terminal app with a live tree
app.py           Streamlit web app
tests/           offline tests (mock provider)
```

The search and reasoning code is generic — all the medical parts sit behind the
`Domain` interface, so it can be pointed at another field by writing a new
`Domain`.

---

## Tests

```bash
pytest -q
```

All tests run offline with the mock provider, so no key is required.

---

## Reference

Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan. *Tree of Thoughts:
Deliberate Problem Solving with Large Language Models.* NeurIPS 2023.
arXiv:2305.10601.
