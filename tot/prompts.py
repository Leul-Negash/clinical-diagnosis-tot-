"""All the prompts in one place: decompose, propose, value, vote, synthesize,
and the IO/CoT baselines. Evaluation uses the sure/maybe/impossible scheme from
the paper. Every output is framed as educational, not medical advice."""

from __future__ import annotations

SYSTEM = (
    "You are a careful clinical-reasoning research assistant used for medical "
    "education and decision-support research. You reason like an experienced "
    "physician building a differential diagnosis: you generate competing "
    "hypotheses, weigh discriminating features, and explicitly rule things in "
    "or out. You never present yourself as a substitute for a clinician, and "
    "you always surface red flags. When asked for JSON, you output ONLY valid "
    "JSON with no commentary."
)

# 1. decompose: structure the raw case
DECOMPOSE = """Extract the salient facts from this clinical vignette so they can \
drive a differential diagnosis.

Vignette:
\"\"\"{case}\"\"\"

Return ONLY JSON of the form:
{{
  "demographics": "<age/sex/relevant background or 'unspecified'>",
  "chief_complaint": "<one line>",
  "key_findings": ["<symptom/sign/risk factor>", "..."],
  "red_flags": ["<any emergency warning sign present>", "..."]
}}"""

# 2. propose: suggest several next hypotheses at once
PROPOSE = """We are building a differential diagnosis by deliberate search.

Case facts:
{case_facts}

Reasoning so far (path of hypotheses already being explored):
{trail}

Propose {k} DISTINCT and clinically plausible next diagnostic hypotheses to \
explore from here. If the reasoning so far already names a broad category, \
refine it into more specific competing diagnoses. Favour breadth across organ \
systems and do not repeat hypotheses already in the path.

Return ONLY JSON:
{{
  "thoughts": [
    {{"hypothesis": "<specific diagnosis>", "rationale": "<why it fits these facts, 1-2 sentences>"}}
  ]
}}"""

# 3a. value: score each hypothesis on its own
VALUE = """Evaluate how well each candidate diagnosis explains the case, the way \
a clinician decides whether to keep pursuing a hypothesis.

Case facts:
{case_facts}

Candidate hypotheses to evaluate:
{candidates}

For each candidate, judge it as:
  - "sure"       -> strongly supported; clearly worth pursuing
  - "maybe"      -> plausible; keep on the differential
  - "impossible" -> effectively excluded by the facts; should be pruned
Also give a numeric score in [0,1] (probability-like, calibrated, not all high).

Return ONLY JSON:
{{
  "evaluations": [
    {{"hypothesis": "<copy exactly>", "label": "sure|maybe|impossible", "score": 0.0, "eval_notes": "<short justification>"}}
  ]
}}"""

# 3b. vote: compare states and pick the best
VOTE = """Compare these competing diagnostic states for the same case and decide \
which single one is most promising to pursue next.

Case facts:
{case_facts}

States (numbered):
{candidates}

Analyse them briefly, then choose the most promising.
Return ONLY JSON: {{"best_choice": <1-based index>, "reason": "<short>"}}"""

# 4. synthesize: write up the surviving hypotheses
SYNTHESIZE = """Using the diagnostic hypotheses that survived deliberate search, \
write the final research summary.

Case facts:
{case_facts}

Surviving ranked hypotheses (best first, with confidence):
{ranked}

Write a concise summary with these sections:
1. **Leading differential** - the top 3 diagnoses with one-line reasoning each.
2. **Discriminating next steps** - the most informative history questions, \
exam manoeuvres, or first-line investigations to separate them.
3. **Red-flag safety net** - symptoms that warrant urgent in-person care.
Begin with a one-line reminder that this is educational research output, not \
medical advice or a diagnosis."""

# baselines for the comparison view
IO_BASELINE = """Give the single most likely diagnosis for this case in one or \
two sentences (educational only).

Case:
\"\"\"{case}\"\"\""""

COT_BASELINE = """Work through this case step by step, then state the most likely \
diagnosis (educational only). Reason in a single linear chain.

Case:
\"\"\"{case}\"\"\""""
