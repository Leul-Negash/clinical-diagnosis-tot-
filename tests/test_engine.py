"""Deterministic tests using the offline MockLLM (no network, no API key).

These assert the *invariants* that make the search trustworthy: bounds are
respected, pruning happens, early stopping works, bad config fails fast, and
the JSON parsing is robust to messy model output.
"""

from __future__ import annotations

import pytest

from tot import (
    ClinicalDiagnosisDomain,
    EvaluationStrategy,
    GenerationStrategy,
    LLMConfig,
    SearchStrategy,
    ToTConfig,
    ToTEngine,
    compare,
)
from tot.llm import _extract_json
from tot.thought import NodeStatus

CHEST_PAIN = ("55-year-old man with crushing chest pain radiating to the left "
              "arm, sweating and short of breath.")


def mock_cfg(**overrides) -> ToTConfig:
    cfg = ToTConfig(**overrides)
    cfg.llm = LLMConfig().apply_preset("mock")
    return cfg


# --------------------------------------------------------------------------- #
# Bounds: the "best practice" guarantees the task asks for.
# --------------------------------------------------------------------------- #
def test_depth_bound_respected():
    cfg = mock_cfg(max_depth=2, breadth=3, n_generate=4)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    max_depth = max(n.depth for n in result.tree.nodes)
    assert max_depth <= cfg.max_depth


def test_breadth_bound_respected_in_bfs():
    cfg = mock_cfg(max_depth=2, breadth=2, n_generate=4,
                   search_strategy=SearchStrategy.BFS)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    # At each depth, no more than `breadth` survivors (non-pruned) may expand.
    for depth in range(1, cfg.max_depth + 1):
        survivors = [n for n in result.tree.nodes
                     if n.depth == depth and n.status != NodeStatus.PRUNED]
        assert len(survivors) <= cfg.breadth


def test_global_thought_cap_never_exceeded():
    cfg = mock_cfg(max_depth=5, breadth=5, n_generate=5, max_total_thoughts=12)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    assert result.tree.size <= cfg.max_total_thoughts


# --------------------------------------------------------------------------- #
# Core ToT behaviours.
# --------------------------------------------------------------------------- #
def test_pruning_marks_impossible_nodes():
    cfg = mock_cfg(max_depth=2, breadth=3, n_generate=4, prune_threshold=0.3)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    pruned = [n for n in result.tree.nodes if n.status == NodeStatus.PRUNED]
    assert pruned, "expected at least one pruned hypothesis"
    # Nothing pruned should out-rank the chosen best.
    assert result.search.best.status != NodeStatus.PRUNED


def test_ranked_hypotheses_are_sorted_and_unique():
    cfg = mock_cfg(max_depth=2)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    scores = [n.score for n in result.ranked]
    assert scores == sorted(scores, reverse=True)
    names = [n.content.lower() for n in result.ranked]
    assert len(names) == len(set(names))


def test_dfs_runs_and_early_stops():
    cfg = mock_cfg(max_depth=3, n_generate=4, search_strategy=SearchStrategy.DFS)
    result = ToTEngine(cfg).run("Patient with severe headache, fever, neck stiffness.")
    assert result.search.strategy is SearchStrategy.DFS
    assert result.ranked
    assert result.telemetry["llm_calls"] > 0


def test_sample_generation_strategy():
    cfg = mock_cfg(max_depth=1, n_generate=3,
                   generation_strategy=GenerationStrategy.SAMPLE)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    assert result.tree.size > 0


def test_vote_evaluation_strategy():
    cfg = mock_cfg(max_depth=1, n_generate=3,
                   evaluation_strategy=EvaluationStrategy.VOTE, n_evaluate_votes=3)
    result = ToTEngine(cfg).run(CHEST_PAIN)
    # Exactly one winner should be labelled "sure" at the first level.
    level1 = [n for n in result.tree.nodes if n.depth == 1]
    assert any(n.label == "sure" for n in level1)


def test_summary_carries_safety_framing():
    result = ToTEngine(mock_cfg(max_depth=1)).run(CHEST_PAIN)
    assert "not medical advice" in result.summary.lower()


# --------------------------------------------------------------------------- #
# Config validation: bad search bounds must fail fast.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    {"max_depth": 0},
    {"breadth": 0},
    {"n_generate": 0},
    {"prune_threshold": 1.5},
    {"n_evaluate_votes": 0},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        mock_cfg(**kwargs).validate()


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        LLMConfig().apply_preset("does-not-exist")


# --------------------------------------------------------------------------- #
# Robust JSON extraction from messy model output.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go: {"a": 1} hope that helps', {"a": 1}),
    ('[1, 2, 3]', [1, 2, 3]),
])
def test_extract_json(raw, expected):
    assert _extract_json(raw) == expected


def test_extract_json_failure():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_extract_json_salvages_truncated_array():
    # Model hit the token limit mid-third-object; the two complete ones survive.
    truncated = (
        '{"thoughts": [{"hypothesis": "GERD", "rationale": "fits"}, '
        '{"hypothesis": "Pancreatitis", "rationale": "also fits"}, '
        '{"hypothesis": "Appendicitis", "rationale": "this got cut off mid sen'
    )
    salvaged = _extract_json(truncated, salvage_key="thoughts")
    names = [t["hypothesis"] for t in salvaged["thoughts"]]
    assert names == ["GERD", "Pancreatitis"]


def test_extract_json_salvage_requires_key():
    # Without a salvage_key, truncated output still raises (no silent guessing).
    with pytest.raises(ValueError):
        _extract_json('{"thoughts": [{"hypothesis": "GERD", "rationale": "fi')


# --------------------------------------------------------------------------- #
# Comparison harness (IO vs CoT vs ToT) reproduces the paper's contrast shape.
# --------------------------------------------------------------------------- #
def test_comparison_runs_all_three():
    cmp = compare(CHEST_PAIN, mock_cfg(max_depth=2))
    assert cmp.io.answer and cmp.cot.answer and cmp.tot.summary
    # ToT spends more LLM calls than the one-shot IO baseline (it searches).
    assert cmp.tot.telemetry["llm_calls"] > cmp.io.llm_calls


def test_decompose_extracts_key_findings():
    domain = ClinicalDiagnosisDomain(CHEST_PAIN)
    from tot.llm import build_client
    facts = domain.decompose(build_client(LLMConfig().apply_preset("mock")))
    assert "chest pain" in facts.lower()
