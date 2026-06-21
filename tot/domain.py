"""Domain adapter: clinical differential diagnosis.

All medicine-specific logic lives behind the `Domain` interface, so the search
and reasoning core stays generic. Swapping in another `Domain` retargets the
engine to a different field without touching the rest.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from . import prompts
from .llm import LLMClient
from .thought import Node


class Domain(ABC):
    """Everything ToT needs to know about a specific problem space."""

    name: str = "abstract"

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def decompose(self, client: LLMClient) -> str:
        """Turn the raw case into a compact facts string used by the prompts."""

    @abstractmethod
    def generate_messages(self, trail: list[str], k: int) -> tuple[list[dict], str]: ...

    @abstractmethod
    def parse_generation(self, payload: Any) -> list[tuple[str, str]]: ...

    @abstractmethod
    def value_messages(self, nodes: list[Node]) -> tuple[list[dict], str]: ...

    @abstractmethod
    def parse_value(self, payload: Any, nodes: list[Node]
                    ) -> dict[int, tuple[float, str, str]]: ...

    @abstractmethod
    def vote_messages(self, nodes: list[Node]) -> tuple[list[dict], str]: ...

    @abstractmethod
    def parse_vote(self, payload: Any, n: int) -> int: ...

    @abstractmethod
    def synthesize_messages(self, ranked: list[Node]) -> tuple[list[dict], str]: ...

    @abstractmethod
    def baseline_messages(self, mode: str) -> tuple[list[dict], str]: ...


class ClinicalDiagnosisDomain(Domain):
    """Differential diagnosis framed as tree search:
        thought   = a candidate diagnosis (or a narrower version of one)
        state     = case facts + the hypotheses explored so far
        generate  = propose competing/narrower diagnoses
        evaluate  = rate each as sure/maybe/impossible
        search    = keep the promising ones, drop the ruled-out, then refine.
    """

    name = "clinical-differential-diagnosis"

    # fallback score per label, used when the model gives a label but no number
    LABEL_FLOOR = {"sure": 0.8, "maybe": 0.45, "impossible": 0.05}

    def __init__(self, case: str):
        self.case = case.strip()
        self.case_facts: str = case.strip()      # filled in by decompose()
        self.structured: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def system_prompt(self) -> str:
        return prompts.SYSTEM

    def _msg(self, user: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user},
        ]

    # 1. Decomposition --------------------------------------------------- #
    def decompose(self, client: LLMClient) -> str:
        try:
            payload = client.complete_json(
                self._msg(prompts.DECOMPOSE.format(case=self.case)), tag="decompose"
            )
        except ValueError:
            payload = {}  # fall back to the raw case if structuring fails
        self.structured = payload if isinstance(payload, dict) else {}
        kf = ", ".join(self.structured.get("key_findings", []) or []) or "see vignette"
        rf = ", ".join(self.structured.get("red_flags", []) or []) or "none stated"
        self.case_facts = (
            f"Demographics: {self.structured.get('demographics', 'unspecified')}\n"
            f"Chief complaint: {self.structured.get('chief_complaint', self.case[:80])}\n"
            f"Key findings: {kf}\n"
            f"Stated red flags: {rf}"
        )
        return self.case_facts

    def red_flags(self) -> list[str]:
        return self.structured.get("red_flags", []) or []

    # 2. Generation ------------------------------------------------------ #
    def generate_messages(self, trail: list[str], k: int) -> tuple[list[dict], str]:
        trail_text = (
            "\n".join(f"  - {t}" for t in trail) if trail else "  (none yet)"
        )
        user = prompts.PROPOSE.format(
            case_facts=self.case_facts, trail=trail_text, k=k
        )
        return self._msg(user), "propose"

    def parse_generation(self, payload: Any) -> list[tuple[str, str]]:
        thoughts = payload.get("thoughts", []) if isinstance(payload, dict) else []
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for t in thoughts:
            if not isinstance(t, dict):
                continue
            name = str(t.get("hypothesis", "")).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            out.append((name, str(t.get("rationale", "")).strip()))
        return out

    # 3a. Value evaluation ---------------------------------------------- #
    def value_messages(self, nodes: list[Node]) -> tuple[list[dict], str]:
        candidates = "\n".join(f"  - {n.content}" for n in nodes)
        user = prompts.VALUE.format(
            case_facts=self.case_facts, candidates=candidates
        )
        return self._msg(user), "value"

    def parse_value(self, payload: Any, nodes: list[Node]
                    ) -> dict[int, tuple[float, str, str]]:
        evals = payload.get("evaluations", []) if isinstance(payload, dict) else []
        by_name = {n.content.lower(): n for n in nodes}
        result: dict[int, tuple[float, str, str]] = {}
        for e in evals:
            if not isinstance(e, dict):
                continue
            name = str(e.get("hypothesis", "")).strip().lower()
            node = by_name.get(name) or _fuzzy_match(name, by_name)
            if node is None:
                continue
            label = str(e.get("label", "maybe")).lower()
            floor = self.LABEL_FLOOR.get(label, 0.45)
            try:
                score = float(e.get("score", floor))
            except (TypeError, ValueError):
                score = floor
            # average the model's number with the label floor so the two agree
            score = max(0.0, min(1.0, 0.5 * score + 0.5 * floor))
            result[node.id] = (score, label, str(e.get("eval_notes", "")).strip())
        # Any node the model skipped gets a neutral default.
        for n in nodes:
            result.setdefault(n.id, (0.4, "maybe", "not explicitly scored"))
        return result

    # 3b. Vote evaluation ----------------------------------------------- #
    def vote_messages(self, nodes: list[Node]) -> tuple[list[dict], str]:
        candidates = "\n".join(
            f"  {i+1}. {n.content}" for i, n in enumerate(nodes)
        )
        user = prompts.VOTE.format(
            case_facts=self.case_facts, candidates=candidates
        )
        return self._msg(user), "vote"

    def parse_vote(self, payload: Any, n: int) -> int:
        if isinstance(payload, dict):
            try:
                idx = int(payload.get("best_choice", 1)) - 1
            except (TypeError, ValueError):
                idx = 0
            return max(0, min(n - 1, idx))
        return 0

    # 4. Synthesis ------------------------------------------------------- #
    def synthesize_messages(self, ranked: list[Node]) -> tuple[list[dict], str]:
        lines = "\n".join(
            f"  {i+1}. {n.content} (confidence {n.score:.0%}; {n.label or 'n/a'})"
            for i, n in enumerate(ranked)
        )
        user = prompts.SYNTHESIZE.format(case_facts=self.case_facts, ranked=lines)
        return self._msg(user), "synthesize"

    # Baselines ---------------------------------------------------------- #
    def baseline_messages(self, mode: str) -> tuple[list[dict], str]:
        if mode == "io":
            return self._msg(prompts.IO_BASELINE.format(case=self.case)), "io"
        if mode == "cot":
            return self._msg(prompts.COT_BASELINE.format(case=self.case)), "cot"
        raise ValueError(f"Unknown baseline mode: {mode}")


def _fuzzy_match(name: str, by_name: dict[str, Node]) -> Node | None:
    """Tolerate minor wording drift between proposal and evaluation."""
    for key, node in by_name.items():
        if name and (name in key or key in name):
            return node
    return None
