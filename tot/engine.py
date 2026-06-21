"""The engine: decompose -> search (generate/evaluate/prune/backtrack) ->
synthesize. It explores a tree of competing hypotheses, prunes the dead ends,
and writes up what survives instead of answering in a single pass."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .config import ToTConfig
from .domain import ClinicalDiagnosisDomain, Domain
from .evaluator import StateEvaluator
from .generator import ThoughtGenerator
from .llm import LLMClient, build_client
from .search import EventCallback, SearchResult, Searcher
from .thought import Node, NodeStatus, Tree


@dataclass
class EngineResult:
    case: str
    case_facts: str
    summary: str
    ranked: list[Node]
    tree: Tree
    search: SearchResult
    red_flags: list[str]
    telemetry: dict
    config: dict
    elapsed_s: float = 0.0

    def render_tree(self) -> dict:
        return self.tree.to_render_dict()


class ToTEngine:
    """Runs Tree-of-Thoughts deliberate reasoning for a given `Domain`."""

    def __init__(
        self,
        cfg: ToTConfig,
        client: Optional[LLMClient] = None,
        domain_factory: Callable[[str], Domain] = ClinicalDiagnosisDomain,
        on_event: EventCallback = None,
    ):
        self.cfg = cfg.validate()
        self.client = client or build_client(cfg.llm)
        self.domain_factory = domain_factory
        self.on_event = on_event

    def run(self, case: str) -> EngineResult:
        t0 = time.time()
        self.client.telemetry.reset()
        domain = self.domain_factory(case)

        # 1. Thought decomposition: structure the case for every later prompt.
        case_facts = domain.decompose(self.client)

        # 2-4. Build the root state, then search the tree of thoughts.
        tree = Tree(root_content=case_facts)
        generator = ThoughtGenerator(self.client, domain)
        evaluator = StateEvaluator(self.client, domain)
        searcher = Searcher(tree, generator, evaluator, self.cfg, self.on_event)
        search_result = searcher.run()

        # Rank the surviving hypotheses across the whole tree.
        ranked = self._rank(tree)

        # 5. Synthesise the surviving differential into a clinician-style answer.
        if ranked:
            messages, tag = domain.synthesize_messages(ranked[:5])
            summary = self.client.complete(messages, tag=tag)[0]
        else:
            summary = ("No hypothesis survived deliberate search. Recommend "
                       "gathering more history and a focused exam.")

        elapsed = time.time() - t0
        return EngineResult(
            case=case,
            case_facts=case_facts,
            summary=summary,
            ranked=ranked,
            tree=tree,
            search=search_result,
            red_flags=getattr(domain, "red_flags", lambda: [])(),
            telemetry={
                "llm_calls": self.client.telemetry.calls,
                "retries": self.client.telemetry.retries,
                "failures": self.client.telemetry.failures,
                "thoughts_generated": tree.size,
                "reached_depth": search_result.reached_depth,
                "stopped_reason": search_result.stopped_reason,
            },
            config=self.cfg.to_dict(),
            elapsed_s=elapsed,
        )

    @staticmethod
    def _rank(tree: Tree) -> list[Node]:
        """Surviving (non-pruned) hypotheses, best score first, de-duplicated."""
        survivors = [
            n for n in tree.nodes
            if not n.is_root() and n.status != NodeStatus.PRUNED
        ]
        survivors.sort(key=lambda n: n.score, reverse=True)
        out: list[Node] = []
        seen: set[str] = set()
        for n in survivors:
            key = n.content.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(n)
        return out
