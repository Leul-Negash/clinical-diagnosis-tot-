"""Search over the tree of thoughts: BFS (Algorithm 1) and DFS (Algorithm 2).

Both are bounded by the config - depth T, breadth b, branching k, prune
threshold v_th, a global node cap, and early stopping at solved_threshold - so
the search always terminates within a known budget.

Each step emits an event through the optional `on_event` callback, which the
UIs use to render the tree as it grows and prunes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .config import SearchStrategy, ToTConfig
from .evaluator import StateEvaluator
from .generator import ThoughtGenerator
from .thought import Node, NodeStatus, Tree

EventCallback = Optional[Callable[[dict], None]]


@dataclass
class SearchResult:
    tree: Tree
    best: Node
    strategy: SearchStrategy
    reached_depth: int
    stopped_reason: str


class Searcher:
    def __init__(
        self,
        tree: Tree,
        generator: ThoughtGenerator,
        evaluator: StateEvaluator,
        cfg: ToTConfig,
        on_event: EventCallback = None,
    ):
        self.tree = tree
        self.gen = generator
        self.eval = evaluator
        self.cfg = cfg
        self._emit = on_event or (lambda e: None)

    # ------------------------------------------------------------------ #
    def run(self) -> SearchResult:
        if self.cfg.search_strategy is SearchStrategy.BFS:
            return self._bfs()
        return self._dfs()

    def _remaining_budget(self) -> int:
        """Thoughts we may still generate before hitting the global cap."""
        return self.cfg.max_total_thoughts - self.tree.size

    def _budget_left(self) -> bool:
        return self._remaining_budget() > 0

    # ------------------------------------------------------------------ #
    # Algorithm 1: Breadth-first search (keep best b states per level).
    # ------------------------------------------------------------------ #
    def _bfs(self) -> SearchResult:
        frontier: list[Node] = [self.tree.root]
        reached, reason = 0, "max_depth reached"

        for depth in range(1, self.cfg.max_depth + 1):
            # --- generate: expand every frontier state by k thoughts --------
            candidates: list[Node] = []
            for state in frontier:
                budget = self._remaining_budget()
                if budget <= 0:
                    reason = "thought budget exhausted"
                    break
                children = self.gen.generate(
                    state, min(self.cfg.n_generate, budget),
                    self.cfg.generation_strategy,
                )
                self.tree.expand(state, children)
                candidates.extend(children)
                for c in children:
                    self._emit({"type": "generate", "node": c, "parent": state,
                                "depth": depth})
            if not candidates:
                reason = "no further hypotheses proposed"
                break

            # --- evaluate the whole candidate frontier ---------------------
            self.eval.evaluate(
                candidates, self.cfg.evaluation_strategy, self.cfg.n_evaluate_votes
            )
            for c in candidates:
                self._emit({"type": "evaluate", "node": c, "depth": depth})

            # --- select: keep the best b, prune the rest -------------------
            ranked = sorted(candidates, key=lambda n: n.score, reverse=True)
            keep = ranked[: self.cfg.breadth]
            for n in ranked[self.cfg.breadth:]:
                n.status = NodeStatus.PRUNED
                self._emit({"type": "prune", "node": n, "reason": "below breadth b",
                            "depth": depth})
            # Also prune anything the evaluator deemed effectively impossible.
            survivors = []
            for n in keep:
                if n.score < self.cfg.prune_threshold or n.label == "impossible":
                    n.status = NodeStatus.PRUNED
                    self._emit({"type": "prune", "node": n,
                                "reason": "score < threshold", "depth": depth})
                else:
                    survivors.append(n)

            reached = depth
            if not survivors:
                reason = "all candidates pruned"
                break

            # --- early stopping: a hypothesis is confident enough ----------
            best = max(survivors, key=lambda n: n.score)
            if best.score >= self.cfg.solved_threshold:
                best.status = NodeStatus.SOLVED
                self._emit({"type": "solved", "node": best, "depth": depth})
                reason = "confident hypothesis found (early stop)"
                frontier = survivors
                break

            frontier = survivors

        return SearchResult(self.tree, self.tree.best_leaf(),
                            SearchStrategy.BFS, reached, reason)

    # ------------------------------------------------------------------ #
    # Algorithm 2: Depth-first search with pruning and backtracking.
    # ------------------------------------------------------------------ #
    def _dfs(self) -> SearchResult:
        state = {"reached": 0, "reason": "max_depth reached", "solved": None}

        def visit(node: Node, depth: int) -> None:
            if state["solved"] is not None:
                return
            budget = self._remaining_budget()
            if depth > self.cfg.max_depth or budget <= 0:
                if budget <= 0:
                    state["reason"] = "thought budget exhausted"
                return

            children = self.gen.generate(
                node, min(self.cfg.n_generate, budget), self.cfg.generation_strategy
            )
            if not children:
                return
            self.tree.expand(node, children)
            for c in children:
                self._emit({"type": "generate", "node": c, "parent": node,
                            "depth": depth})

            self.eval.evaluate(
                children, self.cfg.evaluation_strategy, self.cfg.n_evaluate_votes
            )
            for c in children:
                self._emit({"type": "evaluate", "node": c, "depth": depth})

            # Prune excluded hypotheses up front (judged impossible or below
            # v_th), so they read as pruned even if early stopping returns
            # before the exploration loop reaches them.
            survivors: list[Node] = []
            for c in children:
                if c.score < self.cfg.prune_threshold or c.label == "impossible":
                    c.status = NodeStatus.PRUNED
                    self._emit({"type": "prune", "node": c,
                                "reason": "v(s) <= v_th", "depth": depth})
                else:
                    survivors.append(c)

            # Explore most-promising survivors first (paper: sorted candidates).
            for child in sorted(survivors, key=lambda n: n.score, reverse=True):
                if state["solved"] is not None:
                    break
                state["reached"] = max(state["reached"], depth)
                if child.score >= self.cfg.solved_threshold or depth == self.cfg.max_depth:
                    child.status = NodeStatus.SOLVED
                    state["solved"] = child
                    state["reason"] = "confident hypothesis found (early stop)"
                    self._emit({"type": "solved", "node": child, "depth": depth})
                    return
                visit(child, depth + 1)  # deepen
                self._emit({"type": "backtrack", "node": child, "depth": depth})

        visit(self.tree.root, 1)
        return SearchResult(self.tree, self.tree.best_leaf(),
                            SearchStrategy.DFS, state["reached"], state["reason"])
