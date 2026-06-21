"""Score states so the search knows what to keep and what to prune.

value: rate each state on its own as sure/maybe/impossible -> [0,1].
vote:  compare all states in one prompt; the winner gets the highest share.
Either can be sampled several times and averaged (the `votes` argument).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import EvaluationStrategy
from .domain import Domain
from .llm import LLMClient
from .thought import Node


class StateEvaluator:
    def __init__(self, client: LLMClient, domain: Domain):
        self.client = client
        self.domain = domain

    def evaluate(
        self,
        nodes: list[Node],
        strategy: EvaluationStrategy,
        votes: int = 1,
    ) -> list[Node]:
        if not nodes:
            return nodes
        if strategy is EvaluationStrategy.VALUE:
            self._value(nodes, votes)
        else:
            self._vote(nodes, votes)
        return nodes

    # ------------------------------------------------------------------ #
    def _value(self, nodes: list[Node], votes: int) -> None:
        scores: dict[int, list[float]] = defaultdict(list)
        labels: dict[int, list[str]] = defaultdict(list)
        notes: dict[int, str] = {}
        for _ in range(max(1, votes)):
            messages, tag = self.domain.value_messages(nodes)
            try:
                payload = self.client.complete_json(
                    messages, tag=tag, salvage_key="evaluations"
                )
            except ValueError:
                continue  # skip this unparseable vote; others still count
            result = self.domain.parse_value(payload, nodes)
            for node_id, (score, label, note) in result.items():
                scores[node_id].append(score)
                labels[node_id].append(label)
                if note:
                    notes[node_id] = note
        for node in nodes:
            s = scores.get(node.id) or [0.4]
            node.score = sum(s) / len(s)
            node.label = Counter(labels.get(node.id, ["maybe"])).most_common(1)[0][0]
            node.eval_notes = notes.get(node.id, "")

    def _vote(self, nodes: list[Node], votes: int) -> None:
        tally: Counter[int] = Counter()
        for _ in range(max(1, votes)):
            messages, tag = self.domain.vote_messages(nodes)
            try:
                payload = self.client.complete_json(messages, tag=tag)
            except ValueError:
                continue  # skip this unparseable vote
            idx = self.domain.parse_vote(payload, len(nodes))
            tally[idx] += 1
        total = sum(tally.values()) or 1
        winner = tally.most_common(1)[0][0] if tally else 0
        for i, node in enumerate(nodes):
            node.score = tally.get(i, 0) / total
            node.label = "sure" if i == winner else "maybe"
            node.eval_notes = f"vote share {node.score:.0%}"
