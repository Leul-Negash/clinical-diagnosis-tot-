"""Generate up to k next hypotheses from a state.

propose: one prompt returns several at once (avoids duplicates - default).
sample:  k independent draws (more diverse for open-ended thoughts).
"""

from __future__ import annotations

from .config import GenerationStrategy
from .domain import Domain
from .llm import LLMClient, _extract_json
from .thought import Node


class ThoughtGenerator:
    def __init__(self, client: LLMClient, domain: Domain):
        self.client = client
        self.domain = domain

    def generate(
        self,
        parent: Node,
        k: int,
        strategy: GenerationStrategy,
    ) -> list[Node]:
        trail = parent.thought_trail()
        if strategy is GenerationStrategy.PROPOSE:
            pairs = self._propose(trail, k)
        else:
            pairs = self._sample(trail, k)

        # don't re-propose a hypothesis already on this branch
        on_path = {t.lower() for t in trail}
        nodes: list[Node] = []
        for name, rationale in pairs:
            if name.lower() in on_path:
                continue
            nodes.append(
                Node(content=name, depth=parent.depth + 1, rationale=rationale)
            )
            if len(nodes) >= k:
                break
        return nodes

    # ------------------------------------------------------------------ #
    def _propose(self, trail: list[str], k: int) -> list[tuple[str, str]]:
        messages, tag = self.domain.generate_messages(trail, k)
        try:
            payload = self.client.complete_json(
                messages, tag=tag, salvage_key="thoughts"
            )
        except ValueError:
            # bad output from the model - return nothing and let the search
            # carry on with the rest of the frontier
            return []
        return self.domain.parse_generation(payload)

    def _sample(self, trail: list[str], k: int) -> list[tuple[str, str]]:
        messages, tag = self.domain.generate_messages(trail, k)
        raws = self.client.complete(messages, tag=tag, n=k)  # k independent draws
        seen: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for raw in raws:
            try:
                payload = _extract_json(raw, salvage_key="thoughts")
            except ValueError:
                continue
            for name, rationale in self.domain.parse_generation(payload):
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                pairs.append((name, rationale))
                break  # one thought per i.i.d. sample
        return pairs
