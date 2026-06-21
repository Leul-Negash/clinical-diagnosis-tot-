"""Tree data structures.

A state in ToT is the input plus the thoughts taken so far. Here each Node is a
state, and the root-to-node path is that thought sequence. Tree holds every node
the search touches.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


class NodeStatus(str, Enum):
    ACTIVE = "active"        # frontier node, eligible for expansion
    EXPANDED = "expanded"    # children generated
    PRUNED = "pruned"        # evaluator/backtracking removed it from search
    SOLVED = "solved"        # crossed the solved_threshold; a terminal answer


@dataclass
class Node:
    """A single state in the tree of thoughts."""

    content: str                         # the thought itself (a hypothesis)
    depth: int
    id: int = field(default_factory=lambda: next(_ID_COUNTER))
    parent: Optional["Node"] = field(default=None, repr=False)
    children: list["Node"] = field(default_factory=list, repr=False)

    # set by the evaluator
    score: float = 0.0                   # value in [0, 1]
    label: Optional[str] = None          # sure | maybe | impossible
    status: NodeStatus = NodeStatus.ACTIVE

    rationale: str = ""                   # why this hypothesis was proposed
    eval_notes: str = ""                  # why it got that score

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def path(self) -> list["Node"]:
        """Root-to-self chain, i.e. the thought sequence defining this state."""
        chain: list[Node] = []
        node: Optional[Node] = self
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def thought_trail(self) -> list[str]:
        """The thoughts from the root's children down to this node (excludes the
        synthetic root which holds the case itself)."""
        return [n.content for n in self.path() if n.parent is not None]

    def is_root(self) -> bool:
        return self.parent is None


_ID_COUNTER: Iterator[int] = itertools.count(0)


class Tree:
    """Owns the root state and the full set of explored nodes."""

    def __init__(self, root_content: str):
        self.root = Node(content=root_content, depth=0)
        self.nodes: list[Node] = [self.root]

    def register(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def expand(self, parent: Node, children: list[Node]) -> list[Node]:
        for c in children:
            parent.add_child(c)
            self.register(c)
        parent.status = NodeStatus.EXPANDED
        return children

    # -- introspection used by search bounds and the UIs -------------------- #
    @property
    def size(self) -> int:
        """Total thoughts generated (excludes the synthetic root)."""
        return len(self.nodes) - 1

    def active_leaves(self) -> list[Node]:
        return [
            n for n in self.nodes
            if n.status == NodeStatus.ACTIVE and not n.children
        ]

    def solved(self) -> list[Node]:
        return [n for n in self.nodes if n.status == NodeStatus.SOLVED]

    def best_leaf(self) -> Node:
        """Highest-scoring non-pruned node (the answer the search converged on)."""
        candidates = [n for n in self.nodes
                      if n.status != NodeStatus.PRUNED and not n.is_root()]
        if not candidates:
            return self.root
        return max(candidates, key=lambda n: n.score)

    def to_render_dict(self) -> dict:
        """Serialise the whole tree for the CLI / Streamlit renderers."""
        def encode(node: Node) -> dict:
            return {
                "id": node.id,
                "content": node.content,
                "depth": node.depth,
                "score": round(node.score, 3),
                "label": node.label,
                "status": node.status.value,
                "rationale": node.rationale,
                "eval_notes": node.eval_notes,
                "children": [encode(c) for c in node.children],
            }
        return encode(self.root)
