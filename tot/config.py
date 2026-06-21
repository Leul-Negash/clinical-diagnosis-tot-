"""Configuration for the ToT engine.

The config maps onto the four ToT design choices from the paper:
  1. thought decomposition  -> domain.py
  2. thought generator       -> generation_strategy, n_generate (k)
  3. state evaluator         -> evaluation_strategy, n_evaluate_votes
  4. search                  -> search_strategy, max_depth (T), breadth (b),
                                prune_threshold (v_th)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class GenerationStrategy(str, Enum):
    # sample: k independent draws (better for rich, open-ended thoughts)
    SAMPLE = "sample"
    # propose: one prompt suggests several next thoughts (better when the
    # thought space is constrained, e.g. diagnoses) - the default here
    PROPOSE = "propose"


class EvaluationStrategy(str, Enum):
    # value: score each state on its own (sure/maybe/impossible)
    VALUE = "value"
    # vote: compare states in one prompt and pick the best
    VOTE = "vote"


class SearchStrategy(str, Enum):
    BFS = "bfs"   # keep the best b states per level (Algorithm 1)
    DFS = "dfs"   # depth-first with pruning + backtracking (Algorithm 2)


@dataclass
class LLMConfig:
    """LLM settings. Defaults to Groq's free tier; any OpenAI-compatible
    endpoint works by changing provider/base_url/model. provider='mock' runs
    offline with no key."""

    provider: str = "groq"          # groq | gemini | openai | ollama | mock
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    temperature: float = 0.7        # 0.7 gives some diversity between samples
    max_tokens: int = 2048          # enough room that JSON replies don't get cut off
    request_timeout: float = 60.0
    max_retries: int = 4

    # provider -> endpoint/model defaults
    PRESETS: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "groq": {
                "model": "llama-3.3-70b-versatile",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY",
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "api_key_env": "GEMINI_API_KEY",
            },
            "openai": {
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
            "ollama": {
                "model": "llama3.1",
                "base_url": "http://localhost:11434/v1",
                "api_key_env": "OLLAMA_API_KEY",  # any value; Ollama ignores it
            },
            "mock": {
                "model": "mock-model",
                "base_url": "",
                "api_key_env": "MOCK_API_KEY",
            },
        },
        repr=False,
    )

    def apply_preset(self, provider: str) -> "LLMConfig":
        """Return self updated to a known provider preset."""
        preset = self.PRESETS.get(provider)
        if preset is None:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from {sorted(self.PRESETS)}."
            )
        self.provider = provider
        self.model = preset["model"]
        self.base_url = preset["base_url"]
        self.api_key_env = preset["api_key_env"]
        return self

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


@dataclass
class ToTConfig:
    # search bounds
    max_depth: int = 3              # T  - how deep the tree can go
    breadth: int = 3               # b  - states kept per level (BFS)
    n_generate: int = 4            # k  - hypotheses generated per state
    prune_threshold: float = 0.3   # v_th - drop states scoring below this
    n_evaluate_votes: int = 1      # evaluator samples to average over
    max_total_thoughts: int = 60   # hard cap on total nodes per query

    # strategies
    generation_strategy: GenerationStrategy = GenerationStrategy.PROPOSE
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.VALUE
    search_strategy: SearchStrategy = SearchStrategy.BFS

    # stop deepening once a hypothesis is this confident
    solved_threshold: float = 0.9

    llm: LLMConfig = field(default_factory=LLMConfig)

    def validate(self) -> "ToTConfig":
        """Reject bad bounds early so the search can't run away or hang."""
        if self.max_depth < 1:
            raise ValueError("max_depth (T) must be >= 1")
        if self.breadth < 1:
            raise ValueError("breadth (b) must be >= 1")
        if self.n_generate < 1:
            raise ValueError("n_generate (k) must be >= 1")
        if not 0.0 <= self.prune_threshold <= 1.0:
            raise ValueError("prune_threshold must be in [0, 1]")
        if not 0.0 <= self.solved_threshold <= 1.0:
            raise ValueError("solved_threshold must be in [0, 1]")
        if self.n_evaluate_votes < 1:
            raise ValueError("n_evaluate_votes must be >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("llm", None)
        d["llm"] = {k: v for k, v in asdict(self.llm).items() if k != "PRESETS"}
        return d
