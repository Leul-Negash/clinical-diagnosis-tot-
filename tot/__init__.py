"""Tree-of-Thoughts deep-research engine for clinical differential diagnosis.

Implements Yao et al. 2023 (arXiv:2305.10601) as a modular, domain-agnostic
engine, instantiated for differential-diagnosis research.
"""

from .config import (
    EvaluationStrategy,
    GenerationStrategy,
    LLMConfig,
    SearchStrategy,
    ToTConfig,
)
from .domain import ClinicalDiagnosisDomain, Domain
from .engine import EngineResult, ToTEngine
from .baselines import Comparison, compare, run_baseline
from .llm import LLMClient, MockLLM, build_client
from .thought import Node, NodeStatus, Tree

__all__ = [
    "ToTConfig", "LLMConfig",
    "GenerationStrategy", "EvaluationStrategy", "SearchStrategy",
    "ToTEngine", "EngineResult",
    "Domain", "ClinicalDiagnosisDomain",
    "compare", "run_baseline", "Comparison",
    "LLMClient", "MockLLM", "build_client",
    "Tree", "Node", "NodeStatus",
]

__version__ = "1.0.0"
