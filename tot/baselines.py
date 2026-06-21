"""IO and CoT baselines, for answering the same case three ways:

  IO  - one-shot answer
  CoT - one linear chain of reasoning, then an answer
  ToT - the full tree search (ToTEngine)

Lets you see ToT explore and prune where IO/CoT just commit to one line.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ToTConfig
from .domain import ClinicalDiagnosisDomain
from .engine import EngineResult, ToTEngine
from .llm import LLMClient, build_client


@dataclass
class BaselineResult:
    mode: str          # "io" | "cot"
    answer: str
    llm_calls: int


def run_baseline(case: str, mode: str, cfg: ToTConfig,
                 client: LLMClient | None = None) -> BaselineResult:
    client = client or build_client(cfg.llm)
    client.telemetry.reset()
    domain = ClinicalDiagnosisDomain(case)
    messages, tag = domain.baseline_messages(mode)
    answer = client.complete(messages, tag=tag)[0]
    return BaselineResult(mode=mode, answer=answer.strip(),
                          llm_calls=client.telemetry.calls)


@dataclass
class Comparison:
    case: str
    io: BaselineResult
    cot: BaselineResult
    tot: EngineResult


def compare(case: str, cfg: ToTConfig,
            client: LLMClient | None = None) -> Comparison:
    """Run IO, CoT and ToT on the same case for a side-by-side demo."""
    client = client or build_client(cfg.llm)
    io = run_baseline(case, "io", cfg, client)
    cot = run_baseline(case, "cot", cfg, client)
    tot = ToTEngine(cfg, client=client).run(case)
    return Comparison(case=case, io=io, cot=cot, tot=tot)
