"""LLM client.

One code path for any OpenAI-compatible chat endpoint (Groq, Gemini, OpenAI,
Ollama), plus a MockLLM that runs the whole pipeline offline for tests/demos.
Uses plain `requests` so there's no provider SDK to keep in sync.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .config import LLMConfig

try:  # `requests` is only needed for real providers, not for mock/tests.
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


Message = dict[str, str]


@dataclass
class LLMTelemetry:
    """Counts API calls/retries/failures for one query."""

    calls: int = 0
    failures: int = 0
    retries: int = 0

    def reset(self) -> None:
        self.calls = self.failures = self.retries = 0


class LLMClient:
    """Thin wrapper over an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.telemetry = LLMTelemetry()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tag: str | None = None,          # used only by MockLLM to branch
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
    ) -> list[str]:
        """Return `n` completions. Loops instead of using the server-side `n`
        param, which not every free provider supports."""
        return [
            self._complete_one(messages, temperature, max_tokens)
            for _ in range(max(1, n))
        ]

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        tag: str | None = None,
        temperature: float | None = None,
        salvage_key: str | None = None,
    ) -> Any:
        """Complete and parse the first JSON object/array found in the reply.

        `salvage_key` names an array of flat objects (e.g. "thoughts") that can
        be recovered element-by-element if the model truncates its output.
        """
        raw = self.complete(
            messages, tag=tag, temperature=temperature, max_tokens=None, n=1
        )[0]
        return _extract_json(raw, salvage_key=salvage_key)

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _complete_one(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        if requests is None:  # pragma: no cover
            raise RuntimeError(
                "The 'requests' package is required for live providers. "
                "Install it, or use provider='mock'."
            )
        api_key = self.cfg.api_key
        if not api_key and self.cfg.provider != "ollama":
            raise RuntimeError(
                f"No API key found in env var ${self.cfg.api_key_env}. "
                f"Export it, or run with provider='mock'."
            )

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": list(messages),
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key or 'ollama'}",
                   "Content-Type": "application/json"}

        backoff = 1.0
        last_err: Exception | None = None
        for _ in range(self.cfg.max_retries):
            try:
                self.telemetry.calls += 1
                resp = requests.post(
                    url, headers=headers, json=payload,
                    timeout=self.cfg.request_timeout,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    # Rate-limited or transient server error -> back off & retry.
                    raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except _Retryable as e:
                last_err = e
                self.telemetry.retries += 1
                time.sleep(backoff)
                backoff *= 2  # exponential backoff
            except Exception as e:  # noqa: BLE001 - surface non-retryable errors
                self.telemetry.failures += 1
                raise
        self.telemetry.failures += 1
        raise RuntimeError(f"LLM request failed after retries: {last_err}")


class _Retryable(Exception):
    """Internal marker for errors worth retrying with backoff."""


# --------------------------------------------------------------------------- #
# Deterministic offline mock
# --------------------------------------------------------------------------- #
class MockLLM(LLMClient):
    """Offline stand-in. Branches on the `tag` the engine passes so every step
    (propose/value/vote/decompose/synthesize/cot/io) returns usable output with
    no network. Seeded for reproducibility."""

    # tiny symptom -> candidate diagnoses table, just for the mock
    _KB: dict[str, list[tuple[str, float]]] = {
        "chest pain": [
            ("Acute coronary syndrome", 0.85),
            ("Pulmonary embolism", 0.6),
            ("Pericarditis", 0.45),
            ("Gastro-oesophageal reflux", 0.4),
            ("Musculoskeletal chest pain", 0.35),
        ],
        "shortness of breath": [
            ("Pulmonary embolism", 0.7),
            ("Heart failure", 0.6),
            ("Pneumonia", 0.55),
            ("Asthma exacerbation", 0.45),
        ],
        "headache": [
            ("Migraine", 0.7),
            ("Tension headache", 0.5),
            ("Subarachnoid haemorrhage", 0.4),
            ("Meningitis", 0.35),
        ],
        "fever": [
            ("Community-acquired pneumonia", 0.6),
            ("Urinary tract infection", 0.5),
            ("Influenza", 0.45),
        ],
    }
    _DEFAULT = [
        ("Viral syndrome", 0.5),
        ("Anxiety-related presentation", 0.4),
        ("Undifferentiated illness", 0.3),
    ]

    def __init__(self, cfg: LLMConfig, seed: int = 7):
        super().__init__(cfg)
        self._rng = random.Random(seed)

    def complete(self, messages, *, tag=None, temperature=None,
                 max_tokens=None, n=1) -> list[str]:
        self.telemetry.calls += 1
        user = "\n".join(m["content"] for m in messages if m["role"] == "user").lower()
        handler = {
            "decompose": self._decompose,
            "propose": self._propose,
            "value": self._value,
            "vote": self._vote,
            "synthesize": self._synthesize,
            "cot": self._cot,
            "io": self._io,
        }.get(tag or "", self._io)
        return [handler(user) for _ in range(max(1, n))]

    # -- helpers --------------------------------------------------------- #
    def _candidates(self, text: str) -> list[tuple[str, float]]:
        for key, dxs in self._KB.items():
            if key in text:
                return dxs
        return self._DEFAULT

    @staticmethod
    def _vignette(text: str) -> str:
        """Pull the case out of a triple-quoted prompt; fall back to the tail."""
        m = re.search(r'"""(.*?)"""', text, re.DOTALL)
        return (m.group(1) if m else text).strip()

    def _decompose(self, text: str) -> str:
        case = self._vignette(text)
        return json.dumps({
            "demographics": "unspecified",
            "chief_complaint": (case[:80] or "unspecified"),
            "key_findings": [w for w in self._KB if w in case] or ["unspecified"],
            "red_flags": [],
        })

    def _propose(self, text: str) -> str:
        cands = self._candidates(text)
        # jitter the order a little so successive samples differ slightly
        picks = cands[:]
        self._rng.shuffle(picks)
        thoughts = [
            {"hypothesis": name,
             "rationale": f"Consistent with the presentation ({name.lower()})."}
            for name, _ in picks[:4]
        ]
        return json.dumps({"thoughts": thoughts})

    def _value(self, text: str) -> str:
        cands = dict(self._candidates(text))
        # The prompt embeds the hypothesis being scored; find it.
        scored = []
        for name, base in cands.items():
            if name.lower() in text:
                label = ("sure" if base >= 0.7 else
                         "maybe" if base >= 0.4 else "impossible")
                scored.append({"hypothesis": name, "label": label, "score": base})
        if not scored:
            scored = [{"hypothesis": "candidate", "label": "maybe", "score": 0.5}]
        return json.dumps({"evaluations": scored})

    def _vote(self, text: str) -> str:
        return json.dumps({"best_choice": 1,
                           "reason": "Most consistent with the key findings."})

    def _synthesize(self, text: str) -> str:
        cands = self._candidates(text)
        ranked = "\n".join(
            f"{i+1}. {name} (confidence {score:.0%})"
            for i, (name, score) in enumerate(cands[:3])
        )
        return (
            "Working differential (educational, not medical advice):\n"
            f"{ranked}\n\n"
            "Suggested discriminating steps: focused history, targeted exam, "
            "and the first-line investigation for the leading hypothesis.\n"
            "Red-flag safety net: seek urgent in-person assessment if symptoms "
            "are severe, rapidly worsening, or accompanied by warning signs."
        )

    def _cot(self, text: str) -> str:
        cands = self._candidates(text)
        steps = " ".join(
            f"Considering {name}." for name, _ in cands[:3]
        )
        top = cands[0][0]
        return f"Let's reason step by step. {steps} The most likely is {top}."

    def _io(self, text: str) -> str:
        return f"Most likely diagnosis: {self._candidates(text)[0][0]}."


def _extract_json(raw: str, salvage_key: str | None = None) -> Any:
    """Pull a JSON object/array out of model output.

    Handles ```json fences and surrounding prose. If the reply was cut off
    (hit the token limit mid-output), `salvage_key` recovers the complete
    `{...}` items that did come through.
    """
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Brace/bracket matching fallback.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    # Salvage: recover whatever complete flat objects survived a truncation.
    if salvage_key:
        objects = []
        for chunk in re.findall(r"\{[^{}]*\}", raw):
            try:
                objects.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
        if objects:
            return {salvage_key: objects}
    raise ValueError(f"Could not extract JSON from model output: {raw[:200]!r}")


def build_client(cfg: LLMConfig, *, seed: int = 7) -> LLMClient:
    """Factory: return a MockLLM for provider='mock', else a live client."""
    if cfg.provider == "mock":
        return MockLLM(cfg, seed=seed)
    return LLMClient(cfg)
