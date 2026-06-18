"""Modality-agnostic core contracts for Mirage.

The whole harness is built around four pluggable roles:

    Attacker  -> produces AttackCase objects (adversarial inputs)
    Target    -> responds to a case (the system under test, a black box)
    Judge     -> evaluates a response and emits signal(s)
    Report    -> aggregates verdicts into metrics + exports

Adding a new attack, target, or judge means implementing one of the ABCs
below; the runner and report layer never change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttackCase:
    """A single adversarial test case (one row of a golden set)."""

    id: str
    modality: str                       # "rag" | "reasoning" | "image"
    attack_type: str                    # e.g. "unanswerable", "false_premise", "misleading_hint"
    prompt: str
    context: Optional[str] = None       # supporting text, for RAG cases
    correct_answer: Optional[str] = None  # ground truth, for reasoning cases
    hint: Optional[str] = None          # injected (usually misleading) hint, for reasoning cases
    expected_behavior: str = ""         # human-readable: what a good target should do
    judge_should_flag: bool = False     # GROUND TRUTH for a KNOWN-BAD target on this case
    seed: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class TargetResponse:
    """What a Target produced for one case."""

    case_id: str
    output: str
    raw: dict = field(default_factory=dict)
    errored: bool = False
    error: Optional[str] = None


@dataclass
class Verdict:
    """The judge bank's decision for one case."""

    case_id: str
    flagged: bool                       # did Mirage flag a failure?
    signals: dict = field(default_factory=dict)  # per-judge raw outputs (diagnostic)
    rationale: str = ""
    errored: bool = False               # target/judge errored -> NOT a pass


class Target(ABC):
    """The system under test. Treated as a black box."""

    name: str = "target"

    @abstractmethod
    def respond(self, case: AttackCase) -> TargetResponse:  # pragma: no cover - interface
        ...

    def config(self) -> dict:
        """Serializable description of the target, captured in reports."""
        return {"name": self.name}


class Attacker(ABC):
    """Produces the adversarial cases to run."""

    @abstractmethod
    def generate(self) -> list[AttackCase]:  # pragma: no cover - interface
        ...

    def config(self) -> dict:
        """Serializable description of the attacker (type, sources), for reports."""
        return {"type": type(self).__name__}


class Judge(ABC):
    """Scores a single (case, response) pair.

    Must return a dict that includes at least a boolean ``flag`` key. Extra
    keys (scores, sub-signals) are stored for diagnostics and the dashboard.
    """

    name: str = "judge"

    @abstractmethod
    def evaluate(self, case: AttackCase, resp: TargetResponse) -> dict:  # pragma: no cover
        ...

    def config(self) -> dict:
        """Serializable description of the judge (type, thresholds), for reports."""
        return {"name": self.name}
