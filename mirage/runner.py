"""The Attacker -> Target -> Judge pipeline.

A target that raises, or returns a malformed/mismatched response, is recorded as
``errored`` (NOT a pass). A judge bank crash is likewise contained. ``RunSession``
bundles the results with auto-built reproducibility metadata so that exports
always carry full config (review fix #2) -- callers don't hand-assemble it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core.interfaces import (
    AttackCase, Attacker, Judge, Target, TargetResponse, Verdict,
)
from .judges.bank import JudgeBank
from .meta import environment_metadata


@dataclass
class RunResult:
    case: AttackCase
    response: TargetResponse
    verdict: Verdict


def _validate_response(case: AttackCase, resp) -> str | None:
    """Return an error string if the target response is malformed (fix #4)."""
    if not isinstance(resp, TargetResponse):
        return f"target returned {type(resp).__name__}, expected TargetResponse"
    if resp.case_id != case.id:
        return f"response case_id '{resp.case_id}' != case id '{case.id}'"
    if not resp.errored and not isinstance(resp.output, str):
        return f"response output must be str, got {type(resp.output).__name__}"
    if not isinstance(resp.raw, dict):
        return f"response raw must be dict, got {type(resp.raw).__name__}"
    return None


def run_case(target: Target, judges: JudgeBank, case: AttackCase) -> RunResult:
    try:
        resp = target.respond(case)
    except Exception as exc:  # graceful failure (test C5)
        resp = TargetResponse(case_id=case.id, output="", errored=True, error=str(exc))

    malformed = _validate_response(case, resp)
    if malformed is not None:
        resp = TargetResponse(case_id=case.id, output="", errored=True, error=malformed)

    if resp.errored:
        verdict = Verdict(case_id=case.id, flagged=False, errored=True,
                          rationale=f"target errored: {resp.error}")
    else:
        try:
            verdict = judges.evaluate(case, resp)
        except Exception as exc:  # backstop: judge bank itself crashed
            verdict = Verdict(case_id=case.id, flagged=False, errored=True,
                              rationale=f"judge bank errored: {exc}")
    return RunResult(case=case, response=resp, verdict=verdict)


def run(attacker: Attacker, target: Target, judges: JudgeBank) -> list[RunResult]:
    return [run_case(target, judges, case) for case in attacker.generate()]


def replay(target: Target, judges: JudgeBank, case: AttackCase) -> RunResult:
    """Re-run a single saved case. Used to verify the Reproducibility metric."""
    return run_case(target, judges, case)


@dataclass
class RunSession:
    """Results + auto-built reproducibility metadata for one run.

    ``expected_mode`` controls how the report scores cases: ``"label"`` uses the
    golden ``judge_should_flag`` (for known-bad / trap targets); ``"known_good"``
    expects nothing to be flagged (for known-good control targets).
    """

    results: list[RunResult]
    attacker_config: dict
    target_config: dict
    judges_config: dict
    expected_mode: str = "label"
    environment: dict = field(default_factory=environment_metadata)

    VALID_MODES = ("label", "known_good")

    def __post_init__(self):
        if self.expected_mode not in self.VALID_MODES:
            raise ValueError(
                f"expected_mode must be one of {self.VALID_MODES}, "
                f"got {self.expected_mode!r}")

    @classmethod
    def execute(cls, attacker: Attacker, target: Target, judges: JudgeBank,
                expected_mode: str = "label") -> "RunSession":
        return cls(
            results=run(attacker, target, judges),
            attacker_config=attacker.config(),
            target_config=target.config(),
            judges_config=judges.config(),
            expected_mode=expected_mode,
        )

    @property
    def model_name(self) -> str:
        return self.target_config.get("name", "target")

    def meta(self) -> dict:
        return {
            "attacker": self.attacker_config,
            "target": self.target_config,
            "judges": self.judges_config,
            "expected_mode": self.expected_mode,
            "environment": self.environment,
        }

    def expected(self):
        if self.expected_mode == "known_good":
            return lambda r: False
        return lambda r: r.case.judge_should_flag  # "label" (validated in __post_init__)
