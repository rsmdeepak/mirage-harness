"""Detector validation (spec sections F/G/H).

The crux: a *trap* case (``judge_should_flag = True``) should be flagged when a
KNOWN-BAD target answers it, and NOT flagged when a KNOWN-GOOD target answers
it. So we measure the detector with a paired run:

    * True Positives / False Negatives  <- bad target on the trap cases
    * True Negatives / False Positives  <- good target on ALL cases (never flag)

Combining the two gives Mirage's full confusion matrix and the F-section
metrics. This is also the calibration step: tune judge thresholds here until
known-bad reliably fails and known-good reliably passes, *then* test real models.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attackers.golden import GoldenAttacker
from .core.interfaces import AttackCase, Target
from .judges.bank import JudgeBank
from .report.report import (
    Confusion, confusion, judge_agreement, metrics_from_confusion,
)
from .runner import RunResult, run


@dataclass
class ValidationResult:
    bad_results: list[RunResult]
    good_results: list[RunResult]
    confusion: Confusion
    metrics: dict


def validate_detector(judges: JudgeBank, cases: list[AttackCase],
                      bad_target: Target, good_target: Target) -> ValidationResult:
    traps = [c for c in cases if c.judge_should_flag]

    # TP / FN: known-bad target on trap cases (expected to be flagged).
    bad = run(GoldenAttacker(traps), bad_target, judges)
    bad_c = confusion(bad, expected=lambda r: True)

    # TN / FP: known-good target on all cases (should never be flagged).
    good = run(GoldenAttacker(cases), good_target, judges)
    good_c = confusion(good, expected=lambda r: False)

    combined = Confusion(
        tp=bad_c.tp, fn=bad_c.fn,
        tn=good_c.tn, fp=good_c.fp,
        errored=bad_c.errored + good_c.errored,
    )
    attack_types = {c.attack_type for c in cases}
    m = metrics_from_confusion(
        combined, total=len(bad) + len(good), attack_types=attack_types,
        judge_agreement=judge_agreement(bad + good),  # diagnostic (fix #5)
    )
    return ValidationResult(bad_results=bad, good_results=good,
                            confusion=combined, metrics=m)
