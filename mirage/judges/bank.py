"""Judge bank: routes a case to the judges for its modality and combines them.

The first judge registered for a modality is the *primary* and drives the
flag decision. Additional judges contribute diagnostic signals and feed the
Judge Agreement metric (a diagnostic, not a quality score -- judges can agree
and still be wrong).

A judge that raises is contained: its signal records the error, and if the
*primary* judge raises the whole verdict is marked ``errored`` (never a silent
pass). Secondary judge failures degrade gracefully.
"""
from __future__ import annotations

from ..core.interfaces import AttackCase, Judge, TargetResponse, Verdict


class JudgeBank:
    def __init__(self, judges_by_modality: dict[str, list[Judge]]):
        self.judges_by_modality = judges_by_modality

    def config(self) -> dict:
        return {modality: [j.config() for j in judges]
                for modality, judges in self.judges_by_modality.items()}

    def evaluate(self, case: AttackCase, resp: TargetResponse) -> Verdict:
        judges = self.judges_by_modality.get(case.modality, [])
        if not judges:
            return Verdict(case_id=case.id, flagged=False, errored=True,
                           rationale=f"no judge registered for modality '{case.modality}'")

        signals: dict = {}
        votes: list[bool] = []
        primary_flag = False
        primary_errored = False
        rationale = ""

        for i, judge in enumerate(judges):
            try:
                sig = judge.evaluate(case, resp)
                errored = False
            except Exception as exc:  # contain judge crashes (review fix #5)
                sig = {"flag": False, "error": f"{type(exc).__name__}: {exc}"}
                errored = True
            signals[judge.name] = sig

            if not errored and "flag" in sig:
                votes.append(bool(sig["flag"]))
            if i == 0:
                if errored:
                    primary_errored = True
                    rationale = f"primary judge '{judge.name}' errored: {sig['error']}"
                else:
                    primary_flag = bool(sig.get("flag", False))
                    rationale = sig.get("reason", "")

        if len(votes) > 1:
            signals["_agreement"] = (len(set(votes)) == 1)

        return Verdict(case_id=case.id, flagged=primary_flag, signals=signals,
                       rationale=rationale, errored=primary_errored)
