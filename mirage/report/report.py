"""Aggregation + export.

Quality metrics (TPR/TNR/FPR/FNR) compare each verdict's ``flagged`` against an
*expected* flag. By default the expectation is the golden label
``judge_should_flag`` (correct when running a known-bad / trap target). The
paired-validation routine in ``mirage.validation`` overrides the expectation for
a known-good target (which should never be flagged).

Judge Agreement is reported as a diagnostic only. Errored cases are excluded
from the confusion matrix and counted separately -- never silently passed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from ..runner import RunResult

Expected = Callable[[RunResult], bool]


def _default_expected(r: RunResult) -> bool:
    return r.case.judge_should_flag


@dataclass
class Confusion:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    errored: int = 0

    @property
    def scored(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


def confusion(results: Iterable[RunResult], expected: Expected = _default_expected) -> Confusion:
    c = Confusion()
    for r in results:
        if r.verdict.errored:
            c.errored += 1
            continue
        should, did = expected(r), r.verdict.flagged
        if should and did:
            c.tp += 1
        elif should and not did:
            c.fn += 1
        elif not should and did:
            c.fp += 1
        else:
            c.tn += 1
    return c


def _rate(num: int, den: int):
    return round(num / den, 3) if den else None


def metrics_from_confusion(c: Confusion, total: int, attack_types=None,
                           judge_agreement=None) -> dict:
    return {
        "TPR": _rate(c.tp, c.tp + c.fn),
        "TNR": _rate(c.tn, c.tn + c.fp),
        "FPR": _rate(c.fp, c.fp + c.tn),
        "FNR": _rate(c.fn, c.fn + c.tp),
        "judge_agreement": judge_agreement,  # diagnostic only
        "counts": {"tp": c.tp, "tn": c.tn, "fp": c.fp, "fn": c.fn,
                   "errored": c.errored, "total": total},
        "coverage_by_attack_type": sorted(attack_types or []),
    }


def judge_agreement(results: Iterable[RunResult]):
    """Mean of the per-case agreement diagnostic, or None if <2 voting judges."""
    vals = [r.verdict.signals["_agreement"] for r in results
            if "_agreement" in r.verdict.signals]
    return round(sum(vals) / len(vals), 3) if vals else None


def metrics(results: list[RunResult], expected: Expected = _default_expected) -> dict:
    c = confusion(results, expected)
    return metrics_from_confusion(
        c, total=len(results),
        attack_types={r.case.attack_type for r in results},
        judge_agreement=judge_agreement(results),
    )


def breakdown(results: list[RunResult], key, expected: Expected = _default_expected) -> dict:
    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    return {k: metrics(v, expected) for k, v in sorted(groups.items())}


def _outcome(r: RunResult, expected: Expected) -> str:
    if r.verdict.errored:
        return "errored"
    should, did = expected(r), r.verdict.flagged
    if should and did:
        return "TP"
    if should and not did:
        return "FN"
    if not should and did:
        return "FP"
    return "TN"


def case_status(case_dict: dict) -> str:
    """Human-facing status that can never render an error as 'clear' (fix #6)."""
    if case_dict["errored"]:
        return "errored"
    return "flagged" if case_dict["flagged"] else "clear"


def _judge_errors(verdict) -> list[str]:
    """Names of judges whose signal recorded an error (review fix #8)."""
    return [name for name, sig in verdict.signals.items()
            if isinstance(sig, dict) and "error" in sig]


def _suggested_fix(r) -> str | None:
    """A structured remediation suggestion for a flagged/errored case."""
    v = r.verdict
    if v.errored:
        return f"Investigate target/judge error: {v.rationale}"
    if not v.flagged:
        return None
    sig = v.signals
    if "citation" in sig and sig["citation"].get("flag"):
        return ("Ensure the cited source actually supports the claim; verify the "
                "answer against the cited chunk before returning it.")
    if "groundedness" in sig:  # RAG
        return ("Constrain the model to answer only from retrieved context; add a "
                "refusal instruction and raise the retrieval similarity threshold.")
    if "reasoning" in sig:
        rs = sig["reasoning"]
        if rs.get("caved_to_hint"):
            return ("Instruct the model to ignore unverified user hints and to "
                    "re-derive the answer independently.")
        if rs.get("explanation_consistent") is False:
            return ("Require the final answer to be derived from, and match, the "
                    "stated reasoning.")
        if rs.get("overthinking"):
            return ("Encourage concise reasoning; cap reasoning length on simple "
                    "questions to avoid drift.")
        return "Add an explicit self-check that re-verifies the final answer."
    return "Review the flagged output against the supporting evidence."


def to_dict(results: list[RunResult], model_name: str,
            expected: Expected = _default_expected, meta: dict | None = None) -> dict:
    cases = [
        {
            "id": r.case.id,
            "modality": r.case.modality,
            "attack_type": r.case.attack_type,
            "prompt": r.case.prompt,
            "context": r.case.context,
            "output": r.response.output,
            "expected_flag": expected(r),
            "flagged": r.verdict.flagged,
            "errored": r.verdict.errored,
            "outcome": _outcome(r, expected),
            "rationale": r.verdict.rationale,
            "seed": r.case.seed,
            "signals": r.verdict.signals,
            "judge_errors": _judge_errors(r.verdict),   # secondary errors (fix #8)
            "suggested_fix": _suggested_fix(r),          # remediation per failure
            "target_raw": r.response.raw,                # reproducibility (fix #2)
            "target_error": r.response.error,
        }
        for r in results
    ]
    for c in cases:
        c["status"] = case_status(c)
    # Secondary-judge errors on cases whose verdict is otherwise not errored.
    secondary_errors = sum(1 for c in cases if c["judge_errors"] and not c["errored"])
    return {
        "model": model_name,
        "meta": meta or {},                      # run/env/config metadata (fix #2)
        "summary": metrics(results, expected),
        "secondary_judge_errors": secondary_errors,
        "by_modality": breakdown(results, lambda r: r.case.modality, expected),
        "by_attack_type": breakdown(results, lambda r: r.case.attack_type, expected),
        "cases": cases,
    }


def session_to_dict(session) -> dict:
    """Build a report dict from a RunSession (meta always present)."""
    return to_dict(session.results, session.model_name,
                   session.expected(), session.meta())


def _md_escape(value) -> str:
    """Escape a value for a Markdown table cell (fix #8)."""
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def export_json(results: list[RunResult], model_name: str, path: str,
                expected: Expected = _default_expected, meta: dict | None = None) -> dict:
    # Low-level export: ``meta`` is optional here for flexibility. Prefer
    # ``export_session_json`` so reproducibility metadata is never omitted.
    payload = to_dict(results, model_name, expected, meta)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def render_markdown(results: list[RunResult], model_name: str,
                    expected: Expected = _default_expected, meta: dict | None = None) -> str:
    d = to_dict(results, model_name, expected, meta)
    s = d["summary"]
    cnt = s["counts"]
    lines = [
        f"# Mirage Report — `{_md_escape(model_name)}`",
        "",
        "## Detector validation metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| True Positive Rate | {s['TPR']} |",
        f"| True Negative Rate | {s['TNR']} |",
        f"| False Positive Rate | {s['FPR']} |",
        f"| False Negative Rate | {s['FNR']} |",
        f"| Judge Agreement (diagnostic) | {s['judge_agreement']} |",
        f"| Cases (scored / errored / total) | "
        f"{cnt['tp'] + cnt['tn'] + cnt['fp'] + cnt['fn']} / {cnt['errored']} / {cnt['total']} |",
        f"| Secondary judge errors | {d['secondary_judge_errors']} |",
        f"| Coverage by attack type | {_md_escape(', '.join(s['coverage_by_attack_type']))} |",
    ]
    if d["secondary_judge_errors"]:
        lines.append("")
        lines.append(f"> ⚠️ {d['secondary_judge_errors']} case(s) had a "
                     "secondary judge error (verdict still decided by the primary judge).")

    env = (meta or {}).get("environment", {})
    if meta:
        lines += [
            "",
            "## Run metadata (reproducibility)",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Timestamp (UTC) | {_md_escape(env.get('timestamp_utc'))} |",
            f"| Git commit | {_md_escape(env.get('git_commit'))} |",
            f"| Python | {_md_escape(env.get('python'))} |",
            f"| Platform | {_md_escape(env.get('platform'))} |",
            f"| Attacker | {_md_escape(json.dumps(meta.get('attacker')))} |",
            f"| Target | {_md_escape(json.dumps(meta.get('target')))} |",
            f"| Judges | {_md_escape(json.dumps(meta.get('judges')))} |",
            f"| Expected mode | {_md_escape(meta.get('expected_mode'))} |",
            f"| Dependencies | {_md_escape(json.dumps(env.get('dependencies')))} |",
        ]

    lines += [
        "",
        "## Flagged & errored cases",
        "",
        "| ID | Modality | Attack | Status | Outcome | Rationale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in d["cases"]:
        if case["flagged"] or case["errored"]:
            lines.append(
                f"| {_md_escape(case['id'])} | {_md_escape(case['modality'])} "
                f"| {_md_escape(case['attack_type'])} | {_md_escape(case['status'])} "
                f"| {_md_escape(case['outcome'])} | {_md_escape(case['rationale'])} |"
            )
    return "\n".join(lines) + "\n"


def export_markdown(results: list[RunResult], model_name: str, path: str,
                    expected: Expected = _default_expected, meta: dict | None = None) -> str:
    text = render_markdown(results, model_name, expected, meta)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# --- Session-based exports (meta always included; review fix #2) --------------

def export_session_json(session, path: str) -> dict:
    return export_json(session.results, session.model_name, path,
                       session.expected(), session.meta())


def render_session_markdown(session) -> str:
    return render_markdown(session.results, session.model_name,
                           session.expected(), session.meta())


def export_session_markdown(session, path: str) -> str:
    return export_markdown(session.results, session.model_name, path,
                           session.expected(), session.meta())


# --- Paired-validation reports (known-good vs known-bad) ----------------------

def validation_to_dict(v, modality: str, bad_name: str, good_name: str,
                       meta: dict | None = None) -> dict:
    """Report dict for a paired ValidationResult (TP/FN from bad, TN/FP from good)."""
    bad = to_dict(v.bad_results, bad_name, expected=lambda r: True, meta=meta)
    good = to_dict(v.good_results, good_name, expected=lambda r: False, meta=meta)
    return {
        "mode": "paired_validation",
        "modality": modality,
        "meta": meta or {},
        "metrics": v.metrics,
        "known_bad": {"model": bad_name, "cases": bad["cases"]},
        "known_good": {"model": good_name, "cases": good["cases"]},
    }


def render_validation_markdown(payload: dict) -> str:
    m = payload["metrics"]
    cnt = m["counts"]
    lines = [
        f"# Mirage Paired Validation — `{_md_escape(payload['modality'])}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| True Positive Rate | {m['TPR']} |",
        f"| True Negative Rate | {m['TNR']} |",
        f"| False Positive Rate | {m['FPR']} |",
        f"| False Negative Rate | {m['FNR']} |",
        f"| Judge Agreement (diagnostic) | {m['judge_agreement']} |",
        f"| Counts (tp/tn/fp/fn/errored) | "
        f"{cnt['tp']}/{cnt['tn']}/{cnt['fp']}/{cnt['fn']}/{cnt['errored']} |",
    ]
    for title, key in (("Known-bad target", "known_bad"), ("Known-good target", "known_good")):
        lines += ["", f"## {title} — `{_md_escape(payload[key]['model'])}`", "",
                  "| ID | Attack | Status | Outcome | Rationale |",
                  "| --- | --- | --- | --- | --- |"]
        for c in payload[key]["cases"]:
            lines.append(
                f"| {_md_escape(c['id'])} | {_md_escape(c['attack_type'])} "
                f"| {_md_escape(c['status'])} | {_md_escape(c['outcome'])} "
                f"| {_md_escape(c['rationale'])} |")
    return "\n".join(lines) + "\n"
