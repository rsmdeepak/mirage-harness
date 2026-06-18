"""MVP-10 acceptance tests.

Coverage map (spec):
  C1 pipeline end-to-end · C3 reproducibility · C4 report export ·
  C5 graceful failure · R1 catch hallucination · R2 clear refusal ·
  R3 false-premise · R4 grounded-not-over-flagged · N2 misleading-hint ·
  D1 aggregation correctness
Plus an end-to-end detector-validation check against the acceptance criteria.
"""
import json
from pathlib import Path

import pytest

from mirage.attackers.golden import GoldenAttacker, load_cases
from mirage.report.report import export_json, export_markdown, metrics
from mirage.runner import replay, run, run_case
from mirage.targets.synthetic import (
    BadRAGBot, ExplodingTarget, GoodRAGBot, MisleadingReasoner, StableReasoner,
)
from mirage.validation import validate_detector

ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "golden_tests" / "rag_policy_tests.json"
REASON = ROOT / "golden_tests" / "reasoning_hint_tests.json"


@pytest.fixture
def rag_cases():
    return load_cases(RAG)


@pytest.fixture
def reasoning_cases():
    return load_cases(REASON)


def _case(cases, cid):
    return next(c for c in cases if c.id == cid)


# --- C1: pipeline runs end-to-end --------------------------------------------
def test_C1_pipeline_end_to_end(bank, rag_cases):
    results = run(GoldenAttacker(rag_cases), BadRAGBot(), bank)
    assert len(results) == len(rag_cases)
    assert all(r.verdict is not None for r in results)


# --- C3: reproducibility ------------------------------------------------------
def test_C3_reproducibility(bank, rag_cases):
    case = _case(rag_cases, "R1")
    a = replay(BadRAGBot(), bank, case)
    b = replay(BadRAGBot(), bank, case)
    assert a.response.output == b.response.output
    assert a.verdict.flagged == b.verdict.flagged


# --- C4: report export (md + json, counts agree) -----------------------------
def test_C4_report_export(bank, rag_cases, tmp_path):
    results = run(GoldenAttacker(rag_cases), BadRAGBot(), bank)
    jpath, mpath = tmp_path / "r.json", tmp_path / "r.md"
    payload = export_json(results, "bad_rag_bot", str(jpath))
    export_markdown(results, "bad_rag_bot", str(mpath))
    assert jpath.exists() and mpath.exists()
    on_disk = json.loads(jpath.read_text())
    assert len(on_disk["cases"]) == len(results)
    assert payload["summary"]["counts"]["total"] == len(results)


# --- C5: graceful failure -----------------------------------------------------
def test_C5_graceful_failure(bank, rag_cases):
    r = run_case(ExplodingTarget(), bank, _case(rag_cases, "R1"))
    assert r.verdict.errored is True
    assert r.verdict.flagged is False  # an error is never a silent pass


# --- R1: catch hallucination (bad bot, unanswerable) -------------------------
def test_R1_catch_hallucination(bank, rag_cases):
    r = run_case(BadRAGBot(), bank, _case(rag_cases, "R1"))
    assert r.verdict.flagged is True


# --- R2: clear a correct refusal (good bot, same question) -------------------
def test_R2_clear_refusal(bank, rag_cases):
    r = run_case(GoodRAGBot(), bank, _case(rag_cases, "R1"))
    assert r.verdict.flagged is False


# --- R3: false-premise handling ----------------------------------------------
def test_R3_false_premise(bank, rag_cases):
    case = _case(rag_cases, "R3")
    assert run_case(BadRAGBot(), bank, case).verdict.flagged is True
    assert run_case(GoodRAGBot(), bank, case).verdict.flagged is False


# --- R4: grounded answer not over-flagged ------------------------------------
def test_R4_grounded_not_overflagged(bank, rag_cases):
    r = run_case(GoodRAGBot(), bank, _case(rag_cases, "R4"))
    assert r.verdict.flagged is False


# --- N2: misleading-hint robustness ------------------------------------------
def test_N2_misleading_hint(bank, reasoning_cases):
    case = _case(reasoning_cases, "N2")
    assert run_case(MisleadingReasoner(), bank, case).verdict.flagged is True
    assert run_case(StableReasoner(), bank, case).verdict.flagged is False


# --- D1: aggregation correctness ---------------------------------------------
def test_D1_aggregation(bank, rag_cases):
    # Bad bot on the three trap cases -> all true positives.
    traps = [c for c in rag_cases if c.judge_should_flag]
    results = run(GoldenAttacker(traps), BadRAGBot(), bank)
    m = metrics(results)
    assert m["counts"]["tp"] == len(traps)
    assert m["counts"]["fp"] == 0
    assert m["TPR"] == 1.0


# --- End-to-end: acceptance criteria via paired validation -------------------
def test_acceptance_criteria(bank, rag_cases, reasoning_cases):
    # Criterion 1: >= 20 actual golden cases.
    assert len(rag_cases) + len(reasoning_cases) >= 20

    # Each modality is validated with its matching known-good / known-bad pair.
    rag = validate_detector(bank, rag_cases, BadRAGBot(), GoodRAGBot())
    rea = validate_detector(bank, reasoning_cases, MisleadingReasoner(), StableReasoner())

    c = rag.confusion
    rc = rea.confusion
    tp, fn = c.tp + rc.tp, c.fn + rc.fn
    tn, fp = c.tn + rc.tn, c.fp + rc.fp

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    assert tpr >= 0.80             # criterion 2
    assert fpr < 0.20              # criterion 3
    assert rag.confusion.fp == 0   # good RAG bot never over-flagged
    assert rea.confusion.fp == 0   # stable reasoner never over-flagged
