"""Tests for the capabilities added to make the spec true:
suggested_fix, overthinking, self-consistency, citation faithfulness,
multi-hop/outdated coverage, and the optional NLI judge's graceful degradation.
"""
import importlib.util

import pytest

from mirage import default_judge_bank
from mirage.attackers.golden import load_cases
from mirage.core.interfaces import AttackCase, TargetResponse
from mirage.judges.citation import CitationFaithfulnessJudge
from mirage.judges.consistency import SelfConsistencyJudge
from mirage.judges.nli import NLIGroundednessJudge
from mirage.judges.reasoning import ReasoningJudge
from mirage.report.report import to_dict
from mirage.runner import run_case
from mirage.targets.synthetic import (
    BadRAGBot, FaithfulCiteBot, FlakyReasoner, GoodRAGBot, StableReasoner,
    UnfaithfulCiteBot,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "golden_tests" / "rag_policy_tests.json"
REASON = ROOT / "golden_tests" / "reasoning_hint_tests.json"


def _rag(cid):
    return next(c for c in load_cases(RAG) if c.id == cid)


def _reason(cid):
    return next(c for c in load_cases(REASON) if c.id == cid)


# --- suggested_fix ------------------------------------------------------------
def test_suggested_fix_present_for_failure_absent_for_pass():
    bank = default_judge_bank()
    bad = run_case(BadRAGBot(), bank, _rag("R1"))
    good = run_case(GoodRAGBot(), bank, _rag("R4"))
    bad_case = to_dict([bad], "b", expected=lambda r: True)["cases"][0]
    good_case = to_dict([good], "g", expected=lambda r: False)["cases"][0]
    assert bad_case["suggested_fix"] and "context" in bad_case["suggested_fix"].lower()
    assert good_case["suggested_fix"] is None


# --- overthinking detection ---------------------------------------------------
def test_overthinking_flagged():
    case = AttackCase(id="O", modality="reasoning", attack_type="x",
                      prompt="What is 2+2?", correct_answer="4")
    essay = " ".join(["reasoning"] * 80)
    resp = TargetResponse("O", f"{essay}\nAnswer: 4")
    sig = ReasoningJudge(overthinking_tokens=50).evaluate(case, resp)
    assert sig["is_correct"] is True       # answer is right...
    assert sig["overthinking"] is True     # ...but flagged for overthinking
    assert sig["flag"] is True


def test_concise_correct_answer_not_overthinking():
    case = AttackCase(id="O2", modality="reasoning", attack_type="x",
                      prompt="What is 2+2?", correct_answer="4")
    sig = ReasoningJudge().evaluate(case, TargetResponse("O2", "It is four.\nAnswer: 4"))
    assert sig["overthinking"] is False and sig["flag"] is False


# --- self-consistency ---------------------------------------------------------
def test_self_consistency_flags_flaky_target():
    case = _reason("N1")
    flaky = FlakyReasoner()
    first = flaky.respond(case)
    sig = SelfConsistencyJudge(flaky, n=5).evaluate(case, first)
    assert sig["flag"] is True
    assert sig["consistency"] < 1.0


def test_self_consistency_passes_stable_target():
    case = _reason("N1")
    stable = StableReasoner()
    sig = SelfConsistencyJudge(stable, n=5).evaluate(case, stable.respond(case))
    assert sig["flag"] is False
    assert sig["consistency"] == 1.0


# --- citation faithfulness ----------------------------------------------------
def test_citation_faithful_not_flagged():
    case = _rag("R4")
    judge = CitationFaithfulnessJudge()
    sig = judge.evaluate(case, FaithfulCiteBot().respond(case))
    assert sig["flag"] is False


def test_citation_unfaithful_flagged():
    case = _rag("R1")
    judge = CitationFaithfulnessJudge()
    sig = judge.evaluate(case, UnfaithfulCiteBot().respond(case))
    assert sig["flag"] is True


# --- attack-type coverage -----------------------------------------------------
def test_multi_hop_and_outdated_cases_present():
    attack_types = {c.attack_type for c in load_cases(RAG)}
    assert {"multi_hop", "outdated_context"} <= attack_types


# --- optional NLI judge degrades gracefully -----------------------------------
def test_nli_judge_config_and_graceful_error():
    judge = NLIGroundednessJudge()
    assert judge.config()["type"] == "nli_entailment"
    case = _rag("R1")
    resp = TargetResponse("R1", "some confident answer")
    if importlib.util.find_spec("transformers") is None:
        with pytest.raises(RuntimeError, match="transformers"):
            judge.evaluate(case, resp)
    else:  # pragma: no cover - only when transformers is installed
        sig = judge.evaluate(case, resp)
        assert "entailment" in sig
