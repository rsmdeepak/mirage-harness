"""Tests for the review-hardening fixes (judge errors, schema, consistency, meta)."""
import json
from pathlib import Path

import pytest

from mirage import default_judge_bank
from mirage.attackers.golden import GoldenSchemaError, load_cases
from mirage.core.interfaces import AttackCase, Judge, TargetResponse
from mirage.judges.bank import JudgeBank
from mirage.report.report import case_status, render_markdown, to_dict
from mirage.runner import run_case
from mirage.targets.synthetic import BadRAGBot, InconsistentReasoner, StableReasoner

ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "golden_tests" / "rag_policy_tests.json"
REASON = ROOT / "golden_tests" / "reasoning_hint_tests.json"


class CrashingJudge(Judge):
    name = "crashing"

    def evaluate(self, case, resp):
        raise ValueError("boom")


# --- fix #5: a crashing primary judge becomes an errored verdict, not a pass --
def test_judge_exception_is_contained():
    bank = JudgeBank({"rag": [CrashingJudge()]})
    case = load_cases(RAG)[0]
    r = run_case(BadRAGBot(), bank, case)
    assert r.verdict.errored is True
    assert r.verdict.flagged is False
    assert "boom" in r.verdict.signals["crashing"]["error"]


# --- fix #6: an errored case can never render as 'clear' ----------------------
def test_errored_never_renders_clear():
    bank = JudgeBank({"rag": [CrashingJudge()]})
    case = load_cases(RAG)[0]
    r = run_case(BadRAGBot(), bank, case)
    d = to_dict([r], "x", expected=lambda rr: True)
    assert d["cases"][0]["status"] == "errored"
    assert case_status(d["cases"][0]) != "clear"


# --- fix #7: golden schema validation -----------------------------------------
def test_schema_rejects_missing_field(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "X1", "modality": "rag"}]))  # no prompt/attack
    with pytest.raises(GoldenSchemaError):
        load_cases(bad)


def test_schema_rejects_duplicate_ids(tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([
        {"id": "D1", "modality": "rag", "attack_type": "a", "prompt": "p"},
        {"id": "D1", "modality": "rag", "attack_type": "a", "prompt": "p"},
    ]))
    with pytest.raises(GoldenSchemaError):
        load_cases(dup)


def test_schema_rejects_unknown_modality(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps([
        {"id": "M1", "modality": "audio", "attack_type": "a", "prompt": "p"},
    ]))
    with pytest.raises(GoldenSchemaError):
        load_cases(bad)


# --- fix #9: explanation-consistency is actually checked ----------------------
def test_explanation_inconsistency_flagged_even_when_answer_correct(bank):
    # InconsistentReasoner returns the CORRECT final answer but argues another
    # number in its explanation -> only the consistency check should trip.
    case = next(c for c in load_cases(REASON) if c.id == "N2")
    r = run_case(InconsistentReasoner(), bank, case)
    sig = r.verdict.signals["reasoning"]
    assert sig["is_correct"] is True
    assert sig["explanation_consistent"] is False
    assert r.verdict.flagged is True


def test_consistent_correct_answer_not_flagged(bank):
    case = next(c for c in load_cases(REASON) if c.id == "N2")
    r = run_case(StableReasoner(), bank, case)
    assert r.verdict.flagged is False


# --- fix #8: markdown export escapes pipes/newlines ---------------------------
def test_markdown_escaping():
    import re

    case = AttackCase(id="E1", modality="rag", attack_type="x",
                      prompt="p", context="c", judge_should_flag=True)

    class PipeTarget(BadRAGBot):
        name = "pipe_target"

        def respond(self, c):
            return TargetResponse(case_id=c.id, output="evil | pipe\nnewline", raw={})

    bank = JudgeBank({"rag": [_AlwaysFlag()]})
    r = run_case(PipeTarget(), bank, case)
    md = render_markdown([r], "m", expected=lambda rr: True)
    rows = [ln for ln in md.splitlines() if ln.startswith("| E1 ")]
    assert len(rows) == 1
    row = rows[0]
    # The row has exactly 6 columns -> 7 unescaped pipe delimiters. A leaked
    # pipe from the rationale ("bad | output") would push this to 8.
    unescaped_pipes = len(re.findall(r"(?<!\\)\|", row))
    assert unescaped_pipes == 7
    assert "\\|" in row  # the rationale's pipe survived as an escaped pipe


class _AlwaysFlag(Judge):
    name = "always_flag"

    def evaluate(self, case, resp):
        return {"flag": True, "reason": "bad | output\nhere"}


# --- multi-signal bank: RAG has >=2 voting judges, agreement is populated -----
def test_rag_bank_is_multi_signal_with_agreement(bank):
    case = next(c for c in load_cases(RAG) if c.id == "R1")
    r = run_case(BadRAGBot(), bank, case)
    sig = r.verdict.signals
    assert "groundedness" in sig and "novel_claim" in sig   # two independent signals
    assert "_agreement" in sig                              # diagnostic populated
    assert isinstance(sig["_agreement"], bool)


# --- fix #1: type-strict schema (booleans, seeds, meta) -----------------------
def test_schema_rejects_string_boolean(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps([{"id": "B1", "modality": "rag", "attack_type": "a",
                              "prompt": "p", "judge_should_flag": "false"}]))
    with pytest.raises(GoldenSchemaError):
        load_cases(p)


def test_schema_rejects_noninteger_seed(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": "S1", "modality": "rag", "attack_type": "a",
                              "prompt": "p", "seed": "soon"}]))
    with pytest.raises(GoldenSchemaError):
        load_cases(p)


def test_schema_rejects_nondict_meta(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps([{"id": "X1", "modality": "rag", "attack_type": "a",
                              "prompt": "p", "meta": "bad"}]))
    with pytest.raises(GoldenSchemaError):
        load_cases(p)


# --- fix #4: malformed/mismatched target response becomes errored -------------
def test_response_case_id_mismatch_is_errored(bank):
    case = load_cases(RAG)[0]

    class WrongIdTarget(BadRAGBot):
        name = "wrong_id"

        def respond(self, c):
            return TargetResponse(case_id="NOT_" + c.id, output="x")

    r = run_case(WrongIdTarget(), bank, case)
    assert r.verdict.errored is True
    assert "case_id" in r.verdict.rationale


# --- fix #2/#3/#7: session exports always carry full reproducibility metadata --
def test_session_export_has_required_repro_fields(bank, tmp_path):
    from mirage.attackers.golden import GoldenAttacker
    from mirage.report.report import export_session_json
    from mirage.runner import RunSession

    att = GoldenAttacker.from_files(RAG)
    session = RunSession.execute(att, BadRAGBot(), bank, expected_mode="label")
    payload = export_session_json(session, str(tmp_path / "s.json"))

    meta = payload["meta"]
    assert meta["attacker"]["type"] == "golden"        # attacker config (fix #3)
    assert meta["attacker"]["source_paths"]
    assert meta["target"]["name"] == "bad_rag_bot"
    assert meta["judges"]                              # judge config + thresholds
    assert meta["expected_mode"] == "label"
    env = meta["environment"]
    for key in ("timestamp_utc", "python", "platform", "dependencies"):
        assert key in env
    # per-case reproducibility
    assert all("seed" in c and "target_raw" in c for c in payload["cases"])


# --- fix #5: validate_detector reports judge agreement ------------------------
def test_validate_detector_reports_agreement(bank):
    from mirage.targets.synthetic import GoodRAGBot
    from mirage.validation import validate_detector

    v = validate_detector(bank, load_cases(RAG), BadRAGBot(), GoodRAGBot())
    assert v.metrics["judge_agreement"] is not None    # RAG has 2 voting judges


# --- fix #9: non-numeric short answers are scored -----------------------------
def test_reasoning_handles_boolean_answer(bank):
    from mirage.core.interfaces import AttackCase
    from mirage.judges.reasoning import ReasoningJudge

    case = AttackCase(id="Y", modality="reasoning", attack_type="x",
                      prompt="Is 4 even?", correct_answer="yes")
    good = ReasoningJudge().evaluate(case, TargetResponse("Y", "It is even.\nAnswer: yes"))
    bad = ReasoningJudge().evaluate(case, TargetResponse("Y", "Answer: no, it is odd"))
    assert good["is_correct"] is True and good["flag"] is False
    assert bad["is_correct"] is False and bad["flag"] is True


# --- regression risk #1: invalid expected_mode is rejected --------------------
def test_invalid_expected_mode_rejected():
    from mirage.runner import RunSession
    with pytest.raises(ValueError):
        RunSession(results=[], attacker_config={}, target_config={},
                   judges_config={}, expected_mode="known-good")  # hyphen typo


# --- malformed responses: non-string output, non-dict raw --------------------
def test_nonstring_output_is_errored(bank):
    case = load_cases(RAG)[0]

    class BadOutput(BadRAGBot):
        name = "bad_output"

        def respond(self, c):
            return TargetResponse(case_id=c.id, output=123)  # not a str

    r = run_case(BadOutput(), bank, case)
    assert r.verdict.errored is True


def test_nondict_raw_is_errored(bank):
    case = load_cases(RAG)[0]

    class BadRaw(BadRAGBot):
        name = "bad_raw"

        def respond(self, c):
            return TargetResponse(case_id=c.id, output="ok", raw=["not", "a", "dict"])

    r = run_case(BadRaw(), bank, case)
    assert r.verdict.errored is True


# --- duplicate IDs ACROSS files are rejected ----------------------------------
def test_duplicate_ids_across_files(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    payload = [{"id": "SAME", "modality": "rag", "attack_type": "a", "prompt": "p"}]
    a.write_text(json.dumps(payload))
    b.write_text(json.dumps(payload))
    with pytest.raises(GoldenSchemaError):
        load_cases(a, b)


# --- session markdown carries attacker/target/judge/seed/prompt/output --------
def test_session_markdown_has_repro_fields(bank):
    from mirage.attackers.golden import GoldenAttacker
    from mirage.report.report import render_session_markdown
    from mirage.runner import RunSession

    session = RunSession.execute(GoldenAttacker.from_files(RAG), BadRAGBot(),
                                 bank, expected_mode="label")
    md = render_session_markdown(session)
    assert "bad_rag_bot" in md            # target/model name
    assert "groundedness" in md           # judge config
    assert "golden" in md                 # attacker config
    assert "rag_policy_tests.json" in md  # source path / seed traceability


# --- paired-validation report is exportable -----------------------------------
def test_paired_validation_exportable(bank):
    from mirage.report.report import render_validation_markdown, validation_to_dict
    from mirage.targets.synthetic import GoodRAGBot
    from mirage.validation import validate_detector

    v = validate_detector(bank, load_cases(RAG), BadRAGBot(), GoodRAGBot())
    payload = validation_to_dict(v, "rag", "bad_rag_bot", "good_rag_bot", meta={})
    assert payload["mode"] == "paired_validation"
    assert payload["known_bad"]["cases"] and payload["known_good"]["cases"]
    md = render_validation_markdown(payload)
    assert "Paired Validation" in md and "True Positive Rate" in md
