"""Mirage dashboard.

    streamlit run app/streamlit_app.py

Two modes:
  * Single target  -- run one target against the golden set; inspect per-case
    outcomes, metrics, drill-down, and download Markdown/JSON reports.
  * Paired validation -- run the matching known-good + known-bad pair for a
    modality and show the combined detector confusion matrix (TPR/TNR/FPR/FNR),
    which is the methodology the CLI uses.

Results persist in session state, so interacting with widgets does not wipe the
run. Synthetic targets need no models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demos.baseline_limitations_demo import CASES as ADVERSARIAL_CASES  # noqa: E402
from mirage import default_judge_bank  # noqa: E402
from mirage.attackers.golden import GoldenAttacker  # noqa: E402
from mirage.core.interfaces import AttackCase, TargetResponse  # noqa: E402
from mirage.judges.groundedness import GroundednessJudge  # noqa: E402
from mirage.meta import environment_metadata  # noqa: E402
from mirage.report.report import (  # noqa: E402
    render_session_markdown, render_validation_markdown, session_to_dict,
    validation_to_dict,
)
from mirage.runner import RunSession  # noqa: E402
from mirage.targets.synthetic import (  # noqa: E402
    BadRAGBot, GoodRAGBot, InconsistentReasoner, MisleadingReasoner, StableReasoner,
)
from mirage.validation import validate_detector  # noqa: E402

GOLDEN = ROOT / "golden_tests"
GOLDEN_FILE = {"rag": "rag_policy_tests.json", "reasoning": "reasoning_hint_tests.json"}

SINGLE_TARGETS = {
    "Bad RAG bot (known-bad)": (BadRAGBot, "rag", "label"),
    "Good RAG bot (known-good)": (GoodRAGBot, "rag", "known_good"),
    "Misleading reasoner (known-bad)": (MisleadingReasoner, "reasoning", "label"),
    "Stable reasoner (known-good)": (StableReasoner, "reasoning", "known_good"),
    "Inconsistent reasoner (known-bad)": (InconsistentReasoner, "reasoning", "label"),
}
PAIRS = {
    "RAG (Bad vs Good RAG bot)": ("rag", BadRAGBot, GoodRAGBot),
    "Reasoning (Misleading vs Stable)": ("reasoning", MisleadingReasoner, StableReasoner),
}
STATUS_STYLE = {
    "TP": "background-color:#1b5e20", "TN": "background-color:#1b5e20",
    "FP": "background-color:#b71c1c", "FN": "background-color:#b71c1c",
    "errored": "background-color:#5d4037",
}


def _attacker(modality: str) -> GoldenAttacker:
    return GoldenAttacker.from_files(GOLDEN / GOLDEN_FILE[modality])


def _metric_row(s):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TPR", s["TPR"])
    c2.metric("TNR", s["TNR"])
    c3.metric("FPR", s["FPR"])
    c4.metric("FNR", s["FNR"])
    c5.metric("Judge agreement", s["judge_agreement"])


def _run_adversarial():
    """Feed the lexical groundedness judge the adversarial cases from the demo
    and tally the confusion matrix. Deterministic and instant -- no caching."""
    judge = GroundednessJudge()
    rows, tp, tn, fp, fn = [], 0, 0, 0, 0
    for should_flag, ctx, ans, note in ADVERSARIAL_CASES:
        case = AttackCase(id="D", modality="rag", attack_type="adversarial",
                          prompt="q", context=ctx)
        sig = judge.evaluate(case, TargetResponse("D", ans))
        flagged = bool(sig["flag"])
        if should_flag and flagged:
            outcome = "TP"; tp += 1
        elif should_flag and not flagged:
            outcome = "FN"; fn += 1
        elif not should_flag and flagged:
            outcome = "FP"; fp += 1
        else:
            outcome = "TN"; tn += 1
        rows.append({"note": note, "context": ctx[:55], "answer": ans[:55],
                     "overlap": sig["score"], "should_flag": should_flag,
                     "flagged": flagged, "outcome": outcome})

    def rate(n, d):
        return round(n / d, 3) if d else 0.0

    metrics = {"TPR": rate(tp, tp + fn), "TNR": rate(tn, tn + fp),
               "FPR": rate(fp, fp + tn), "FNR": rate(fn, fn + tp),
               "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}
    return rows, metrics


def _case_table(cases):
    rows = [{
        "id": c["id"], "attack": c["attack_type"], "status": c["status"],
        "outcome": c["outcome"], "expected_flag": c["expected_flag"],
        "judge_errors": ", ".join(c.get("judge_errors", [])),
        "rationale": c["rationale"], "output": (c["output"] or "")[:80],
    } for c in cases]
    df = pd.DataFrame(rows)
    st.dataframe(df.style.map(lambda v: STATUS_STYLE.get(v, ""), subset=["outcome"]),
                 use_container_width=True)


st.set_page_config(page_title="Mirage", layout="wide")
st.title("🌫️ Mirage — Multi-Modal AI Red-Team Harness")
st.caption("An AI lie detector for text and reasoning (image modality = bonus path).")

mode = st.sidebar.radio(
    "Mode", ["Single target", "Paired validation", "Adversarial stress test"])

# --- Single-target mode -------------------------------------------------------
if mode == "Single target":
    choice = st.sidebar.selectbox("Target under test", list(SINGLE_TARGETS))
    target_cls, modality, expected_mode = SINGLE_TARGETS[choice]
    if st.sidebar.button("▶ Run harness", type="primary"):
        session = RunSession.execute(_attacker(modality), target_cls(),
                                     default_judge_bank(), expected_mode)
        st.session_state["single"] = {
            "report": session_to_dict(session),
            "markdown": render_session_markdown(session),
            "label": choice,
        }

    state = st.session_state.get("single")
    if not state:
        st.info("Pick a target and click **Run harness**. Try a known-bad target "
                "(should be flagged), then a known-good one (should pass).")
        st.stop()

    d = state["report"]
    st.subheader(f"Results — {state['label']}")
    _metric_row(d["summary"])
    if d["summary"]["counts"]["errored"]:
        st.warning(f"{d['summary']['counts']['errored']} case(s) errored — "
                   "shown as 'errored', never as a pass.")
    if d["secondary_judge_errors"]:
        st.warning(f"{d['secondary_judge_errors']} case(s) had a secondary judge "
                   "error (primary judge still decided the verdict).")
    _case_table(d["cases"])

    with st.expander("🔎 Drill into a case", expanded=True):
        cid = st.selectbox("Case", [c["id"] for c in d["cases"]])
        case = next(c for c in d["cases"] if c["id"] == cid)
        st.write("**Prompt**", case["prompt"])
        if case.get("context"):
            st.write("**Context**", case["context"])
        st.write("**Target output**", case["output"])
        status_label = {"errored": "⚠️ errored", "flagged": "🚩 flagged",
                        "clear": "✅ clear"}[case["status"]]
        st.write("**Verdict**", status_label, "·", case["rationale"])
        st.json(case["signals"])
        st.caption(f"Reproduce with seed={case['seed']} · "
                   f"target={d['meta'].get('target', {}).get('name')}")

    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇ JSON report", data=json.dumps(d, indent=2),
                        file_name="mirage_report.json", mime="application/json")
    dl2.download_button("⬇ Markdown report", data=state["markdown"],
                        file_name="mirage_report.md", mime="text/markdown")

# --- Paired-validation mode ---------------------------------------------------
elif mode == "Paired validation":
    choice = st.sidebar.selectbox("Modality pair", list(PAIRS))
    modality, bad_cls, good_cls = PAIRS[choice]
    if st.sidebar.button("▶ Run paired validation", type="primary"):
        att = _attacker(modality)
        bank = default_judge_bank()
        v = validate_detector(bank, att.generate(), bad_cls(), good_cls())
        meta = {"attacker": att.config(), "judges": bank.config(),
                "known_bad": bad_cls().config(), "known_good": good_cls().config(),
                "environment": environment_metadata()}
        payload = validation_to_dict(v, modality, bad_cls().name, good_cls().name, meta)
        st.session_state["paired"] = {
            "payload": payload,
            "markdown": render_validation_markdown(payload),
            "label": choice,
        }

    state = st.session_state.get("paired")
    if not state:
        st.info("Pick a modality pair and click **Run paired validation** to see "
                "the combined known-good/known-bad detector confusion matrix.")
        st.stop()

    payload = state["payload"]
    st.subheader(f"Paired detector validation — {state['label']}")
    _metric_row(payload["metrics"])
    st.caption(f"counts: {payload['metrics']['counts']}")
    st.markdown("**Known-bad target (true positives / false negatives)**")
    _case_table(payload["known_bad"]["cases"])
    st.markdown("**Known-good target (true negatives / false positives)**")
    _case_table(payload["known_good"]["cases"])

    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇ JSON report", data=json.dumps(payload, indent=2),
                        file_name="mirage_paired_validation.json", mime="application/json")
    dl2.download_button("⬇ Markdown report", data=state["markdown"],
                        file_name="mirage_paired_validation.md", mime="text/markdown")

# --- Adversarial stress-test mode ---------------------------------------------
else:
    st.sidebar.caption("Feeds the lexical groundedness judge hand-built "
                       "adversarial RAG cases (paraphrases + keyword-smuggled "
                       "claims) to expose where word overlap ≠ entailment.")
    rows, metrics = _run_adversarial()

    st.subheader("Adversarial stress test — lexical groundedness judge")
    st.warning(
        "⚠️ This is **not** the calibration result. Unlike paired validation "
        "(perfectly separable synthetic bots → FPR/FNR = 0), these cases are "
        "built to **break** the lexical baseline, so FPR/FNR are intentionally "
        "non-zero — the regression target for the opt-in NLI entailment judge.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TPR", metrics["TPR"])
    c2.metric("TNR", metrics["TNR"])
    c3.metric("FPR", metrics["FPR"], help="Good answers wrongly flagged "
              "(paraphrases with low word overlap).")
    c4.metric("FNR", metrics["FNR"], help="Bad answers missed "
              "(keyword-smuggled claims with high word overlap).")
    st.caption(f"counts: {metrics['counts']}  ·  flag threshold = 0.45 overlap "
               f"·  source: demos/baseline_limitations_demo.py")

    df = pd.DataFrame(rows)
    st.dataframe(df.style.map(lambda v: STATUS_STYLE.get(v, ""), subset=["outcome"]),
                 use_container_width=True)
