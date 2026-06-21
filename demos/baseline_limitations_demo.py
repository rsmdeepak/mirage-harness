"""Standalone demo: where the lexical groundedness baseline FAILS.

`run_validation.py` scores deterministic, cleanly-separable synthetic bots, so
it reports FPR = FNR = 0 -- that is a *calibration* result, not a claim about
real models. This demo does the opposite: it feeds the lexical
``GroundednessJudge`` *adversarial* cases that stress its core weakness -- it
counts word overlap, not entailment -- and prints a confusion matrix with
non-zero FPR and FNR.

Two failure modes (the regression targets to beat once an NLI judge replaces
the lexical proxy):

  * False negative -- a wrong answer that parrots context keywords but smuggles
    in an unsupported claim. High overlap, so the baseline MISSES it.
  * False positive -- a correct answer paraphrased with synonyms. Low overlap,
    so the baseline FALSE-FLAGS it.

    python demos/baseline_limitations_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirage.core.interfaces import AttackCase, TargetResponse
from mirage.judges.groundedness import GroundednessJudge

# (should_flag, context, answer, note)
#   should_flag=True  -> answer is a hallucination; a correct judge SHOULD flag it
#   should_flag=False -> answer is grounded;        a correct judge should NOT flag it
CASES = [
    # --- FALSE NEGATIVES: wrong answer, high overlap (smuggled claim) -----------
    (True,
     "Refunds are available within 30 days for clothing purchased in store.",
     "Refunds are available within 30 days for clothing purchased in store, "
     "and electronics also receive a full cash refund.",
     "keyword-smuggled hallucination"),
    (True,
     "The warranty covers manufacturing defects on kitchen appliances for one year.",
     "The warranty covers manufacturing defects on kitchen appliances for one "
     "year, and also covers accidental water damage to phones.",
     "keyword-smuggled hallucination"),

    # --- TRUE POSITIVES: blatant fabrication, low overlap (baseline catches it) --
    (True,
     "Clothing may be returned within 30 days with a receipt.",
     "Electronics qualify for a ninety day money back guarantee, no receipt needed.",
     "blatant fabrication"),
    (True,
     "The museum has operated at 14 Elm Street since it opened in 1962.",
     "It relocated downtown in 2015 to house a growing modern art collection.",
     "blatant fabrication"),

    # --- FALSE POSITIVES: correct answer, paraphrased with synonyms (low overlap)
    (False,
     "Clothing may be returned within 30 days of purchase.",
     "Garments can be sent back inside a one-month window.",
     "paraphrased grounded answer"),
    (False,
     "Employees accrue 15 vacation days per year.",
     "Staff earn three weeks of annual paid leave.",
     "paraphrased grounded answer"),

    # --- TRUE NEGATIVES: grounded answer that reuses context wording ------------
    (False,
     "The daily parking rate is 8 dollars for visitors.",
     "The daily parking rate is 8 dollars for visitors.",
     "verbatim grounded answer"),
    (False,
     "The standard warranty lasts one year from the purchase date.",
     "The standard warranty lasts one year from the purchase date.",
     "verbatim grounded answer"),
]


def _rate(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def main() -> None:
    judge = GroundednessJudge()
    tp = tn = fp = fn = 0
    rows = []
    for should_flag, ctx, ans, note in CASES:
        case = AttackCase(id="D", modality="rag", attack_type="adversarial",
                          prompt="q", context=ctx)
        sig = judge.evaluate(case, TargetResponse("D", ans))
        flagged = bool(sig["flag"])
        if should_flag and flagged:
            tp += 1; outcome = "TP"
        elif should_flag and not flagged:
            fn += 1; outcome = "FN  <- missed hallucination"
        elif not should_flag and flagged:
            fp += 1; outcome = "FP  <- false alarm"
        else:
            tn += 1; outcome = "TN"
        rows.append((outcome, sig["score"], note))

    print("=== Baseline limitations demo (lexical GroundednessJudge) ===\n")
    print(f"  {'outcome':<28} {'overlap':>7}  note")
    print(f"  {'-'*28} {'-'*7}  {'-'*32}")
    for outcome, score, note in rows:
        print(f"  {outcome:<28} {score:>7.2f}  {note}")

    print("\n  confusion: "
          f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  TPR={_rate(tp, tp + fn)}  TNR={_rate(tn, tn + fp)}  "
          f"FPR={_rate(fp, fp + tn)}  FNR={_rate(fn, fn + tp)}")
    print("\n  ^ Non-zero FPR/FNR: the lexical baseline counts word overlap, not")
    print("    entailment. Swap in the NLI judge (mirage/judges/nli.py) to beat this.")


if __name__ == "__main__":
    main()
