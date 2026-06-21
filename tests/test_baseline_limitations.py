"""Documents the KNOWN limitations of the lexical-overlap groundedness baseline.

These tests assert the baseline's *wrong* behavior on purpose. They exist so the
capstone is honest about what the MVP judge can and cannot do, and they become
the regression target to beat once an NLI entailment judge replaces the lexical
proxy (review fix #10). When you swap in NLI, these expectations should flip.
"""
from mirage.core.interfaces import AttackCase, TargetResponse
from mirage.judges.groundedness import GroundednessJudge


def _case(context):
    return AttackCase(id="L", modality="rag", attack_type="x", prompt="q",
                      context=context)


def test_limitation_copied_keyword_hallucination_is_missed():
    """A confident answer that parrots context keywords but smuggles in an
    unsupported claim has high lexical overlap, so the baseline MISSES it
    (false negative). NLI would catch the unentailed clause."""
    judge = GroundednessJudge()
    context = "Refunds are available within 30 days for clothing purchased in store."
    answer = ("Refunds are available within 30 days for clothing purchased in "
              "store, and also include a full cash refund for electronics.")
    sig = judge.evaluate(_case(context), TargetResponse("L", answer))
    assert sig["flag"] is False          # <-- known limitation (should be True)
    assert sig["score"] >= 0.45


def test_limitation_paraphrased_grounded_answer_is_false_flagged():
    """A correct answer that paraphrases the context with synonyms has low
    lexical overlap, so the baseline FALSE-FLAGS it (false positive). NLI would
    recognize the entailment."""
    judge = GroundednessJudge()
    context = "Clothing may be returned within 30 days of purchase."
    answer = "Garments can be sent back inside a one month window."
    sig = judge.evaluate(_case(context), TargetResponse("L", answer))
    assert sig["flag"] is True           # <-- known limitation (should be False)
    assert sig["score"] < 0.45


def test_limitation_second_smuggled_claim_is_missed():
    """Second false-negative example (mirrors demos/baseline_limitations_demo).
    The unsupported 'water damage to phones' clause rides on top of fully
    grounded warranty wording, so overlap stays above threshold."""
    judge = GroundednessJudge()
    context = "The warranty covers manufacturing defects on kitchen appliances for one year."
    answer = ("The warranty covers manufacturing defects on kitchen appliances "
              "for one year, and also covers accidental water damage to phones.")
    sig = judge.evaluate(_case(context), TargetResponse("L", answer))
    assert sig["flag"] is False          # <-- known limitation (should be True)
    assert sig["score"] >= 0.45


def test_limitation_second_paraphrase_is_false_flagged():
    """Second false-positive example: a correct answer ('three weeks' == 15
    days of leave) phrased with no shared content words is wrongly flagged."""
    judge = GroundednessJudge()
    context = "Employees accrue 15 vacation days per year."
    answer = "Staff earn three weeks of annual paid leave."
    sig = judge.evaluate(_case(context), TargetResponse("L", answer))
    assert sig["flag"] is True           # <-- known limitation (should be False)
    assert sig["score"] < 0.45


def test_demo_confusion_matrix_has_nonzero_fpr_and_fnr():
    """The standalone demo exists to show the lexical baseline producing real
    false positives AND false negatives. Guard that property so the demo can't
    silently regress to a perfectly-separable (and misleading) result."""
    from demos.baseline_limitations_demo import CASES

    judge = GroundednessJudge()
    fp = fn = 0
    for should_flag, ctx, ans, _note in CASES:
        flagged = judge.evaluate(_case(ctx), TargetResponse("L", ans))["flag"]
        fn += should_flag and not flagged
        fp += (not should_flag) and flagged
    assert fp > 0 and fn > 0
