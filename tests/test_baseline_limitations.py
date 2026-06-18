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
