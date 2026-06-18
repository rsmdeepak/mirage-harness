"""Deterministic known-good / known-bad targets.

These make NO model calls. They exist to *validate and calibrate Mirage itself*
before any real model is wired in: a known-bad target must reliably trip the
detectors (true positives) and a known-good target must reliably pass (true
negatives). Run these first, tune thresholds, then point Mirage at real models.
"""
from __future__ import annotations

from ..core.interfaces import AttackCase, Target, TargetResponse
from ..core.text import tokens


class BadRAGBot(Target):
    """Always answers confidently, ignoring whether the context supports it.

    Used to measure True Positive Rate: every confident, ungrounded answer it
    produces on an unanswerable/false-premise case should be flagged.
    """

    name = "bad_rag_bot"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "always_confident_ignores_context"}

    def respond(self, case: AttackCase) -> TargetResponse:
        # Fabricate a plausible-sounding answer that does not draw on context.
        fabricated = case.meta.get(
            "fabricated_answer",
            "Yes. Based on our standard policy, this is fully covered and "
            "approved without exception.",
        )
        return TargetResponse(case_id=case.id, output=fabricated,
                              raw={"strategy": "always_confident"})


class GoodRAGBot(Target):
    """Answers only when the context plausibly supports it; otherwise refuses.

    Naive keyword retrieval stands in for real grounding: if the question's
    content words overlap a context sentence, it answers with that sentence;
    otherwise it says the answer is not in its sources.

    Known limitation (by design, not a bug): on a *false-premise* question whose
    keywords happen to appear in the context, it returns the relevant grounded
    sentence rather than explicitly rejecting the premise. That is still
    not-a-hallucination (the answer is grounded in context), so the groundedness
    judge correctly does not flag it -- but a real good RAG bot would also
    surface the contradiction. Replace this with an Ollama+Chroma target to get
    genuine premise rejection.
    """

    name = "good_rag_bot"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "keyword_retrieval_or_refuse"}

    def respond(self, case: AttackCase) -> TargetResponse:
        context = case.context or ""
        q_terms = set(tokens(case.prompt))
        # Find the context sentence with the most overlap with the question.
        best, best_score = "", 0
        for sentence in _split_sentences(context):
            score = len(q_terms & set(tokens(sentence)))
            if score > best_score:
                best, best_score = sentence, score
        if best_score >= 1:
            return TargetResponse(case_id=case.id, output=best.strip(),
                                  raw={"strategy": "grounded", "overlap": best_score})
        return TargetResponse(case_id=case.id,
                              output="Not found in the sources.",
                              raw={"strategy": "refused"})


class MisleadingReasoner(Target):
    """Always adopts the injected hint, even when the hint is wrong.

    Used to measure False Negative Rate on misleading-hint tests: it should be
    flagged because it produces an incorrect final answer under a bad hint.
    """

    name = "misleading_reasoner"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "always_follows_hint"}

    def respond(self, case: AttackCase) -> TargetResponse:
        if case.hint:
            answer = case.hint
            explanation = f"You suggested {case.hint}, so the answer is {case.hint}."
        else:
            answer = case.correct_answer or ""
            explanation = "Working from the given facts."
        return TargetResponse(case_id=case.id,
                              output=f"{explanation}\nAnswer: {answer}",
                              raw={"strategy": "follows_hint"})


class StableReasoner(Target):
    """Ignores misleading hints and solves from the given facts (control).

    Used to measure True Negative Rate on misleading-hint tests: it should NOT
    be flagged because it returns the correct answer regardless of the hint.
    """

    name = "stable_reasoner"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "ignores_hint"}

    def respond(self, case: AttackCase) -> TargetResponse:
        answer = case.correct_answer or ""
        note = " I am ignoring the misleading hint." if case.hint else ""
        return TargetResponse(case_id=case.id,
                              output=f"Reasoning from the given facts.{note}\nAnswer: {answer}",
                              raw={"strategy": "ignores_hint"})


class InconsistentReasoner(Target):
    """Lands on the CORRECT final answer but argues a different number along the
    way -- used to validate the explanation-consistency check (the final answer
    is right, so only inconsistency should trip the judge)."""

    name = "inconsistent_reasoner"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "explanation_contradicts_answer"}

    def respond(self, case: AttackCase) -> TargetResponse:
        correct = case.correct_answer or "0"
        wrong = case.hint or "999"
        return TargetResponse(
            case_id=case.id,
            output=f"Working it out, the result clearly comes to {wrong}.\nAnswer: {correct}",
            raw={"strategy": "explanation_contradicts_answer"},
        )


class FlakyReasoner(Target):
    """Non-deterministic: alternates between the correct answer and a wrong one
    on successive calls -- used to exercise the self-consistency judge."""

    name = "flaky_reasoner"

    def __init__(self):
        self._calls = 0

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": False,
                "strategy": "alternates_answer_each_call"}

    def respond(self, case: AttackCase) -> TargetResponse:
        self._calls += 1
        answer = case.correct_answer if self._calls % 2 else (case.hint or "0")
        return TargetResponse(case_id=case.id, output=f"Thinking...\nAnswer: {answer}",
                              raw={"call": self._calls})


class FaithfulCiteBot(Target):
    """Answers from, and cites, the supporting context sentence (faithful)."""

    name = "faithful_cite_bot"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "cites_supporting_chunk"}

    def respond(self, case: AttackCase) -> TargetResponse:
        chunk = (case.context or "").strip()
        return TargetResponse(case_id=case.id, output=chunk or "Not found in the sources.",
                              raw={"cited_chunk": chunk})


class UnfaithfulCiteBot(Target):
    """Returns a confident claim but cites a chunk that does not support it."""

    name = "unfaithful_cite_bot"

    def config(self) -> dict:
        return {"name": self.name, "kind": "synthetic", "deterministic": True,
                "strategy": "cites_unsupporting_chunk"}

    def respond(self, case: AttackCase) -> TargetResponse:
        claim = case.meta.get("fabricated_answer",
                              "Yes, this is fully approved without exception.")
        return TargetResponse(case_id=case.id, output=claim,
                              raw={"cited_chunk": case.context or ""})


class ExplodingTarget(Target):
    """Always raises -- used to verify graceful-failure handling (test C5)."""

    name = "exploding_target"

    def respond(self, case: AttackCase) -> TargetResponse:
        raise RuntimeError("simulated target outage")


def _split_sentences(text: str) -> list[str]:
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out
