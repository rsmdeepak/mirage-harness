"""Mirage — a multi-modal AI hallucination & red-team harness."""
from .judges.bank import JudgeBank
from .judges.groundedness import GroundednessJudge
from .judges.novelty import NovelClaimJudge
from .judges.reasoning import ReasoningJudge

__all__ = [
    "JudgeBank", "GroundednessJudge", "NovelClaimJudge", "ReasoningJudge",
    "default_judge_bank",
]


def default_judge_bank() -> JudgeBank:
    """MVP judge bank.

    RAG is genuinely multi-signal: a primary groundedness judge plus an
    independent novel-claim judge, so the Judge Agreement diagnostic is
    populated. Reasoning uses one judge that emits several sub-signals
    (correctness, hint-resistance, explanation-consistency).
    """
    return JudgeBank({
        "rag": [GroundednessJudge(), NovelClaimJudge()],
        "reasoning": [ReasoningJudge()],
    })
