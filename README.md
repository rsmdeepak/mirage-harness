# Mirage — Hallucination & Red-Team Harness

Mirage is a small, deterministic capstone harness for testing hallucination and
reasoning-failure detectors. It is built around a pluggable pipeline:

```text
Attacker -> Target -> Judge -> Report
```

The current implementation validates the detector against synthetic
known-good/known-bad targets before any real model is connected. That makes the
demo reproducible and keeps the scope honest: Mirage currently proves the
harness architecture, metrics, reporting, and two baseline text evaluators. It
does not yet claim production-grade semantic hallucination detection.

## Designed vs. Implemented

| Capability | Designed | Implemented now |
| --- | --- | --- |
| Pluggable pipeline | Attacker, Target, Judge, and Report roles can be swapped independently. | Yes. Core ABC/dataclass contracts live in `mirage/core/interfaces.py`; execution lives in `mirage/runner.py`. |
| Golden test loading | Load labeled cases from `golden_tests/*.json`, validate schema, reject malformed labels. | Yes. `GoldenAttacker.from_files`, `from_glob`, and `from_dir` load sorted JSON cases with duplicate-ID and type validation. |
| Synthetic known-good / known-bad bots | Use deterministic targets to calibrate detector behavior before real model integration. | Yes. RAG and reasoning bots live in `mirage/targets/synthetic.py`. |
| RAG hallucination evaluation | Detect unsupported answers, false premises, citation failures, contradictions, outdated facts, and multi-hop grounding issues. | Mostly. Default judges are lexical (overlap + novel-claim). Adversarial prompt categories now include unanswerable, false-premise, multi-hop, and outdated-context (all surface as groundedness failures). An opt-in `CitationFaithfulnessJudge` verifies answers against the cited chunk, and an optional model-backed `NLIGroundednessJudge` adds entailment/contradiction scoring. Still no *staleness* or *true multi-hop reasoning* detection — those prompts are caught only as ungrounded answers. |
| Reasoning robustness evaluation | Detect incorrect final answers, misleading-hint susceptibility, inconsistent explanations, overthinking, and self-consistency failures. | Yes. Short-form correctness, hint following, numeric explanation/final-answer consistency, and an overthinking heuristic (reasoning-length threshold). A real `SelfConsistencyJudge` (resamples the target) ships as an opt-in judge with a non-deterministic synthetic target to exercise it. |
| Multi-signal judge bank | Run multiple judges and report agreement as a diagnostic, not as a quality score. | Yes for RAG: groundedness plus novel-claim judges. Reasoning uses one judge with multiple sub-signals. Judge agreement is reported separately and does not affect TPR/TNR/FPR/FNR. |
| Detector metrics | Compute TPR, TNR, FPR, FNR from known-bad and known-good paired validation. | Yes. Implemented in `mirage/report/report.py` and `mirage/validation.py`. |
| Reproducibility logging | Log prompt, context, output, model/target config, judge config, attacker config, seed, expected-label mode, environment, and dependency versions. | Yes when using `RunSession` and session export helpers. Low-level export functions remain available for internal/simple use. |
| Streamlit dashboard | Show single-target runs and paired known-good/known-bad validation with downloadable reports. | Yes. `app/streamlit_app.py` has single-target and paired-validation modes with JSON and Markdown downloads. |
| Markdown / JSON report export | Export reproducible per-case reports. | Yes. Reports include metrics, case outcomes, signals, errors, raw target metadata, and run metadata. |
| Suggested fix field | Emit a remediation suggestion per failure. | Yes. Reports include a per-case `suggested_fix` derived from the failure category (only on flagged/errored cases). |
| Optional model-backed judges | Swap lexical baselines for NLI/LLM judges behind the same interface. | `NLIGroundednessJudge` is implemented (lazy `transformers` import, clear error if absent) and gives entailment + contradiction scores. LLM-as-judge is still designed-only. |
| Image / pixel modality | Add image targets and VLM judges behind the same interfaces. | Designed only. No image target, judge, dataset, or dashboard workflow ships in this MVP. |

## What This MVP Demonstrates

- The harness architecture is pluggable and modality-aware.
- Golden datasets can be loaded from JSON with schema validation.
- Synthetic known-bad targets trigger the detector and known-good targets pass.
- Detector metrics are computed from paired validation.
- RAG evaluation has two independent lexical signals.
- Reasoning evaluation handles short-form answer correctness, misleading hints,
  and visible numeric explanation consistency.
- Reports and dashboard outputs include reproducibility metadata.
- Target and judge failures are contained and surfaced as errors, not silent
  passes.

## What This MVP Does Not Claim

Mirage does not yet provide a production-grade hallucination detector. In
particular:

- The **default** RAG judges are lexical, not semantic entailment (an NLI judge
  is available but opt-in and not in the default bank).
- It does not detect *staleness* itself — outdated-context prompts are caught
  only because the fabricated answer is ungrounded, not because the judge knows
  the context is old.
- It does not perform *true multi-hop reasoning* — multi-hop prompts are caught
  as ordinary groundedness failures.
- Contradiction detection exists only via the optional NLI judge; there is no
  deterministic contradiction judge or contradiction-specific golden case.
- LLM-as-judge scoring is not implemented.
- Overthinking detection is a coarse reasoning-length heuristic, not a
  difficulty-aware measure.
- Citation checking requires the target to expose its cited chunk; it is a
  lexical overlap check, not entailment.
- It does not implement image/pixel evaluation.

Those remaining items require model-backed judges or richer golden data.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
python run_validation.py
streamlit run app/streamlit_app.py
```

The MVP runs without model downloads, Ollama, torch, or external APIs.

## Current Validation Results

The current golden set has 42 labeled cases:

- 24 RAG cases (unanswerable, false-premise, multi-hop, outdated-context, grounded)
- 18 reasoning cases
- 73 paired known-good/known-bad scored evaluations

Current synthetic validation:

```text
RAG module        TPR=1.0  TNR=1.0  FPR=0.0  FNR=0.0
Reasoning module  TPR=1.0  TNR=1.0  FPR=0.0  FNR=0.0
golden cases: 42 (target >= 20 PASS)
```

These numbers are synthetic calibration results, not claims about real model
performance.

## Implemented Modules

| Module | Files |
| --- | --- |
| Core interfaces | `mirage/core/interfaces.py` |
| Runner and reproducible sessions | `mirage/runner.py` |
| Golden attacker and schema validation | `mirage/attackers/golden.py` |
| Synthetic targets | `mirage/targets/synthetic.py` |
| RAG lexical groundedness judge | `mirage/judges/groundedness.py` |
| RAG novel-claim lexical judge | `mirage/judges/novelty.py` |
| Reasoning judge (correctness/hint/consistency/overthinking) | `mirage/judges/reasoning.py` |
| Citation-faithfulness judge (opt-in) | `mirage/judges/citation.py` |
| Self-consistency judge (opt-in) | `mirage/judges/consistency.py` |
| NLI entailment judge (optional, model-backed) | `mirage/judges/nli.py` |
| Judge routing and agreement diagnostic | `mirage/judges/bank.py` |
| Metrics and report export | `mirage/report/report.py` |
| Environment metadata | `mirage/meta.py` |
| Paired detector validation | `mirage/validation.py` |
| Streamlit dashboard | `app/streamlit_app.py` |

## Tests

The test suite covers:

- pipeline execution
- reproducibility
- Markdown and JSON export
- graceful target failure
- primary and secondary judge errors
- golden schema validation
- duplicate IDs across files
- target response validation
- RAG synthetic hallucination detection
- reasoning misleading-hint detection
- explanation consistency
- metric aggregation
- judge agreement diagnostics
- paired-validation export
- known limitations of lexical RAG judging

Run:

```bash
python -m pytest -q
```

## Known Limitations

The RAG baseline has two deliberately documented failure modes in
`tests/test_baseline_limitations.py`:

- A copied-keyword hallucination can be missed because lexical overlap is high.
- A paraphrased grounded answer can be false-flagged because lexical overlap is
  low.

These tests are intentionally "wrong-on-purpose" today. When an NLI entailment
judge is added, those expectations should flip.

The reasoning judge is also scoped to short-form answers: numbers, booleans,
single words, and multiple-choice letters. Long free-form answers need a
semantic-equivalence judge.

## Next Work To Make The Designed Scope True

Done in this iteration: structured `suggested_fix`; multi-hop and
outdated-context golden cases; opt-in `CitationFaithfulnessJudge`; opt-in
`SelfConsistencyJudge` (+ non-deterministic target); overthinking heuristic;
optional model-backed `NLIGroundednessJudge` (entailment + contradiction).

Still open:

1. Wire the NLI judge into a default bank once `transformers` is available, and
   flip the lexical-baseline limitation tests.
2. Add a difficulty-aware overthinking metric (not just reasoning length).
3. Add a deterministic contradiction judge, or rely on NLI contradiction scores.
4. Add real RAG targets (Ollama + ChromaDB) behind `Target.respond`.
5. Add an LLM-as-judge signal.
6. Add image targets and VLM judges only after text/RAG claims are solid.

## Repository Layout

```text
mirage/
  core/        interfaces + text utilities
  attackers/   golden-set loader + attacker
  targets/     synthetic known-good / known-bad bots
  judges/      RAG and reasoning judges
  report/      metrics, JSON export, Markdown export
  meta.py      environment metadata for reproducibility
  runner.py    Attacker -> Target -> Judge execution
  validation.py paired known-good/known-bad detector validation
golden_tests/  labeled validation cases
app/           Streamlit dashboard
tests/         MVP and hardening tests
run_validation.py
```
