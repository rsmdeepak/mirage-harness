"""Golden-set attacker and loader.

For the MVP the "attacker" replays a fixed, labeled golden set so every metric
is computed against ground truth. The adaptive LLM attacker (milestone M4)
implements the same ``Attacker`` interface and drops in without runner changes.

Loading is type-strict: a JSON author who writes ``"false"`` (a string) for a
boolean, or a non-integer seed, gets a clear ``GoldenSchemaError`` rather than a
silently wrong label or a raw ``ValueError``.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.interfaces import AttackCase, Attacker

VALID_MODALITIES = {"rag", "reasoning", "image"}
REQUIRED_FIELDS = ("id", "modality", "attack_type", "prompt")
OPTIONAL_STR_FIELDS = ("context", "correct_answer", "hint", "expected_behavior")


class GoldenSchemaError(ValueError):
    """Raised when a golden test file is malformed."""


def _require_str(raw: dict, field: str, where: str, *, allow_empty=False) -> str:
    val = raw.get(field)
    if not isinstance(val, str) or (not allow_empty and not val.strip()):
        raise GoldenSchemaError(f"{where}: field '{field}' must be a non-empty string")
    return val


def _optional_str(raw: dict, field: str, where: str):
    val = raw.get(field)
    if val is None:
        return None
    if not isinstance(val, str):
        raise GoldenSchemaError(f"{where}: field '{field}' must be a string if present")
    return val


def _require_bool(raw: dict, field: str, where: str) -> bool:
    val = raw.get(field, False)
    if not isinstance(val, bool):  # rejects "false", 0, 1, etc. (fix #1)
        raise GoldenSchemaError(
            f"{where}: field '{field}' must be a JSON boolean (got {type(val).__name__})")
    return val


def _require_int(raw: dict, field: str, where: str, default: int = 0) -> int:
    val = raw.get(field, default)
    if isinstance(val, bool) or not isinstance(val, int):  # bool is an int subclass
        raise GoldenSchemaError(f"{where}: field '{field}' must be an integer")
    return val


def load_cases(*paths: str | Path) -> list[AttackCase]:
    cases: list[AttackCase] = []
    seen_ids: dict[str, str] = {}

    for path in paths:
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except FileNotFoundError as exc:
            raise GoldenSchemaError(f"golden file not found: {p}") from exc
        except json.JSONDecodeError as exc:
            raise GoldenSchemaError(f"{p}: invalid JSON ({exc})") from exc
        if not isinstance(data, list):
            raise GoldenSchemaError(f"{p}: top level must be a JSON array of cases")

        for i, raw in enumerate(data):
            where = f"{p}[{i}]"
            if not isinstance(raw, dict):
                raise GoldenSchemaError(f"{where}: each case must be an object")

            for field in REQUIRED_FIELDS:
                _require_str(raw, field, where)

            cid = raw["id"]
            if cid in seen_ids:
                raise GoldenSchemaError(
                    f"{where}: duplicate id '{cid}' (also in {seen_ids[cid]})")
            seen_ids[cid] = where

            modality = raw["modality"]
            if modality not in VALID_MODALITIES:
                raise GoldenSchemaError(
                    f"{where}: modality '{modality}' not in {sorted(VALID_MODALITIES)}")

            context = _optional_str(raw, "context", where)
            correct_answer = _optional_str(raw, "correct_answer", where)
            hint = _optional_str(raw, "hint", where)
            expected_behavior = raw.get("expected_behavior", "")
            if not isinstance(expected_behavior, str):
                raise GoldenSchemaError(f"{where}: 'expected_behavior' must be a string")
            judge_should_flag = _require_bool(raw, "judge_should_flag", where)
            seed = _require_int(raw, "seed", where)

            meta = raw.get("meta", {})
            if not isinstance(meta, dict):
                raise GoldenSchemaError(f"{where}: 'meta' must be an object if present")

            if modality == "rag" and judge_should_flag and not context:
                raise GoldenSchemaError(
                    f"{where}: rag trap case needs 'context' to evaluate groundedness")
            if modality == "reasoning" and not correct_answer:
                raise GoldenSchemaError(f"{where}: reasoning case needs 'correct_answer'")

            cases.append(AttackCase(
                id=cid, modality=modality, attack_type=raw["attack_type"],
                prompt=raw["prompt"], context=context, correct_answer=correct_answer,
                hint=hint, expected_behavior=expected_behavior,
                judge_should_flag=judge_should_flag, seed=seed, meta=meta,
            ))
    return cases


class GoldenAttacker(Attacker):
    def __init__(self, cases: list[AttackCase], source_paths: list[str] | None = None):
        self._cases = cases
        self.source_paths = source_paths or []

    @classmethod
    def from_files(cls, *paths: str | Path) -> "GoldenAttacker":
        return cls(load_cases(*paths), source_paths=[str(p) for p in paths])

    @classmethod
    def from_glob(cls, pattern: str, root: str | Path = ".") -> "GoldenAttacker":
        paths = sorted(Path(root).glob(pattern))
        if not paths:
            raise GoldenSchemaError(f"no golden files matched '{pattern}' under {root}")
        return cls.from_files(*paths)

    @classmethod
    def from_dir(cls, directory: str | Path) -> "GoldenAttacker":
        """Load every ``*.json`` golden file in a directory (sorted, dedup-checked)."""
        return cls.from_glob("*.json", root=directory)

    def filter(self, modality: str | None = None) -> "GoldenAttacker":
        if modality is None:
            return self
        return GoldenAttacker([c for c in self._cases if c.modality == modality],
                              source_paths=self.source_paths)

    def generate(self) -> list[AttackCase]:
        return list(self._cases)

    def config(self) -> dict:
        return {
            "type": "golden",
            "source_paths": self.source_paths,
            "num_cases": len(self._cases),
            "modalities": sorted({c.modality for c in self._cases}),
        }
