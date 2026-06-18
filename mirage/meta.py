"""Run/environment metadata for reproducible reports (review fix #2).

Captures everything needed to reproduce a run later: timestamp, interpreter,
platform, git commit (when available), and versions of the libraries that
actually affect results.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

# Libraries whose versions can change model behavior / scoring.
_TRACKED = [
    "streamlit", "pandas", "pytest", "transformers", "torch",
    "sentence-transformers", "chromadb", "diffusers", "ollama",
]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def environment_metadata() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "dependencies": dependency_versions(),
    }
