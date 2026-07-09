"""Repo-anchored paths, so entry points work from any working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"
DOCS_DATA = REPO_ROOT / "docs" / "data"
