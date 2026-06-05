"""Test that tqdm is a declared core dependency.

Issue #88: tqdm must be importable after a fresh install.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"


class TestTqdmDependency:
    """Verify tqdm is declared in pyproject.toml dependencies."""

    def test_tqdm_in_dependencies(self):
        """tqdm>=4.60.0 must be listed in project.dependencies."""
        with open(PYPROJECT, "rb") as f:
            data = tomllib.load(f)

        deps = data["project"]["dependencies"]
        tqdm_deps = [d for d in deps if d.startswith("tqdm")]
        assert len(tqdm_deps) == 1, f"Expected exactly one tqdm dep, got: {tqdm_deps}"
        assert "4.60.0" in tqdm_deps[0], f"Expected tqdm>=4.60.0, got: {tqdm_deps[0]}"

    def test_tqdm_importable(self):
        """tqdm can be imported in the current environment."""
        from tqdm import tqdm  # noqa: F401
