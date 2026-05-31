"""Issue #52 — stale src/ directory must not exist.

After the package rename to hmm_futures_analysis/, the old src/ directory
was left behind with only __pycache__/ artifacts.  This test ensures the
directory is gone and cannot accidentally reappear.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_src_directory_does_not_exist():
    """The stale src/ directory (orphaned __pycache__ only) must be removed."""
    src = REPO_ROOT / "src"
    assert not src.exists(), (
        "src/ directory still exists with only stale __pycache__ artifacts. "
        "Remove with: rm -rf src/"
    )


def test_gitignore_covers_pycache():
    """Acceptance: .gitignore must cover __pycache__/ at repo root."""
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "__pycache__/" in gitignore or "__pycache__" in gitignore
