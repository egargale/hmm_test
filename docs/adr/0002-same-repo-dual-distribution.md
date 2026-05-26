# Same repo, dual distribution

The hmm_test repo ships as both a pip-installable Python library and an
installable Claude Code skill (via `npx skills add`). One `SKILL.md` at the
repo root, one codebase under `hmm_futures_analysis/`, self-bootstrapping
`run.sh` for skill consumers.

## Considered options

### A) Separate repos for skill and library

One repo for the Python library (pip-installable), another for the
SKILL.md + instructions. Each repo optimized for its consumer.

**Rejected because**: the SKILL.md references executable Python code —
`scripts/cli.py`, `references/`, `test_data/`. A separate skill repo would
either duplicate or submodule the library, creating a sync burden. Two repos
also doubles CI, issue tracking, and release overhead for a small project.

### B) Skill-only (no pip install)

Drop the library distribution entirely. Consumers either clone the repo or
install via `npx skills add`. No `pyproject.toml`, no console scripts.

**Rejected because**: non-Claude users (data scientists, researchers, other
Python projects) lose the ability to `pip install` and `import
hmm_futures_analysis`. The code is general-purpose regime detection — it
shouldn't be locked behind a single agent ecosystem.

### C) Same repo, dual distribution (chosen)

One repo contains both the `SKILL.md` (for `npx skills add`) and a
fully-formed Python package with `pyproject.toml`, `console_scripts`, and
installable dependencies. A self-bootstrapping `run.sh` handles venv creation
for skill consumers. Library consumers install via
`pip install git+https://github.com/egargale/hmm_test.git`.

**Chosen because**: (1) single source of truth — one codebase, one set of
tests, one issue tracker; (2) `npx skills add egargale/hmm_test` copies the
whole repo into `.claude/skills/hmm-regime-detection/` and it works; (3)
`pip install` gives library consumers a `hmm-regime` CLI and importable
`hmm_futures_analysis` package; (4) no publishing to PyPI needed — GitHub
install is sufficient for the current audience.

## Consequences

- **Package namespace is `hmm_futures_analysis/`.** Code lives under
  `hmm_futures_analysis/` (renamed from `scripts/`) so that pip-installed
  imports are `from hmm_futures_analysis.regime.pipeline import run` instead
  of the generic `scripts.*`.
- **`run.sh` at the repo root.** Skill consumers invoke this instead of
  `python scripts/cli.py` directly. It auto-creates a venv, installs
  dependencies (including the `yfinance` extra), and delegates to
  `hmm_futures_analysis/cli.py`. Library consumers never see it — they use
  the `hmm-regime` console script.
- **SKILL.md uses `$SKILL_DIR` as a path variable.** The skill's install
  location varies (global vs project, different agent directories). The
  SKILL.md doesn't hardcode paths — Claude resolves `$SKILL_DIR` from the
  filesystem at runtime.
- **GitHub-only distribution.** No PyPI release workflow. Library consumers
  use `pip install git+https://github.com/egargale/hmm_test.git`. Can be
  added later without restructuring.
- **yfinance stays optional in `pyproject.toml`.** The `[yfinance]` extra is
  optional for library consumers who don't need `--ticker`. But `run.sh`
  installs it by default (`pip install ".[yfinance]"`) so skill consumers
  never hit a missing dependency.
- **Dev-only files ship in the skill.** `tests/`, `AGENTS.md`, `CONTEXT.md`,
  `PLAN.md`, `docs/agents/` all get copied into `.claude/skills/` by
  `npx skills add`. They're inert — Claude only reads files that SKILL.md
  references — but they take up space. Acceptable for now.
