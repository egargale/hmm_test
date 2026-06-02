# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
This is a Hidden Markov Model (HMM) futures trading analysis project. It provides an Agent Skill that detects market regimes (Bull/Bear/Sideways) using five independent engines: threshold (fast, close-only), messina (HMM + 19 Messina features), hmm (HMM + ~50 generic features), robust_hmm (HMM + outlier-resistant emissions), and fshmm (Feature Saliency HMM). Features include BIC-based state count selection (`--n-states auto`), optional PCA whitening, walk-forward whipsaw filters (`--dwell-bars`, `--hysteresis`), duration forecasting via survival analysis, and per-phase timing instrumentation.

## Agent skills

### Issue tracker

Issues tracked in GitHub (`egargale/hmm_test`). See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles using default label names. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.

### Research archive

Technology landscape scans and literature surveys live in `docs/research/`. Key file:
- `docs/research/technology-scan-2026-05.md` — prioritised scan of bleeding-edge regime detection tech (what was evaluated, what was selected, what was deprioritised and why)
