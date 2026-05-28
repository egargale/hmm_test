# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
This is a Hidden Markov Model (HMM) futures trading analysis project. It provides an Agent Skill that detects market regimes (Bull/Bear/Sideways) using three independent engines: threshold (fast, close-only), messina (HMM + 19 Messina features), and hmm (HMM + ~50 generic features). Features include BIC-based state count selection (`--n-states auto`), optional PCA whitening, and walk-forward whipsaw filters (`--dwell-bars`, `--hysteresis`).

## Agent skills

### Issue tracker

Issues tracked in GitHub (`egargale/hmm_test`). See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles using default label names. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.
