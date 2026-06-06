> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: AST-backed except where explicitly marked inferred

# HMM Regime Detection — Knowledge Base Index

## What This Is

A Hidden Markov Model futures trading analysis project. Detects market regimes (Bull/Bear/Sideways) using five independent engines with walk-forward backtesting, duration forecasting via survival analysis, and per-phase timing instrumentation. Ships as an **Agent Skill** for LLM coding agents.

## Architecture — 8 Systems + 1 Module

| System | Code Path | Hotness | Lines |
|--------|-----------|---------|-------|
| **CLI Entrypoint** | `hmm_futures_analysis/cli.py` | 🔴 high (25 changes) | ~628 |
| **Pipeline & Orchestration** | `hmm_futures_analysis/regime/` (excl. engines/) | 🔴 high (35 changes) | ~1200 |
| **Engine Implementations (5)** | `hmm_futures_analysis/regime/engines/` | 🟡 medium (7-13 ea) | 7 files, ~1700 |
| **Data Processing** | `hmm_futures_analysis/data_processing/` | normal | 6 files, ~1600 |
| ↳ **Ticker Disk Cache** | `data_processing/ticker_cache.py` | normal (1 commit) | 96 |
| **Backtesting & Evaluation** | `hmm_futures_analysis/backtesting/`, `eval.py` | normal | 3 files, ~200 |
| **Utilities** | `hmm_futures_analysis/utils/` | normal | 3 files, 61 |
| **Sweep Scripts** | `scripts/` | normal | 13 files, ~3500 |
| **Test Suite** | `tests/` | normal | 55+ files, ~11500 |

## Key Findings

- **Central hub**: `pipeline.py` has 32 downstream consumers (highest fan-in). It orchestrates everything.
- **Tightest coupling**: `pipeline.py` ↔ `walk_forward.py` at 1.00 (perfect co-change). These are deeply interwoven.
- **Engine isolation works**: None of the 5 engines import pipeline.py — they only know the protocol.
- **New**: `ticker_cache.py` (96 lines) provides on-disk yfinance caching with `--cache-dir`, `--refresh`, `--no-cache` CLI flags. Tested by 201-line test suite.
- **Bash (2 files)**: parsed with module-only coverage.

## Domain Language

Engine · Walk-forward backtest · Regime · HMM latent state · Signal · Transition matrix · OHLCV · Feature engineering · Discrete trade · Degenerate auto-recovery · BIC state selection · PCA whitening · Dwell-time filter · Hysteresis filter · Posteriors · Robust HMM · FSHMM · HMMEngineBase

See `.nexus-map/concepts/domains.md` for full definitions.

## [操作指南] 强制执行步骤

> 本节是对所有读取本文件的 AI 发出的硬性操作指令，不是建议。

### 步骤1 — 必须先读完以下所有文件（顺序不限）

读完本文件后，在执行任何任务之前，必须依次 read 以下文件完整内容：

- `.nexus-map/arch/systems.md` — 系统边界与代码位置
- `.nexus-map/arch/dependencies.md` — 系统间依赖关系与 Mermaid 图
- `.nexus-map/arch/test_coverage.md` — 测试面与证据缺口
- `.nexus-map/hotspots/git_forensics.md` — Git 热点与耦合风险
- `.nexus-map/concepts/domains.md` — 核心领域概念

> 这些文件均为高密度摘要，总量通常 < 5000 tokens，是必要的上下文成本。
> 不得以"任务简单"或"只改一个文件"为由跳过。

### 步骤2 — 按任务类型追加操作（步骤1 完成后执行）

- 若任务涉及**接口修改、新增跨模块调用、删除/重命名公共函数**：
  → 必须运行 `query_graph.py --impact <目标文件>` 确认影响半径后再写代码。
- 若任务需要**判断某文件被谁引用**：
  → 运行 `query_graph.py --who-imports <模块名>`。
- 若仓库结构已发生重大变化（新增系统、重构模块边界）：
  → 任务完成后评估是否需要重新运行 nexus-mapper 更新知识库。
