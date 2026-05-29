# Technology Scan — Beyond Issue #20 HMM Engines

**Date**: 2026-05-29
**Context**: Review of bleeding-edge regime detection technologies to identify the next best additions beyond the 6 engines specified in Issue #20 (Wasserstein, Robust HMM, GH-HMM, FSHMM, Student-t, Ensemble).

## Search Coverage

16 queries across 4 research waves covering:
- Deep learning HMM hybrids (NRSM-MIA, Mixture-VAE)
- Online/streaming inference (Streaming HMM, online filters)
- Uncertainty quantification (conformal prediction, Bayesian HMM)
- Nonparametric regime discovery (HDP-HMM, Dirichlet processes)
- Regime duration forecasting (survival analysis, Weibull/Cox PH)
- Causal regime detection (CASTOR, FANTOM, structural causal models)
- Multi-timeframe fusion (wavelet decomposition, multi-scale ensembles)
- Synthetic data generation (diffusion models, CoFinDiff)
- Contrastive representation learning (DGRCL, LENS)
- Transfer/meta-learning for cross-asset regime detection

## Technology Landscape

### Selected for PRD Creation

| Technology | Breakthrough | Compatibility | Issue |
|---|---|---|---|
| **HDP-HMM** — nonparametric state discovery | Eliminates `--n-states` — model learns state count from data. Sticky variant prevents rapid switching. | Pure numpy/scipy, Gibbs sampler ~200 lines. Plugs into RegimeEngine protocol. | #27 |
| **Regime Duration Forecasting** — survival analysis | Adds temporal dimension: "how long will this regime last?" Answered via Weibull AFT / Cox PH models on historical regime spells. | Post-processing layer, engine-agnostic. Works with any engine's regime sequence. scipy (already present). | #28 |

### Evaluated and Deprioritized

| Technology | Why Deprioritized |
|---|---|
| **Conformal Prediction** (Adaptive Conformal Inference) | Valid prediction intervals without distributional assumptions. Deferred, not rejected — low implementation effort, wraps any engine. Revisit after core engine work. |
| **Online/Streaming HMM** (SHMM beam search) | Real-time inference with fixed O(K²) per observation. Requires new pipeline mode alongside existing batch walk-forward — architectural discussion needed first. |
| **Wavelet Multi-Timeframe Fusion** | Decomposes signals into frequency bands, detects regimes per band. New `pywt` dependency, fusion logic adds complexity. |
| **CASTOR Causal Regime Detection** | Learns causal DAG per regime. Answers "what caused the regime change?" Significant refactoring of feature engineering layer needed. Future exploration item. |
| **NRSM-MIA** (Neural Regime-Switching Model) | End-to-end differentiable HMM with Gumbel-Softmax + multi-head attention. Requires PyTorch ≥2.0, GPU for practical use. Paradigm shift from EM-based engine protocol. |
| **Mixture-VAE** | VAE with mixture latent space for regime detection. PyTorch dependency, less interpretable than standard HMM. |
| **Diffusion-based Synthetic Data** (CoFinDiff, Diff-Stega) | Controllable generation of regime-conditioned financial data. PyTorch, tangential to engine pipeline. |
| **Contrastive Learning** (DGRCL, LENS) | Regime-aware embeddings without labels. PyTorch, different paradigm from supervised regime mapping. |
| **Transfer/Meta-Learning** | Zero-shot regime detection across assets. Less applicable — project is single-asset focused. |

### Already in Issue #20 (Input to PRD Split)

| Engine | Key Innovation | Issue |
|---|---|---|
| `wasserstein` | 2-Wasserstein template matching for label stability | #21 |
| `robust_hmm` | Huber/MCD robust M-step for outlier-resistant parameters | #22 |
| `gh_hmm` | GH emissions + L1-penalized sparse precision matrices | #23 |
| `fshmm` | Feature saliency weights learned during EM | #24 |
| `ensemble` | Multi-engine voting wrapper | #25 |
| `student_t` | Heavy-tailed Student-t emissions | #26 |

## Implementation Sequence Rationale

1. **#22 Robust HMM** first — utilities in `_hmm_shared.py` are reused by other engines
2. **#21 Wasserstein** — most directly validated by paper (Sharpe 2.18 vs 1.18)
3. **#26 Student-t** — simpler custom BaseHMM subclassing, warm-up for GH
4. **#23 GH-HMM** — superset of Student-t, more complex M-step with GraphLasso
5. **#24 FSHMM** — saliency EM augmentation on existing HMM structure
6. **#25 Ensemble** — architectural wrapper, depends on engines being available
7. **#27 HDP-HMM** — nonparametric, eliminates state count entirely
8. **#28 Duration Forecasting** — post-processing, works with any engine

## Key References from Research

- **Fox, E.B. et al. (2010).** "A Sticky HDP-HMM." IEEE SPM 2010 / Annals of Applied Statistics. Foundational paper for nonparametric regime discovery.
- **Foroni, B., Merlo, L. & Petrella, L. (2024).** GH-HMM with penalized EM. arXiv:2412.03668.
- **Fons, E. et al. (2019/2021).** FSHMM for smart beta. arXiv:1902.10849 + GitHub code.
- **Boukardagha, A. (2026).** Wasserstein HMM for regime-aware investing. arXiv:2603.04441.
- **Küçükdağ, H.B. & Hekimoğlu, M. (2026).** Robust ridge-regularized EM for HMM. Sensors, 26(4). DOI: 10.3390/s26041321.
- **DS3M-ACI** (conformal prediction): https://arxiv.org/pdf/2512.03298
- **Streaming HMM**: https://github.com/gerdm/streaming-hmm
- **CASTOR**: Rahmani et al. (2025). Proceedings of MLR, 258.
- **Lunde & Timmermann (2004).** Duration dependence in stock prices. JBES, 22(3).

## Dependency Policy

All selected technologies use only packages already in `pyproject.toml`:
- **numpy**: all engines
- **scipy**: Bessel functions, matrix sqrt, linear sum assignment, Weibull fitting
- **scikit-learn**: MinCovDet, GraphicalLasso, PCA
- **hmmlearn**: BaseHMM subclassing

No PyTorch, JAX, statsmodels, or pywt required. This preserves the project's lightweight dependency profile.
