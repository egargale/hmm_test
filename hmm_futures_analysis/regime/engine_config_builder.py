"""Map argparse Namespace → engine config dataclass.

Extracted from cli.py so that the config-building adapter can be
tested independently and reused outside the CLI.
"""

from __future__ import annotations

import argparse

from .engine_configs import (
    FSHMMConfig,
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)


def build_engine_config(args: argparse.Namespace) -> object:
    """Construct the correct config dataclass from CLI args.

    Only passes ``pca_variance`` when the user explicitly provides it
    via ``--pca-variance``.  Otherwise the engine's dataclass default
    wins (messina=0.95, robust_hmm=0.90, hmm/fshmm=None).
    """
    engine = args.engine
    pca_kw = {}
    if args.pca_variance is not None:
        pca_kw["pca_variance"] = args.pca_variance

    if engine == "threshold":
        return ThresholdConfig(
            window=args.window,
            threshold=args.threshold,
        )
    elif engine == "hmm":
        return HMMGenericConfig(
            n_states=args.n_states,
            **pca_kw,
        )
    elif engine == "messina":
        return HMMMMessinaConfig(
            n_states=args.n_states,
            **pca_kw,
        )
    elif engine == "robust_hmm":
        return RobustHMMConfig(
            n_states=args.n_states,
            **pca_kw,
            robust_method=args.robust_method,
        )
    elif engine == "fshmm":
        return FSHMMConfig(
            n_states=args.n_states,
            **pca_kw,
            saliency_threshold=args.saliency_threshold,
        )
    else:
        raise ValueError(f"Unknown engine: {engine!r}")
