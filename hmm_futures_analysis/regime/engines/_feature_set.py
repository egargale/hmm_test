"""Feature sets — the single source of truth for an engine's feature mode.

Each HMM-family engine holds a :class:`FeatureSet` value that owns three
facts which were previously encoded three different ways:

* the **label** (the ``config.features`` value),
* the **builder** (generic :func:`add_features` or messina
  :func:`add_messina_features`), and
* the resulting **column count**.

Before this module, the engine's ``use_messina`` boolean, the config's
``features`` string, and the pipeline's ``_count_features`` lookup table
were three parallel encodings of one fact that had to be kept in sync by
hand.  ``FeatureSet`` collapses them into one object.

The threshold engine opts out — it is close-only, its ``precompute``
returns ``None``, and the low-data warning special-cases it to count=1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ...data_processing.feature_engineering import add_features
from ...data_processing.messina_features import (
    MESSINA_FEATURE_COLUMNS,
    add_messina_features,
)

# OHLCV columns produced by add_features that are prices, not features.
_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


@runtime_checkable
class FeatureSet(Protocol):
    """A bundle of label + builder + count for one feature-engineering mode.

    The label is the ``config.features`` value; the count is read off the
    built frame (never a magic number).
    """

    #: The feature-engineering mode label (``"generic"``, ``"messina"``).
    label: str

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer the feature columns from OHLCV data.

        Returns a frame containing feature columns only (no OHLCV prices).
        Raises ``ValueError`` if no numeric features survive.
        """
        ...

    @property
    def count(self) -> int:
        """Number of feature columns this set produces.

        For messina, a fixed declared count (``len(MESSINA_FEATURE_COLUMNS)``);
        for generic, derived from the built columns.
        """
        ...


class GenericFeatureSet:
    """Generic ~50-indicator feature set (SMA-based).

    Used by the ``hmm``, ``robust_hmm``, and ``fshmm`` engines.
    """

    label = "generic"

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        df = add_features(data, min_periods=10)
        df = df.dropna(axis=1, how="all")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric_cols if c not in _OHLCV_COLUMNS]

        if not cols:
            raise ValueError("No numeric features after engineering")

        return df[cols]

    @property
    def count(self) -> int:
        # Generic count is data-dependent (columns survive dropna), so it is
        # read off the built frame by callers via ClassifyOutput.n_features
        # rather than declared here.  This property returns the canonical
        # ~50 figure for the degenerate-fit heuristic when no frame is on
        # hand; the pipeline prefers the built count when available.
        return 50


class MessinaFeatureSet:
    """The 19 Messina indicators (Wilder's smoothing, VSTOP, ADX/DI, …).

    Used by the ``messina`` engine.  The count is the declared length of
    :data:`MESSINA_FEATURE_COLUMNS` — the single source of truth that the
    builder and the count both read.
    """

    label = "messina"

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        df = add_messina_features(data)
        cols = [c for c in MESSINA_FEATURE_COLUMNS if c in df.columns]

        if not cols:
            raise ValueError("No numeric features after engineering")

        return df[cols]

    @property
    def count(self) -> int:
        return len(MESSINA_FEATURE_COLUMNS)
