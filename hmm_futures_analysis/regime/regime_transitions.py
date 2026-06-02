"""Extract historical regime transition events from a regime sequence.

Pure function — no I/O, no engine dependency, no side effects.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class TransitionEvent(NamedTuple):
    """A single regime transition event."""

    date: str  # ISO format
    from_regime: str  # e.g. "bear"
    to_regime: str  # e.g. "bull"
    bar_index: int


_STATE_NAMES = ("bear", "sideways", "bull")


def extract_transitions(
    regimes: np.ndarray,
    dates: pd.DatetimeIndex,
) -> list[TransitionEvent]:
    """Extract regime transition events from a classified regime sequence.

    Walks the regime array, compares adjacent pairs, and emits one
    TransitionEvent per change.  Returns an empty list for empty or
    single-element arrays, or when no regime changes occur.

    Parameters
    ----------
    regimes : np.ndarray
        1-D integer array of regime labels (0=bear, 1=sideways, 2=bull).
    dates : pd.DatetimeIndex
        Date index aligned with *regimes* (same length).

    Returns
    -------
    list[TransitionEvent]
        Chronological list of transition events.
    """
    if len(regimes) < 2:
        return []

    transitions: list[TransitionEvent] = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            transitions.append(
                TransitionEvent(
                    date=pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                    from_regime=_STATE_NAMES[int(regimes[i - 1])],
                    to_regime=_STATE_NAMES[int(regimes[i])],
                    bar_index=i,
                )
            )
    return transitions
