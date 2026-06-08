"""Tests for _remap_to_prev_states pure function.

Issue #43 — https://github.com/egargale/hmm_test/issues/43
"""

import numpy as np

from hmm_futures_analysis.regime.engines._hmm_engine import _remap_to_prev_states


class TestRemapIdentity:
    """When prev_means ≈ current means, raw_state maps to itself."""

    def test_3state_identity_remap(self):
        means = np.array([[1.0], [2.0], [3.0]])
        prev_means = np.array([[1.0], [2.0], [3.0]])

        assert _remap_to_prev_states(means, 0, prev_means) == 0
        assert _remap_to_prev_states(means, 1, prev_means) == 1
        assert _remap_to_prev_states(means, 2, prev_means) == 2


class TestRemapPermutation:
    """When prev_means are permuted, raw_state follows the swap."""

    def test_3state_swap_0_2(self):
        # Current: state 0=low(1.0), 1=mid(2.0), 2=high(3.0)
        means = np.array([[1.0], [2.0], [3.0]])
        # Previous: state 0=high(3.0), 1=low(1.0), 2=mid(2.0)
        # So prev_order by col0 ascending → [1, 2, 0] → ranks {1:0, 2:1, 0:2}
        prev_means = np.array([[3.0], [1.0], [2.0]])

        # raw_state 0 (current low) matches prev state 1 (low=1.0)
        # prev state 1 has rank 0 → result 0
        assert _remap_to_prev_states(means, 0, prev_means) == 0
        # raw_state 2 (current high) matches prev state 0 (high=3.0)
        # prev state 0 has rank 2 → result 2
        assert _remap_to_prev_states(means, 2, prev_means) == 2


class TestRemapCollapse:
    """When prev had >3 states, they collapse to 3 regimes."""

    def test_5state_collapse_low_to_bear(self):
        # 5 current + 5 prev states, both sorted ascending by col0.
        # prev_order = [0,1,2,3,4] (already ascending)
        # collapse i*3//5: {0:0, 1:0, 2:1, 3:1, 4:2}
        means = np.array([[-100.0], [-50.0], [0.0], [50.0], [100.0]])
        prev_means = np.array([[-100.0], [-50.0], [0.0], [50.0], [100.0]])

        # Identity assignment. raw_state 0 (lowest) → collapse[0]=0 (bear)
        assert _remap_to_prev_states(means, 0, prev_means) == 0
        # raw_state 4 (highest) → collapse[4]=2 (bull)
        assert _remap_to_prev_states(means, 4, prev_means) == 2

    def test_5state_collapse_high_to_bull(self):
        means = np.array([[-10.0], [0.0], [10.0]])
        prev_means = np.array([[10.0], [5.0], [0.0], [-5.0], [-10.0]])

        # raw_state 2 (mean=10) matches prev state 0 (mean=10) → collapse 2 (bull)
        assert _remap_to_prev_states(means, 2, prev_means) == 2

    def test_5state_collapse_mid_to_sideways(self):
        means = np.array([[-10.0], [0.0], [10.0]])
        prev_means = np.array([[10.0], [5.0], [0.0], [-5.0], [-10.0]])

        # raw_state 1 (mean=0) matches prev state 2 (mean=0) → collapse 1 (sideways)
        assert _remap_to_prev_states(means, 1, prev_means) == 1


class TestRemapFallback:
    """Unmatched raw_state returns the default regime."""

    def test_unmatched_returns_default(self):
        means = np.array([[1.0], [2.0], [3.0]])
        prev_means = np.array([[10.0], [20.0], [30.0]])

        # raw_state 99 has no match in assignment → returns default (0)
        assert _remap_to_prev_states(means, 99, prev_means) == 0

    def test_unmatched_custom_default(self):
        means = np.array([[1.0], [2.0], [3.0]])
        prev_means = np.array([[10.0], [20.0], [30.0]])

        assert _remap_to_prev_states(means, 99, prev_means, default=2) == 2


class TestRemapPCABoundsGuard:
    """Issue #107 — return_component exceeds PCA-reduced dimensionality."""

    def test_return_component_exceeds_prev_cols_does_not_crash(self):
        """return_component=9 but prev_means only has 5 columns (PCA reduced dims)."""
        # After PCA: 19 features → 5 components, but return_component was
        # computed in the original space as index 9.
        means = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        prev_means = np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])

        # Must not raise IndexError
        result = _remap_to_prev_states(means, 0, prev_means, return_component=9)
        assert isinstance(result, int)

    def test_return_component_exceeds_cols_falls_back_to_col0(self):
        """When return_component is out of bounds, sorting falls back to col 0."""
        # prev_means: state 0 has LOWEST col0 but HIGHEST col9 (if it existed)
        means = np.array([[1.0, 0.3], [2.0, 0.1], [3.0, 0.2]])
        prev_means = np.array([[1.0, 0.3], [2.0, 0.1], [3.0, 0.2]])

        # With return_component=9 (out of bounds), falls back to col 0.
        # Col 0 ascending: state 0=1.0 < state 1=2.0 < state 2=3.0
        # So raw_state=0 → prev rank 0 → regime 0 (bear)
        result = _remap_to_prev_states(means, 0, prev_means, return_component=9)
        assert result == 0
