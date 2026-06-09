"""Tests for build_label_map — HMM latent state → regime bucket mapping.

Covers the deduplicated label-map construction extracted from
_classify_hmm_slice and _remap_to_prev_states.
"""

import numpy as np
import pytest

from hmm_futures_analysis.regime.engines._state_mapping import build_label_map


class TestBuildLabelMap:
    """Map HMM latent states to regime buckets (Bear=0, Sideways=1, Bull=2)."""

    def test_single_state_maps_to_sideways(self):
        means = np.array([[5.0]])
        lm = build_label_map(means, sort_col=0)
        assert lm == {0: 1}

    def test_two_states_map_to_bear_bull(self):
        means = np.array([[-1.0], [1.0]])
        lm = build_label_map(means, sort_col=0)
        assert lm == {0: 0, 1: 2}

    def test_two_states_ascending_order(self):
        means = np.array([[3.0], [-2.0]])
        lm = build_label_map(means, sort_col=0)
        # sorted ascending: state 1 (mean=-2) → bear(0), state 0 (mean=3) → bull(2)
        assert lm == {1: 0, 0: 2}

    def test_three_states_identity(self):
        means = np.array([[1.0], [2.0], [3.0]])
        lm = build_label_map(means, sort_col=0)
        assert lm == {0: 0, 1: 1, 2: 2}

    def test_three_states_unsorted(self):
        means = np.array([[3.0], [1.0], [2.0]])
        lm = build_label_map(means, sort_col=0)
        # sorted ascending: state 1(1.0)→0, state 2(2.0)→1, state 0(3.0)→2
        assert lm == {0: 2, 1: 0, 2: 1}

    def test_five_states_collapses_to_3_buckets(self):
        means = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        lm = build_label_map(means, sort_col=0)
        # 5 states: i*3//5 → 0,0,1,1,2
        assert lm == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}

    def test_four_states_collapses_to_3_buckets(self):
        means = np.array([[0.0], [1.0], [2.0], [3.0]])
        lm = build_label_map(means, sort_col=0)
        # 4 states: i*3//4 → 0,0,1,2
        assert lm == {0: 0, 1: 0, 2: 1, 3: 2}

    def test_multifeature_sorts_by_sort_col(self):
        means = np.array([[99.0, -1.0], [99.0, 1.0]])
        lm = build_label_map(means, sort_col=1)
        # sorted by col 1: state 0(mean=-1) → bear(0), state 1(mean=1) → bull(2)
        assert lm == {0: 0, 1: 2}

    def test_multifeature_sort_col_0(self):
        means = np.array([[-1.0, 50.0], [1.0, -50.0]])
        lm = build_label_map(means, sort_col=0)
        assert lm == {0: 0, 1: 2}

    def test_equal_means_single(self):
        means = np.array([[2.0], [2.0]])
        lm = build_label_map(means, sort_col=0)
        # Both states have equal means — argsort is stable but order is arbitrary.
        # Result should still be {state: bucket} with buckets {0, 2}.
        assert set(lm.values()) == {0, 2}
        assert len(lm) == 2
