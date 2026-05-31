"""Tests for feature_engineering module — public interface stability.

These tests verify that the only public entry point (add_features) works
correctly and that dead code (FeatureEngineer class) has been removed.
"""

import importlib

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Test 1: add_features is importable and functional
# ---------------------------------------------------------------------------


class TestAddFeaturesPublicAPI:
    """Verify add_features — the sole public entry point — works end-to-end."""

    @staticmethod
    def _make_price_df(n_rows: int = 200) -> pd.DataFrame:
        """Build a minimal OHLCV DataFrame that add_features accepts."""
        dates = pd.bdate_range("2024-01-01", periods=n_rows, freq="B")
        rng = np.random.default_rng(42)
        close = 100.0 + rng.standard_normal(n_rows).cumsum()
        return pd.DataFrame(
            {
                "date": dates,
                "open": close + rng.standard_normal(n_rows) * 0.5,
                "high": close + np.abs(rng.standard_normal(n_rows)),
                "low": close - np.abs(rng.standard_normal(n_rows)),
                "close": close,
                "volume": rng.integers(1_000, 100_000, n_rows).astype(float),
            }
        )

    def test_add_features_returns_dataframe(self):
        """add_features(df) returns a DataFrame with more columns than input."""
        from hmm_futures_analysis.data_processing.feature_engineering import (
            add_features,
        )

        df = self._make_price_df()
        result = add_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > len(df.columns)

    def test_add_features_preserves_original_columns(self):
        """add_features must not drop any of the input columns."""
        from hmm_futures_analysis.data_processing.feature_engineering import (
            add_features,
        )

        df = self._make_price_df()
        result = add_features(df)
        for col in df.columns:
            assert col in result.columns, f"Missing original column: {col}"

    def test_add_features_importable_from_package(self):
        """add_features is re-exported via __init__.py."""
        from hmm_futures_analysis.data_processing import add_features

        assert callable(add_features)


# ---------------------------------------------------------------------------
# Test 2: FeatureEngineer class must NOT exist
# ---------------------------------------------------------------------------


class TestDeadCodeRemoval:
    """Verify that the unused FeatureEngineer class has been fully removed."""

    def test_feature_engineer_class_does_not_exist(self):
        """FeatureEngineer must not be defined in the module."""
        mod = importlib.import_module(
            "hmm_futures_analysis.data_processing.feature_engineering"
        )
        assert not hasattr(mod, "FeatureEngineer"), (
            "FeatureEngineer class still exists — it should have been deleted"
        )

    def test_no_feature_selection_imports(self):
        """No lazy imports from feature_selection should remain."""
        import inspect

        from hmm_futures_analysis.data_processing import feature_engineering as fe_mod

        source = inspect.getsource(fe_mod)
        assert "feature_selection" not in source, (
            "feature_selection import still present in feature_engineering.py"
        )
