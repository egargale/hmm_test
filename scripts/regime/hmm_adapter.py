import numpy as np
import pandas as pd

from data_processing.feature_engineering import add_features
from hmm_models.gaussian_hmm import GaussianHMMModel
from hmm_models.gmm_hmm import GMMHMMModel
from regime.markov_chain import compute_stationary_distribution


def run_hmm_regime(
    prices: pd.DataFrame,
    n_states: int = 3,
    model_type: str = "gaussian",
    covariance_type: str = "full",
    n_iter: int = 100,
) -> dict:
    """
    Run HMM regime detection and return regime skill JSON contract.

    Returns {"available": True, ...} on success,
    {"available": False, "reason": "..."} on failure.
    """
    try:
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name="close")
            prices["open"] = prices["close"]
            prices["high"] = prices["close"]
            prices["low"] = prices["close"]
            prices["volume"] = 0

        df = add_features(prices)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ohlcv = {"open", "high", "low", "close", "volume"}
        feature_cols = [c for c in numeric_cols if c not in ohlcv]
        if not feature_cols:
            return {"available": False, "reason": "No numeric features after engineering"}
        df_clean = df[feature_cols].dropna()
        if len(df_clean) < n_states + 1:
            return {"available": False, "reason": f"Not enough clean rows ({len(df_clean)})"}

        if model_type == "gmm":
            model = GMMHMMModel(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
            )
        else:
            model = GaussianHMMModel(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
            )

        model.fit(df_clean)
        states = model.predict(df_clean)
        transmat = model.get_transition_matrix()
        stationary = compute_stationary_distribution(transmat)

        means = model.get_model_parameters()["means_"]
        if "log_ret" in [c.lower() for c in df_clean.columns]:
            lr_idx = [i for i, c in enumerate(df_clean.columns) if c.lower() == "log_ret"][0]
            state_means = means[:, lr_idx]
        else:
            state_means = means[:, 0]

        order = np.argsort(state_means)
        state_labels = {int(order[0]): "bear", int(order[1]): "sideways", int(order[2]): "bull"}

        regimes = []
        for s in states:
            regimes.append(state_labels.get(int(s), "sideways"))

        labeled_transmat = transmat[np.ix_(order, order)]

        return {
            "available": True,
            "regimes": regimes,
            "transition_matrix": labeled_transmat.tolist(),
            "stationary_distribution": {
                "bear": float(stationary[order[0]]),
                "sideways": float(stationary[order[1]]),
                "bull": float(stationary[order[2]]),
            },
            "caveat": "HMM state labels are inferred from ascending mean return; labels may swap on re-fit",
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}
