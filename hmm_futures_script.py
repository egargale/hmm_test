#!/usr/bin/env python
# hmm_futures_script.py
"""
Train / deploy a 3-state Hidden Markov Model on a huge CSV file
with futures OHLCV data.  Generates a regime signal and
runs a simple back-test (long state 0, short state 2, flat state 1).

Examples
--------
# Fit on 5-minute ES data and save the model
python hmm_futures_script.py  data/es_continuous_5min.csv  --symbol ES --model-out model.pkl

# Reload model and test on other months
python hmm_futures_script.py  data/es_2024Q3.csv --symbol ES --model-path model.pkl

All I/O is memory-efficient; even 1 GB+ files are chunked through Dask &
pandas.read_csv(..., dtype=...) using 16-byte floats where possible.
"""

import argparse
import os
import warnings

import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# --- Feature Engineering ----------------------------------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal set of features: log-returns + rolling volatility ATM."""
    df = df.copy()
    # Basic sanity checks ------------------------------------------------------------
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise KeyError(f"Column {col} missing from CSV")
    # compute log returns ---------------------------------------------------------
    df["log_ret"] = np.log(df["Close"]).diff()
    # 20-period ATR-style volatility proxy
    df["range"] = ((df["High"] - df["Low"]) * 0.5).rolling(20).mean()
    # nan clean-up
    df.dropna(inplace=True)
    return df[["log_ret", "range"]]  # <- observations sent to HMM


# --- Model Training ---------------------------------------------------------
def train_model(X: np.ndarray, n_states: int = 3, seed: int = 42):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=seed,
        verbose=0,
    )
    model.fit(X)
    return model


# --- Back-Test helper -------------------------------------------------------
def simple_backtest(df: pd.DataFrame, states: np.ndarray) -> pd.Series:
    """Dummy signal/state -> position map."""
    position = np.zeros(len(df))
    position[states == 0] = 1  # long low-vol up
    position[states == 2] = -1  # short high-vol down
    # vectorized pnl (log return * signed position)
    df = df.copy()
    df["next_ret"] = np.log(df["Close"]).diff().shift(-1)
    pnl = df["next_ret"] * position
    cum_pnl = pnl.dropna().cumsum()
    return cum_pnl


def perf_metrics(series: pd.Series):
    """Annualized Sharpe & max drawdown assuming intraday data."""
    sharpe = series.diff().mean() / series.diff().std() * np.sqrt(252 * 78)
    drawdown = (series - series.cummax()).min()
    return sharpe, drawdown


# --- Chunked CSV reader (memory-mapped via Dask) ----------------------------
def load_big_csv(path: str, symbol: str = "ES", dtype_downcast=True):
    """Returns a pandas DataFrame with only the subset for the symbol."""
    # Dask auto-detects gzip, parquet, etc.
    ddf = dd.read_csv(path, parse_dates=["Datetime"])
    if dtype_downcast:
        ddf = ddf.astype(
            {
                "Open": np.float32,
                "High": np.float32,
                "Low": np.float32,
                "Close": np.float32,
                "Volume": np.float32,
            }
        )
    # Filter by symbol (optional)
    if "Symbol" in ddf.columns:
        ddf = ddf[ddf["Symbol"].str.strip() == symbol]
    else:
        ddf = ddf
    return ddf.compute()  # Bring to pandas (still chunked nicely)


# -------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--model-out", help="Path to serialize the fitted model")
    parser.add_argument("--model-path", help="Pre-trained HMM path → skip fitting")
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # --- Load raw data ---------------------------------------------------------
    print("Loading CSV …")
    df_big = load_big_csv(args.csv_file, args.symbol)

    # --- Make clean features ---------------------------------------------------
    feats = make_features(df_big)
    X = feats.values

    # --- Train or load model ---------------------------------------------------
    if args.model_path and os.path.exists(args.model_path):
        print("Loading pre-trained HMM …")
        model = joblib.load(args.model_path)
    else:
        print("Fitting Gaussian HMM …")
        model = train_model(X, n_states=args.n_states, seed=args.seed)
        if args.model_out:
            joblib.dump(model, args.model_out)
            print("Model saved →", args.model_out)

    # --- decode (Viterbi) ------------------------------------------------------
    states = model.predict(X)

    # --- simple back-test ------------------------------------------------------
    df_big = df_big.iloc[len(df_big) - len(states) :]  # align index
    cum_pnl = simple_backtest(df_big, states)
    sharpe, max_dd = perf_metrics(cum_pnl)
    print("\n===== Back-test Summary =====")
    print("Total bars :", len(df_big))
    print(
        "States:",
        [f"{k} → mean {X[states == k].mean(0)}" for k in range(args.n_states)],
    )
    print("Final Equity:", f"{cum_pnl.iloc[-1]:.4f}")
    print("Sharpe (est):", f"{sharpe:.2f}")
    print("Max drawdown:", f"{max_dd:.4f}")

    # Quick plot ---------------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    cum_pnl.plot(title="Regime-Switching P&L")
    plt.show()


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
