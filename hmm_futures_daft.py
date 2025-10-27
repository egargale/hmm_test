#!/usr/bin/env python3
"""
End-to-end HMM + back-test on futures OHLCV using **Daft**.
"""
import argparse
import os
import warnings
from pathlib import Path

import daft as daf
import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# 1. Data ingestion via Daft (out-of-core, lazy scan)
# ------------------------------------------------------------------
def load_daft(path: str, symbol: str | None) -> daf.DataFrame:
    """Return Daft DataFrame filtered to symbol; always cast numeric cols."""
    dtypes = {
        "Datetime": "datetime[ns]",
        "Open":     "float32",
        "High":     "float32",
        "Low":      "float32",
        "Close":    "float32",
        "Volume":   "float32",
    }
    # Daft reads huge CSV lazily via Arrow → very memory friendly
    df = daf.read_csv(Path(path).expanduser(), dtype=dtype_fix_dict(dtypes))

    # case-insensitive symbol column if provided
    if symbol and "Symbol" in df.schema().column_names():
        df = df.where(daf.col("Symbol").str.upper() == symbol.upper())
    return df


def dtype_fix_dict(d):
    """Daft dtype helper."""
    return {c: daf.DataType.from_pandas_dtype(tp) for c, tp in d.items()}


# ------------------------------------------------------------------
# 2. Minimal feature engineering directly in Daft
# ------------------------------------------------------------------
def make_features_daft(df: daf.DataFrame) -> daf.DataFrame:
    """In-Daft vectorised features: log-returns & rolling ATR-proxy."""
    df = df.with_column("log_ret", daf.log(daf.col("Close")).diff())
    # 20-period EWMA of intraday range as cheap vol proxy
    df = df.with_column("range", 0.5 * (daf.col("High") - daf.col("Low")))
    df = df.with_column("vol", daf.col("range").ewm(span=20).mean())
    df = df.drop_columns(["range"])
    return df.drop_nulls()[["log_ret", "vol"]].collect()  # → Pandas for HMM


# ------------------------------------------------------------------
# 3. HMM utilities
# ------------------------------------------------------------------
def train_hmm(mat: np.ndarray, k: int, seed: int):
    model = GaussianHMM(n_components=k,
                        covariance_type="diag",
                        n_iter=1000,
                        random_state=seed,
                        verbose=False)
    model.fit(mat)
    return model


# ------------------------------------------------------------------
# 4. Naïve strategy + Back-test utilities
# ------------------------------------------------------------------
def regime_positions(states: np.ndarray) -> np.ndarray:
    """Long state 0, short state 2, flat state 1."""
    pos = np.zeros(len(states))
    pos[states == 0] = 1.0
    pos[states == 2] = -1.0
    return pos


def compute_pnl(price: np.ndarray, positions: np.ndarray) -> np.ndarray:
    # Shift positions by 1 to avoid lookahead bias. The position at time `t`
    # is based on features at `t`, so we can only trade on the return from
    # `t` to `t+1`.
    positions = np.roll(positions, 1)
    positions[0] = 0.0  # No position at the very start

    logret = np.concatenate([[0.0], np.log(price[1:] / price[:-1])])
    # PnL is based on the previous period's position
    return np.cumsum(logret * positions)


def perf(stats: np.ndarray):
    rets = np.diff(stats)
    sharpe = rets.mean() / rets.std() * np.sqrt(252 * 78)  # intraday factor
    dd = np.min(stats - np.maximum.accumulate(stats))
    return sharpe, dd


# ------------------------------------------------------------------
# 5. Argparse CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HMM + back-test – Daft Edition")
    parser.add_argument("csv_file")
    parser.add_argument("--symbol", default=None,
                        help="Filter to that symbol (case-insensitive)")
    parser.add_argument("--model-out", help="Save trained HMM to file")
    parser.add_argument("--model-path", help="Pre-trained model, skip training")
    parser.add_argument("--states", type=int, default=3, help="Number HMM states")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    print("⚙️  Scanning CSV with Daft …")

    # Load CSV via Daft — stays Arrow backed until finally collected
    daf_df = load_daft(args.csv_file, args.symbol)

    # Materialise features into RAM-converted Pandas for hmmlearn
    features_df = make_features_daft(daf_df)
    X = features_df.values

    # Align close prices with features (some rows are dropped by feature engineering)
    close_series = daf_df.select(daf.col("Close")).collect()["Close"].values
    close_series = close_series[-len(X):]

    # Train or load model & scaler ---------------------------------------------
    if args.model_path and os.path.exists(args.model_path):
        bundle = joblib.load(args.model_path)
        model = bundle['model']
        scaler = bundle['scaler']
        print("Loaded existing HMM and Scaler.")
        # Scale features with the loaded scaler
        X_scaled = scaler.transform(X)
    else:
        print("Training new HMM …")
        # 1. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Train HMM on scaled features
        model = train_hmm(X_scaled, k=args.states, seed=args.seed)
        if args.model_out:
            bundle = {'model': model, 'scaler': scaler}
            joblib.dump(bundle, args.model_out)
            print(f"Model and scaler saved -> {args.model_out}")

    # Viterbi decode ------------------------------------------------------------
    states = model.predict(X_scaled)

    # Back-test on daily data ---------------------------------------------------
    positions = regime_positions(states)
    equity_curve = compute_pnl(close_series, positions)

    sharpe, dd = perf(equity_curve)
    print("\n==== RESULTS ====")
    print("Total bars        :", len(close_series))
    print("Final log-equity  :", f"{equity_curve[-1]:.4f}")
    print("Sharpe            :", f"{sharpe:.2f}")
    print("Max draw-down     :", f"{dd:.4f}")

    # Simple matplotlib plot
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve, label="HMM-based equity")
    plt.title("Regime-switching back-test – Daft pipeline")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
