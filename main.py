"""
hmm_futures.py
Train and use a Hidden Markov Model on HUGE futures CSV files.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import ROCIndicator
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

###############################################################################
# Feature engineering
###############################################################################
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log-returns, ATR, and ROC to the dataframe.
    Returns a *new* dataframe with NaNs removed.
    """
    df = df.copy()

    # 1. Log returns
    df["log_ret"] = np.log(df["Close"]).diff()

    # 2. ATR (window 14)
    atr = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["atr"] = atr.average_true_range()

    # 3. Rate-of-change (momentum, window 5)
    roc = ROCIndicator(close=df["Close"], window=5)
    df["roc"] = roc.roc()

    # Drop NA rows caused by indicators
    df = df.dropna().reset_index(drop=True)
    return df





###############################################################################
# Main pipeline
###############################################################################
def main(args):
    ###########################################################################
    # 1. Validate arguments
    ###########################################################################
    if args.n_states < 1:
        raise ValueError("Number of states must be positive")
    if args.max_iter < 1:
        raise ValueError("Max iterations must be positive")
    if args.chunksize < 1:
        raise ValueError("Chunk size must be positive")
        
    ###########################################################################
    # 2. Load / build features
    ###########################################################################
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    feat_df = stream_features(csv_path, chunksize=args.chunksize)

    # Drop rows if any NaNs left
    feat_df = feat_df.dropna()

    # Validate that we have enough data
    if len(feat_df) == 0:
        raise ValueError("No data remaining after feature engineering")
    if len(feat_df) < args.n_states:
        raise ValueError(f"Insufficient data ({len(feat_df)} rows) for {args.n_states} states")

    feature_cols = ["log_ret", "atr", "roc"]
    X = feat_df[feature_cols].values

    ###########################################################################
    # 2. Scale features (important!) or load pre-trained scaler
    ###########################################################################
    if args.model_path and os.path.exists(args.model_path):
        try:
            logging.info("Loading pre-trained model and scaler ...")
            with open(args.model_path, 'rb') as f:
                saved_data = pickle.load(f)
            model = saved_data['model']
            scaler = saved_data['scaler']
            X_scaled = scaler.transform(X)
        except Exception as e:
            logging.error("Failed to load model: %s", str(e))
            raise
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ###########################################################################
        # 3. Train Gaussian HMM or load pre-trained model
        ###########################################################################
        logging.info("Fitting %d-state Gaussian HMM ...", args.n_states)
        model = GaussianHMM(
            n_components=args.n_states,
            covariance_type="diag",
            n_iter=args.max_iter,
            random_state=42,
            verbose=True,
        )
        try:
            model.fit(X_scaled)
        except Exception as e:
            logging.error("Failed to train HMM: %s", str(e))
            raise

        logging.info("Model converged: %s", model.monitor_.converged)
        logging.info("Log-likelihood: %.2f", model.score(X_scaled))

        # Save model and scaler if requested
        if args.model_out:
            try:
                logging.info("Saving model and scaler to %s ...", args.model_out)
                with open(args.model_out, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'scaler': scaler
                    }, f)
            except Exception as e:
                logging.error("Failed to save model: %s", str(e))
                raise

    ###########################################################################
    # 4. Decode hidden states
    ###########################################################################
    states = model.predict(X_scaled)
    
    # Apply position shifting to prevent lookahead bias if requested
    if args.prevent_lookahead:
        states = np.roll(states, 1)
        states[0] = states[1]  # Fill first value

    feat_df["state"] = states

    ###########################################################################
    # 5. Save results
    ###########################################################################
    out_path = csv_path.with_suffix(".hmm_states.csv")
    feat_df.to_csv(out_path)
    logging.info("States saved to %s", out_path)

    ###########################################################################
    # 6. Backtesting if requested
    ###########################################################################
    if args.backtest:
        logging.info("Running backtest ...")
        backtest_results = simple_backtest(feat_df, states)
        sharpe, max_dd = perf_metrics(backtest_results)
        logging.info("Backtest Results:")
        logging.info("  Final Equity: %.4f", backtest_results.iloc[-1])
        logging.info("  Sharpe Ratio: %.2f", sharpe)
        logging.info("  Max Drawdown: %.4f", max_dd)
        
        # Save backtest results
        try:
            backtest_path = csv_path.with_suffix(".backtest.csv")
            backtest_results.to_csv(backtest_path)
            logging.info("Backtest results saved to %s", backtest_path)
        except Exception as e:
            logging.error("Failed to save backtest results: %s", str(e))
            raise

    ###########################################################################
    # 7. Optional: quick sanity plot
    ###########################################################################
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 4))
            plt.plot(feat_df.index, feat_df["Close"], label="Close")
            for s in range(args.n_states):
                mask = feat_df["state"] == s
                plt.scatter(
                    feat_df.index[mask],
                    feat_df["Close"][mask],
                    label=f"State {s}",
                    s=5,
                )
            plt.legend()
            plt.title("Futures Close Price & HMM States")
            plt.tight_layout()
            plot_path = csv_path.with_suffix(".png")
            plt.savefig(plot_path)
            logging.info("Plot saved to %s", plot_path)
        except ImportError:
            logging.warning("matplotlib not found – skipping plot.")
        except Exception as e:
            logging.error("Failed to generate plot: %s", str(e))
            raise


###############################################################################
# Backtesting
###############################################################################
def simple_backtest(df: pd.DataFrame, states: np.ndarray) -> pd.Series:
    """Simple backtest using state-based positions."""
    position = np.zeros(len(df))
    position[states == 0] = 1    # long low-vol up
    position[states == 2] = -1   # short high-vol down
    # vectorized pnl (log return * signed position)
    df = df.copy()
    df['next_ret'] = np.log(df['Close']).diff().shift(-1)
    pnl = df['next_ret'] * position
    cum_pnl = pnl.dropna().cumsum()
    return cum_pnl

def perf_metrics(series: pd.Series):
    """Annualized Sharpe & max drawdown assuming intraday data."""
    sharpe = series.diff().mean() / series.diff().std() * np.sqrt(252 * 78)
    drawdown = (series - series.cummax()).min()
    return sharpe, drawdown

###############################################################################
# Streaming reader for huge files
###############################################################################
def stream_features(
    csv_path: Path,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Lazily reads CSV in chunks, adds features, concatenates.
    """
    logging.info("Streaming CSV and computing features ...")
    
    # Validate CSV has required columns
    try:
        first_chunk = pd.read_csv(csv_path, nrows=1)
        required_cols = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in first_chunk.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    except Exception as e:
        logging.error("Failed to validate CSV format: %s", str(e))
        raise
    
    reader = pd.read_csv(
        csv_path,
        parse_dates=["DateTime"],
        index_col="DateTime",
        chunksize=chunksize,
    )
    frames = []
    for chunk in tqdm(reader, desc="Processing chunks"):
        try:
            # Downcast dtypes for memory efficiency
            chunk = chunk.astype({
                "Open": np.float32,
                "High": np.float32,
                "Low": np.float32,
                "Close": np.float32,
                "Volume": np.float32
            })
            frames.append(add_features(chunk))
        except Exception as e:
            logging.error("Failed to process chunk: %s", str(e))
            raise

    full = pd.concat(frames, ignore_index=False).sort_index()
    logging.info("Features computed. Total rows: %d", len(full))
    return full

###############################################################################
# CLI
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HMM on huge futures CSV"
    )
    parser.add_argument("csv", help="Path to futures OHLCV CSV")
    parser.add_argument(
        "-n",
        "--n_states",
        type=int,
        default=3,
        help="Number of hidden states (default 3)",
    )
    parser.add_argument(
        "-i",
        "--max_iter",
        type=int,
        default=100,
        help="Max EM iterations (default 100)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Save quick sanity plot",
    )
    parser.add_argument(
        "--model-path",
        help="Path to pre-trained model and scaler",
    )
    parser.add_argument(
        "--model-out",
        help="Path to save trained model and scaler",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for reading CSV (default 100000)",
    )
    parser.add_argument(
        "--prevent-lookahead",
        action="store_true",
        help="Prevent lookahead bias by shifting positions",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run simple backtest after training",
    )
    args = parser.parse_args()
    main(args)


