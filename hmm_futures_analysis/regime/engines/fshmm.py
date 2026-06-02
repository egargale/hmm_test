"""Feature Saliency HMM regime classification engine."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm

from ..engine_protocol import ClassifyOutput, ClassifyResult
from ._hmm_engine import _classify_hmm_slice, engineer_features


class FSHMMEngine:
    """Regime engine using Feature Saliency HMM (Adams et al. 2016).

    Learns per-feature saliency weights rho_k during EM training.
    Features with rho_k < saliency_threshold are masked as irrelevant.
    """

    def __init__(
        self,
        n_states: int = 3,
        pca_variance: float | None = None,
        saliency_threshold: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.pca_variance = pca_variance
        self.saliency_threshold = saliency_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None:
        if data is None:
            raise ValueError("FSHMMEngine requires OHLCV data for feature engineering")
        return engineer_features(data, use_messina=False)

    def enrich_info(self, info: dict) -> dict:
        result = {**info}
        result["caveat"] = "HMM states sorted by mean return; labels may swap on re-fit"
        if hasattr(self, "_last_saliency") and self._last_saliency is not None:
            result["feature_saliency"] = self._last_saliency
            result["selected_features"] = self._last_selected_features
        return result

    def run_classify(
        self,
        prices: pd.Series,
        ohlcv: pd.DataFrame | None,
        returns: pd.Series,
        min_train: int,
        **kwargs,
    ) -> ClassifyOutput:
        from ._hmm_pipeline import _hmm_classify_pipeline

        return _hmm_classify_pipeline(
            self, prices, ohlcv, returns, min_train, **kwargs
        )

    def classify(
        self, data: pd.DataFrame, prev_means: np.ndarray | None = None
    ) -> ClassifyResult:
        features_clean = data.bfill().dropna()
        if len(features_clean) < self.n_states + 1:
            raise ValueError(
                f"Not enough clean rows ({len(features_clean)}) "
                f"for {self.n_states} HMM states"
            )
        feature_names = list(features_clean.columns)
        X = features_clean.to_numpy(dtype=np.float64)

        # Z-score normalisation
        center = np.mean(X, axis=0)
        scale = np.std(X, axis=0) + 1e-8
        X_norm = ((X - center) / scale).astype(np.float64)

        # Optional PCA whitening (ADR-0005)
        pca_transform = None
        if self.pca_variance is not None:
            from sklearn.decomposition import PCA

            pca_transform = PCA(
                n_components=self.pca_variance,
                svd_solver="full",
                random_state=self.random_state,
            )
            X_norm = pca_transform.fit_transform(X_norm).astype(np.float64)

        # Run Feature Saliency EM
        model, rho = self._fit_fshmm(X_norm)

        # Last bar (already normalized)
        X_last = X_norm[-1:]

        # Delegate regime-mapping to shared pipeline
        result = _classify_hmm_slice(
            model,
            X_last,
            self.n_states,
            prev_means,
        )

        # Attach saliency metadata
        selected_features = [
            name for name, r in zip(feature_names, rho) if r >= self.saliency_threshold
        ]

        result.feature_saliency = rho
        result.selected_features = selected_features or None

        # Store on instance for pipeline access
        self._last_saliency = rho.tolist() if rho is not None else None
        self._last_selected_features = selected_features

        return result

    # ------------------------------------------------------------------
    # Core FSHMM EM
    # ------------------------------------------------------------------

    def _fit_fshmm(self, X: np.ndarray) -> tuple[hmm.GaussianHMM, np.ndarray]:
        """Run Feature Saliency EM on z-scored (and optionally PCA'd) data.

        Returns (trained_hmm_model, rho_array).
        """
        T, D = X.shape
        K = self.n_states

        # --- Initialise base HMM for starting mu, sigma2, pi, A ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = hmm.GaussianHMM(
                n_components=K,
                covariance_type="diag",
                n_iter=30,
                tol=1e-4,
                random_state=self.random_state,
                params="stmc",
                init_params="stmc",
            )
            base_model.fit(X)

        # Pull initial parameters from base model
        mu = base_model.means_.copy()  # (K, D)
        sigma2 = base_model._covars_.copy()  # (K, D) diagonal
        pi = base_model.startprob_.copy()  # (K,)
        A = base_model.transmat_.copy()  # (K, K)

        # Saliency parameters
        rho = np.full(D, 0.5)
        epsilon = np.mean(X, axis=0)  # background mean
        tau2 = np.var(X, axis=0) + 1e-8  # background variance

        plateau_count = 0
        prev_ll = 0.0

        for _iteration in range(self.max_iter):
            # --- Standard E-step: forward-backward ---
            # Compute log-likelihood under current (mu, sigma2)
            log_signal = _log_gaussian_diag(X, mu, sigma2)  # (T, K)
            log_pi = np.log(np.clip(pi, 1e-300, None))
            log_A = np.log(np.clip(A, 1e-300, None))

            # Forward pass
            log_alpha = np.empty((T, K))
            log_alpha[0] = log_pi + log_signal[0]
            for t in range(1, T):
                log_alpha[t] = (
                    _logsumexp_row(log_alpha[t - 1] + log_A.T) + log_signal[t]
                )

            # Backward pass
            log_beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                log_beta[t] = _logsumexp_row(
                    log_A + log_signal[t + 1] + log_beta[t + 1]
                )

            # Log-likelihood for plateau early-exit (before gamma exp)
            log_lik = float(_logsumexp_row(log_alpha[-1]))

            # Gamma (posteriors)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_row(log_gamma)[:, np.newaxis]
            gamma = np.exp(log_gamma)  # (T, K)

            # --- Saliency E-step (vectorized) ---
            p_signal = _gaussian_pdf(
                X[:, np.newaxis, :], mu[np.newaxis, :, :], sigma2[np.newaxis, :, :]
            )  # (T, K, D)
            p_bg = _gaussian_pdf(X, epsilon, tau2)  # (T, D)

            rho_clamp = np.clip(rho, 1e-10, 1 - 1e-10)

            num_signal = rho_clamp[np.newaxis, np.newaxis, :] * p_signal
            num_bg = (1 - rho_clamp)[np.newaxis, np.newaxis, :] * p_bg[:, np.newaxis, :]
            denom = num_signal + num_bg + 1e-300

            u = gamma[:, :, np.newaxis] * num_signal / denom  # (T, K, D)
            v = gamma[:, :, np.newaxis] - u  # (T, K, D)

            # --- M-step ---

            # Update mu[i, l]
            u_sum_t = np.sum(u, axis=0)  # (K, D)
            u_xt = np.einsum("tkd,td->kd", u, X)  # (K, D)
            mask = u_sum_t > 1e-10
            mu = np.where(mask, u_xt / np.maximum(u_sum_t, 1e-10), mu)

            # Update sigma2[i, l]
            sq_diff = (X[:, np.newaxis, :] - mu[np.newaxis, :, :]) ** 2
            u_sq = np.einsum("tkd,tkd->kd", u, sq_diff)
            sigma2 = np.where(mask, u_sq / np.maximum(u_sum_t, 1e-10), sigma2)
            sigma2 = np.maximum(sigma2, 1e-8)

            # Update epsilon[l]
            v_sum_i = np.sum(v, axis=1)  # (T, D)
            v_sum_it = np.sum(v_sum_i, axis=0)  # (D,)
            epsilon = np.sum(v_sum_i * X, axis=0) / np.maximum(v_sum_it, 1e-10)

            # Update tau2[l]
            sq_diff_eps = (X - epsilon) ** 2
            tau2 = np.sum(v_sum_i * sq_diff_eps, axis=0) / np.maximum(v_sum_it, 1e-10)
            tau2 = np.maximum(tau2, 1e-8)

            # Update rho[l] — MAP closed-form (Adams 2016)
            u_sum = np.sum(u, axis=(0, 1))  # (D,)
            k_param = 1.0
            T_hat = T + 1 + k_param
            discriminant = np.maximum(T_hat**2 - 4 * k_param * u_sum, 0)
            rho_new = (T_hat - np.sqrt(discriminant)) / (2 * k_param)
            rho_new = np.clip(rho_new, 1e-10, 1 - 1e-10)

            # Update pi analytically from gamma
            pi = gamma[0] / np.sum(gamma[0])

            # Vectorised transition M-step uses updated mu, sigma2
            log_signal_upd = _log_gaussian_diag(X, mu, sigma2)
            signal_exp = np.exp(log_signal_upd[1:])
            xi_all = np.einsum("ti,tj,ij->tij", gamma[:-1], signal_exp, A)
            xi_all /= np.maximum(xi_all.sum(axis=2, keepdims=True), 1e-300)
            xi_sum = xi_all.sum(axis=0)
            A = xi_sum / np.maximum(np.sum(xi_sum, axis=1, keepdims=True), 1e-300)
            # Ensure valid transition matrix
            A = np.clip(A, 1e-10, None)
            A /= A.sum(axis=1, keepdims=True)

            # Convergence checks
            delta = np.max(np.abs(rho_new - rho))
            rho = rho_new

            # Plateau early-exit: bail if log-likelihood flat for 3 iterations
            if _iteration > 2:
                delta_ll = abs(log_lik - prev_ll)
                if delta_ll < 1e-5 * max(abs(prev_ll), 1.0):
                    plateau_count += 1
                    if plateau_count >= 3:
                        break
                else:
                    plateau_count = 0
            prev_ll = log_lik

            if delta < self.tol:
                break

        # Build final model with converged parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model = hmm.GaussianHMM(
                n_components=K,
                covariance_type="diag",
                n_iter=1,
                random_state=self.random_state,
                params="",
                init_params="",
            )
            # Fit once to initialise internal state, then overwrite
            final_model.fit(X[: K + 1])
            final_model.means_ = mu
            final_model._covars_ = sigma2
            final_model.startprob_ = pi
            final_model.transmat_ = A

        return final_model, rho


# ------------------------------------------------------------------
# Vectorised helpers
# ------------------------------------------------------------------


def _gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Vectorised Gaussian PDF (diagonal covariance)."""
    return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * (x - mu) ** 2 / sigma2)


def _log_gaussian_diag(X: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Log-likelihood of X under diagonal Gaussian (per-state).

    X: (T, D), mu: (K, D), sigma2: (K, D) → (T, K)
    """
    n_features = mu.shape[1]
    return (
        -0.5 * n_features * np.log(2 * np.pi)
        - 0.5 * np.sum(np.log(sigma2), axis=1)
        - 0.5
        * np.sum(
            (X[:, np.newaxis, :] - mu[np.newaxis, :, :]) ** 2
            / sigma2[np.newaxis, :, :],
            axis=2,
        )
    )


def _logsumexp_row(a: np.ndarray) -> np.ndarray:
    """Numerically stable logsumexp along last axis."""
    a_max = np.max(a, axis=-1, keepdims=True)
    return np.log(np.sum(np.exp(a - a_max), axis=-1)) + a_max.squeeze(-1)
