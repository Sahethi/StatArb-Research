"""
HMM Regime Detection for Statistical Arbitrage.

Fits a 2-state Gaussian HMM on two features derived from the cross-sectional
distribution of idiosyncratic residuals:

    Feature 1: rolling mean of daily cross-sectional std of residuals
               (level of idiosyncratic volatility in the market)
    Feature 2: rolling std of daily cross-sectional std of residuals
               (instability / volatility-of-volatility)

Features are standardized using training-window statistics before fitting,
which prevents covariance degeneracy in the EM algorithm.

The two states correspond to:
    - Favorable (mean-reverting): low and stable cross-sectional vol
    - Unfavorable (trending/crisis): high or erratic cross-sectional vol

After fitting on a training window, regime probabilities are inferred using
the causal forward algorithm (no lookahead into future observations).

At each trading date, if P(favorable | data up to t) < entry_threshold,
the engine skips opening new positions.
"""
import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    2-state HMM regime filter for the stat-arb engine.

    Args:
        n_states: Number of hidden states (default 2: favorable / unfavorable).
        training_window: Number of leading days used to fit the HMM.
            The rest of the history uses causal forward inference only.
        feature_window: Rolling window (days) for computing HMM features.
        random_state: Seed for reproducible HMM fitting.
    """

    def __init__(
        self,
        n_states: int = 2,
        training_window: int = 252,
        feature_window: int = 20,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.training_window = training_window
        self.feature_window = feature_window
        self.random_state = random_state
        self.model: GaussianHMM | None = None
        self.favorable_state: int = 0
        self._is_fitted: bool = False
        # Training-window statistics for feature standardization
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def build_features(self, residuals: pd.DataFrame) -> np.ndarray:
        """
        Compute causal features from daily idiosyncratic residuals.

        Args:
            residuals: DataFrame of shape (dates, tickers) with daily
                idiosyncratic returns from the factor model.

        Returns:
            Array of shape (T, 2). Rows with insufficient history are NaN.
        """
        w = self.feature_window
        cross_vol = residuals.std(axis=1)  # daily cross-sectional std

        vol_level = cross_vol.rolling(w, min_periods=w).mean()
        vol_of_vol = cross_vol.rolling(w, min_periods=w).std()

        return np.column_stack([vol_level.values, vol_of_vol.values])

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        """Apply training-window mean/std scaling."""
        if self._feat_mean is None or self._feat_std is None:
            raise RuntimeError("Scaler not fitted — call fit() first.")
        return (features - self._feat_mean) / self._feat_std

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "HMMRegimeDetector":
        """
        Fit HMM parameters on the training portion of features.

        Only the first `training_window` rows are used for fitting.
        Features are standardized using training-window statistics to
        prevent covariance degeneracy in the EM algorithm.

        Args:
            features: Array of shape (T, n_features) from build_features().

        Returns:
            self
        """
        train = features[: self.training_window]
        valid_mask = ~np.isnan(train).any(axis=1)
        valid = train[valid_mask]

        if len(valid) < 2 * self.n_states:
            raise ValueError(
                f"Only {len(valid)} valid training rows after dropping NaN — "
                "need at least 2 × n_states. Increase training_window or "
                "reduce feature_window."
            )

        # Standardize features using training statistics
        self._feat_mean = valid.mean(axis=0)
        self._feat_std = valid.std(axis=0)
        self._feat_std[self._feat_std < 1e-10] = 1.0  # avoid division by zero
        valid_scaled = (valid - self._feat_mean) / self._feat_std

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=300,
            min_covar=0.01,   # prevents covariance collapse
            random_state=self.random_state,
        )
        # hmmlearn prints "Model is not converging" directly to sys.stderr
        # when the log-likelihood delta is tiny but technically negative
        # (numerical noise near a saddle point).  This is harmless, so
        # redirect stderr during fitting to keep output clean.
        with contextlib.redirect_stderr(io.StringIO()):
            self.model.fit(valid_scaled)

        # Identify the favorable state as the one with lower mean vol level
        # (Feature 0 is vol level: lower = calmer = more mean-reverting)
        means_vol = self.model.means_[:, 0]
        self.favorable_state = int(np.argmin(means_vol))

        logger.info(
            "HMM fitted on %d observations. Favorable state: %d. "
            "State means (vol level): %s",
            len(valid),
            self.favorable_state,
            means_vol,
        )
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Causal inference (forward algorithm, no lookahead)
    # ------------------------------------------------------------------

    def predict_proba_causal(self, features: np.ndarray) -> np.ndarray:
        """
        Compute P(state_t | obs_1, ..., obs_t) for every t using the
        forward algorithm.

        This is strictly causal: the probability at t depends only on
        observations up to and including t. NaN feature rows carry
        forward the previous valid probability.

        Features are standardized with the training-window scaler before
        computing emission probabilities.

        Args:
            features: Array of shape (T, n_features).

        Returns:
            Array of shape (T, n_states) with state probabilities.
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Call fit() before predict_proba_causal().")

        features_scaled = self._standardize(features)
        T = len(features_scaled)
        n = self.n_states

        log_start = np.log(self.model.startprob_ + 1e-300)
        log_trans = np.log(self.model.transmat_ + 1e-300)
        log_emission = self.model._compute_log_likelihood(features_scaled)

        log_alpha = np.full((T, n), -np.inf)
        proba = np.full((T, n), np.nan)
        last_valid_t: int | None = None

        for t in range(T):
            if np.isnan(features[t]).any():
                # No valid observation: carry forward last state estimate
                if last_valid_t is not None:
                    proba[t] = proba[last_valid_t]
                continue

            if last_valid_t is None:
                # First valid observation: initialise with start probs
                log_alpha[t] = log_start + log_emission[t]
            else:
                # Standard forward step: marginalise over previous state
                for j in range(n):
                    log_alpha[t, j] = (
                        np.logaddexp.reduce(
                            log_alpha[last_valid_t] + log_trans[:, j]
                        )
                        + log_emission[t, j]
                    )

            # Normalise at each step to prevent underflow
            log_sum = np.logaddexp.reduce(log_alpha[t])
            log_alpha[t] -= log_sum
            proba[t] = np.exp(log_alpha[t])
            last_valid_t = t

        return proba

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    def get_favorable_proba(self, proba: np.ndarray) -> np.ndarray:
        """
        Extract P(favorable state | observations up to t) for each day.

        Args:
            proba: Output from predict_proba_causal(), shape (T, n_states).

        Returns:
            1-D array of shape (T,) with favorable-regime probabilities.
        """
        return proba[:, self.favorable_state]
