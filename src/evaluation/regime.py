"""
Regime-conditional forecast evaluation.

Implements a 2-state Markov regime-switching model on the daily log-RV
series (Hamilton 1989) and uses the smoothed regime probabilities to
classify each test-set day as belonging to a *low-vol* or *high-vol*
regime. Forecast losses are then averaged conditionally on regime.

The implementation uses statsmodels'
``tsa.regime_switching.markov_regression.MarkovRegression`` with
switching mean and switching variance — the standard 2-state Gaussian
HMM for volatility-level regime detection (Ang & Bekaert 2002 use the
same framework for asset returns).

The conditioning question we ask is the one Christensen et al. (2023)
do not: does the ML-vs-HAR gap *depend on regime*? Two limit cases:

* If ML mostly captures conditional persistence that HAR misses, gain
  concentrated in the **high-vol regime** (regimes where the
  unconditional mean is far from the realised value).
* If ML's gain comes from feature signal at low vol levels (e.g. IV, M1W
  bringing real predictive content when RV is otherwise smooth), gain
  concentrated in the **low-vol regime**.

The empirical finding either way is reportable and useful for §4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


@dataclass(frozen=True)
class RegimeResult:
    smoothed_probs: pd.DataFrame   # columns: regime_0, regime_1
    regime: pd.Series              # 0 / 1 hard assignment via argmax
    means: tuple[float, float]
    variances: tuple[float, float]
    transition: np.ndarray         # 2x2 matrix


def fit_two_state_hmm(rv: pd.Series,
                      log_transform: bool = True,
                      n_iter: int = 200,
                      random_state: int = 42,
                      search_reps: int = 25) -> RegimeResult:
    """Fit a 2-state Markov regime-switching model with switching variance.

    Standard Hamilton (1989)-style switching regression on a log-vol series.
    The high-mean / high-variance state is interpreted as the "high-vol"
    regime. To avoid the well-known EM-divergence pathology with random
    start params, we (i) standardise the series before fitting and (ii)
    pass quantile-based start values plus ``search_reps`` random restarts.
    """
    rv_clean = rv.dropna()
    x_raw = np.log(np.maximum(rv_clean.to_numpy(), 1e-16)) if log_transform \
        else rv_clean.to_numpy()
    idx = rv_clean.index

    # Standardise to keep the EM in a well-conditioned region.
    x_mean = float(np.mean(x_raw))
    x_std = float(np.std(x_raw) or 1.0)
    x = (x_raw - x_mean) / x_std

    np.random.seed(random_state)
    model = MarkovRegression(
        x, k_regimes=2, trend="c", switching_variance=True,
    )

    # Quantile-based start: regime 0 = below-median (low vol),
    # regime 1 = above-median.  ``start_params`` ordering for this model is
    # [P_00, P_11, const[0], const[1], sigma2[0], sigma2[1]].
    lo_mask = x < np.median(x)
    mu0 = float(np.mean(x[lo_mask])) if lo_mask.any() else -0.5
    mu1 = float(np.mean(x[~lo_mask])) if (~lo_mask).any() else 0.5
    s0 = float(np.var(x[lo_mask]) or 0.25)
    s1 = float(np.var(x[~lo_mask]) or 1.0)
    start_params = np.array([0.95, 0.95, mu0, mu1, s0, s1])

    try:
        res = model.fit(
            start_params=start_params,
            maxiter=n_iter,
            disp=False,
            search_reps=search_reps,
        )
    except (np.linalg.LinAlgError, ValueError):
        # Fall back to no extra restarts if the search itself failed.
        res = model.fit(
            start_params=start_params,
            maxiter=n_iter,
            disp=False,
        )

    # smoothed posterior probabilities of each regime for each obs
    smoothed = res.smoothed_marginal_probabilities
    if isinstance(smoothed, pd.DataFrame):
        sm_df = smoothed.copy()
    else:
        sm_df = pd.DataFrame(np.asarray(smoothed), index=idx,
                              columns=[f"regime_{i}" for i in range(2)])
    sm_df.columns = ["regime_0", "regime_1"]
    sm_df.index = idx

    # Identify which regime is high-vol via mean of x conditional on regime.
    # ``res.params`` may be a pandas Series (labeled) or a numpy array,
    # depending on how the model was constructed; handle both.
    params_arr = np.asarray(res.params)
    # MarkovRegression with trend="c" and switching_variance places
    # parameters in the order: [P_00, P_11, const[0], const[1],
    # sigma2[0], sigma2[1]].
    means = params_arr[2:4]
    sigma2 = params_arr[4:6]
    if means[0] < means[1]:
        low_idx, high_idx = 0, 1
    else:
        low_idx, high_idx = 1, 0
        sm_df = sm_df.rename(columns={"regime_0": "_tmp"})
        sm_df = sm_df.rename(columns={"regime_1": "regime_0", "_tmp": "regime_1"})
    regime = (sm_df["regime_1"] >= 0.5).astype(int).rename("regime")

    variances_std = (
        float(sigma2[low_idx]),
        float(sigma2[high_idx]),
    )
    # Back-transform means and variances onto the original (log-RV) scale
    # so that downstream interpretation matches the data.
    mu_low_raw = float(means[low_idx]) * x_std + x_mean
    mu_high_raw = float(means[high_idx]) * x_std + x_mean
    var_low_raw = variances_std[0] * (x_std ** 2)
    var_high_raw = variances_std[1] * (x_std ** 2)
    P = res.regime_transition[..., 0]
    if low_idx == 1:
        P = P[::-1, ::-1]
    return RegimeResult(
        smoothed_probs=sm_df,
        regime=regime,
        means=(mu_low_raw, mu_high_raw),
        variances=(var_low_raw, var_high_raw),
        transition=np.asarray(P),
    )


def regime_conditional_losses(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    regime: pd.Series,
    loss_fn,
) -> pd.DataFrame:
    """Mean loss per (model, regime) for the test-set period.

    Returns a 2 × M DataFrame with rows {low_vol, high_vol} and columns
    per model.
    """
    common = y_true.index.intersection(regime.index)
    y = y_true.loc[common].to_numpy()
    reg = regime.loc[common].to_numpy()
    rows = {}
    for label in ("low_vol", "high_vol"):
        rows[label] = {}
    for m, p in predictions.items():
        idx = pd.Index(common).intersection(p.index)
        if len(idx) == 0:
            continue
        yp = p.loc[idx].to_numpy()
        ya = y_true.loc[idx].to_numpy()
        r = regime.loc[idx].to_numpy()
        l_low = loss_fn(ya[r == 0], yp[r == 0])
        l_high = loss_fn(ya[r == 1], yp[r == 1])
        rows["low_vol"][m] = float(np.mean(l_low)) if len(l_low) else float("nan")
        rows["high_vol"][m] = float(np.mean(l_high)) if len(l_high) else float("nan")
    return pd.DataFrame(rows).T
