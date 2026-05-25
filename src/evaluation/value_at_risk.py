"""
Value-at-Risk forecasting and back-testing.

Replicates Christensen, Siggaard, Veliyev (2023) Section 5: an economic
application of the volatility forecasts. For each test day t we build a
one-day-ahead α-quantile forecast of the log-return distribution using
filtered historical simulation (FHS; see Barone-Adesi, Bourgoin &
Giannopoulos 1998 and Barone-Adesi, Giannopoulos & Vosper 1999):

1. Standardise in-sample log-returns by the model's volatility forecast:
   ``z_s = r_s / sqrt(RV_hat_s)`` for s ≤ T.
2. The empirical α-quantile of the standardised residuals ``q_α`` is the
   percentile.
3. Forecast VaR: ``VaR_t = q_α · sqrt(RV_hat_t)``.

The forecast is evaluated using:

* **Asymmetric quantile loss** (Koenker & Bassett 1978):
  ``L = (α - 1{r ≤ VaR}) · (r - VaR)``.

* **Kupiec unconditional coverage test** (1995): likelihood-ratio test
  that the realised hit frequency equals α.

* **Christoffersen conditional coverage test** (1998): joint test of
  hit-frequency AND hit-independence (no clustering).

All tests are returned with their LR statistic and p-value.

References:
    Koenker, R., and G. Bassett (1978). "Regression Quantiles." *Econometrica*.
    Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk
        Measurement Models." *Journal of Derivatives*.
    Christoffersen, P. (1998). "Evaluating Interval Forecasts."
        *International Economic Review*.
    Barone-Adesi, G., F. Bourgoin, and K. Giannopoulos (1998).
        "Don't Look Back." *Risk*, 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class VaRResult:
    var_forecast: pd.Series             # negative numbers (VaR is a return floor)
    hits: pd.Series                     # 1 if r_t ≤ VaR_t else 0
    alpha: float
    quantile_loss: float
    kupiec_lr: float
    kupiec_pvalue: float
    christoffersen_lr: float
    christoffersen_pvalue: float
    expected_hits: float
    observed_hits: int
    n: int


def _quantile_loss(r: np.ndarray, var: np.ndarray, alpha: float) -> float:
    """Asymmetric pinball loss (Koenker-Bassett 1978).

    .. math::
        L = (\\alpha - \\mathbb{1}\\{r \\le \\mathrm{VaR}\\})(r - \\mathrm{VaR})
    """
    hits = (r <= var).astype(float)
    return float(np.mean((alpha - hits) * (r - var)))


def _kupiec_unconditional(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) likelihood-ratio test for unconditional coverage.

    Tests ``H0: P(hit) = α`` against ``H1: P(hit) ≠ α``. Returns ``(LR, p)``.
    LR is distributed χ²(1) under the null.
    """
    n = len(hits)
    x = int(hits.sum())
    if n == 0:
        return float("nan"), float("nan")
    pi_hat = x / n
    if pi_hat == 0 or pi_hat == 1:
        # likelihood degenerate; use a tiny floor for stability
        pi_hat = max(min(pi_hat, 1 - 1e-12), 1e-12)
    # Log-likelihoods
    ll_null = x * np.log(alpha) + (n - x) * np.log(1 - alpha) if 0 < alpha < 1 else 0.0
    ll_alt = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
    lr = -2.0 * (ll_null - ll_alt)
    pval = float(1 - stats.chi2.cdf(lr, df=1)) if np.isfinite(lr) else float("nan")
    return float(lr), pval


def _christoffersen_conditional(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen (1998) joint coverage + independence test.

    The joint LR test is the sum of the Kupiec unconditional LR and the
    independence LR (Markov-chain transitions). Distributed χ²(2) under null.
    """
    n = len(hits)
    if n < 2:
        return float("nan"), float("nan")
    # Transition counts
    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / (n - 1) if (n - 1) > 0 else 0.0
    # Independence log-likelihoods
    def _logl(p, k_hit, k_nohit):
        if p in (0.0, 1.0):
            return 0.0
        return k_hit * np.log(p) + k_nohit * np.log(1 - p)
    ll_indep = _logl(pi, n01 + n11, n00 + n10)
    ll_dep = _logl(pi01, n01, n00) + _logl(pi11, n11, n10)
    lr_indep = -2.0 * (ll_indep - ll_dep)
    lr_unc, _ = _kupiec_unconditional(hits, alpha)
    lr_joint = lr_unc + lr_indep
    pval = float(1 - stats.chi2.cdf(lr_joint, df=2)) if np.isfinite(lr_joint) else float("nan")
    return float(lr_joint), pval


def filtered_historical_simulation(
    log_returns: pd.Series,
    rv_forecasts: pd.Series,
    train_end: pd.Timestamp,
    alpha: float = 0.05,
    rv_realised: pd.Series | None = None,
) -> VaRResult:
    """Filtered historical simulation VaR back-test.

    The FHS quantile is computed from in-sample standardised residuals
    ``z_s = r_s / σ̂_s``. The reference volatility ``σ̂_s`` in-sample is the
    *realised* RV — supplied via ``rv_realised`` — because our models only
    forecast the out-of-sample test period. This is the standard choice
    when only an out-of-sample forecast series is available
    (see Audrino, Huang & Okhrin 2019).

    Out-of-sample, ``σ̂_t = sqrt(rv_forecasts[t])`` and
    ``VaR_t = q_α · σ̂_t`` where ``q_α`` is the in-sample α-quantile of z.

    Parameters
    ----------
    log_returns
        Daily log-return series indexed by date (full sample).
    rv_forecasts
        Daily one-day-ahead RV forecast for the test period.
    train_end
        Last date included in the in-sample window (used for standardisation).
    alpha
        VaR confidence level (e.g. 0.05 → 5% one-day VaR).
    rv_realised
        Realised RV (full-sample, used for in-sample standardisation). If
        ``None``, falls back to ``rv_forecasts`` (which only works when the
        forecast series spans the in-sample period too).
    """
    # In-sample standardisation: use realised RV when supplied.
    if rv_realised is not None:
        common_train = log_returns.index.intersection(rv_realised.index)
        common_train = common_train[common_train <= train_end]
        if len(common_train) < 50:
            raise ValueError("Insufficient training history for FHS quantile")
        r_train = log_returns.loc[common_train].to_numpy()
        sigma_train = np.sqrt(np.maximum(rv_realised.loc[common_train].to_numpy(), 1e-12))
        z_train = r_train / sigma_train
        q = float(np.quantile(z_train, alpha))
    else:
        common = log_returns.index.intersection(rv_forecasts.index)
        r = log_returns.loc[common]
        sigma_hat = np.sqrt(np.maximum(rv_forecasts.loc[common].to_numpy(), 1e-12))
        z = r.to_numpy() / sigma_hat
        train_mask = common <= train_end
        if train_mask.sum() < 50:
            raise ValueError("Insufficient training history for FHS quantile")
        q = float(np.quantile(z[train_mask], alpha))

    # Out-of-sample VaR.
    test_idx = log_returns.index.intersection(rv_forecasts.index)
    test_idx = test_idx[test_idx > train_end]
    if len(test_idx) == 0:
        raise ValueError("No out-of-sample period after train_end")
    sigma_test = np.sqrt(np.maximum(rv_forecasts.loc[test_idx].to_numpy(), 1e-12))
    var_t = q * sigma_test
    r_t = log_returns.loc[test_idx].to_numpy()
    hits = (r_t <= var_t).astype(int)

    qloss = _quantile_loss(r_t, var_t, alpha)
    lr_unc, p_unc = _kupiec_unconditional(hits, alpha)
    lr_joint, p_joint = _christoffersen_conditional(hits, alpha)

    return VaRResult(
        var_forecast=pd.Series(var_t, index=test_idx, name=f"VaR_{int(alpha*100):02d}"),
        hits=pd.Series(hits, index=test_idx, name="hit"),
        alpha=alpha,
        quantile_loss=qloss,
        kupiec_lr=lr_unc,
        kupiec_pvalue=p_unc,
        christoffersen_lr=lr_joint,
        christoffersen_pvalue=p_joint,
        expected_hits=float(alpha * len(test_idx)),
        observed_hits=int(hits.sum()),
        n=int(len(test_idx)),
    )


def var_table(
    log_returns: pd.Series,
    rv_forecasts_by_model: dict[str, pd.Series],
    train_end: pd.Timestamp,
    alphas: tuple[float, ...] = (0.05, 0.01),
    rv_realised: pd.Series | None = None,
) -> pd.DataFrame:
    """Run FHS-VaR for many models and stack the diagnostics into one table."""
    rows = []
    for alpha in alphas:
        for label, rv in rv_forecasts_by_model.items():
            try:
                res = filtered_historical_simulation(log_returns, rv, train_end, alpha,
                                                       rv_realised=rv_realised)
                rows.append({
                    "alpha": alpha,
                    "model": label,
                    "n": res.n,
                    "expected_hits": res.expected_hits,
                    "observed_hits": res.observed_hits,
                    "hit_rate": res.observed_hits / res.n if res.n > 0 else float("nan"),
                    "quantile_loss": res.quantile_loss,
                    "kupiec_LR": res.kupiec_lr,
                    "kupiec_p": res.kupiec_pvalue,
                    "christoffersen_LR": res.christoffersen_lr,
                    "christoffersen_p": res.christoffersen_pvalue,
                })
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def expanding_quantile_fhs(
    log_returns: pd.Series,
    rv_forecasts: pd.Series,
    train_end: pd.Timestamp,
    alpha: float = 0.05,
    rv_realised: pd.Series | None = None,
    recalibrate_every: int = 1,
) -> VaRResult:
    """FHS-VaR with an *expanding* standardised-residual quantile.

    Unlike :func:`filtered_historical_simulation`, which freezes the α-quantile
    ``q_α`` on the in-sample (2016–2019) window, this recomputes ``q_α`` on an
    expanding window that grows by each *realised* test day, recalibrating every
    ``recalibrate_every`` test days. ``recalibrate_every=1`` (daily) satisfies
    the "at least annually" requirement; pass ``~252`` for annual recalibration.

    Standardisation is self-consistent with the VaR scaling: a test-day residual
    is ``z_t = r_t / sqrt(rv_forecasts[t])`` — the *same* volatility that scales
    ``VaR_t`` — so systematic forecast bias is absorbed into ``q_α`` and
    unconditional coverage becomes valid by construction as the window fills.
    The in-sample window seeds the pool with realised-RV-standardised residuals
    (no in-sample forecast is available; cf. the fixed-quantile FHS docstring).

    No look-ahead: ``VaR_t`` for test day ``t`` uses only residuals strictly
    before ``t`` (the seed plus realised test days ``< t``).
    """
    if rv_realised is None:
        rv_realised = rv_forecasts

    # In-sample seed residuals, standardised by realised RV.
    seed_idx = log_returns.index.intersection(rv_realised.index)
    seed_idx = seed_idx[seed_idx <= train_end]
    if len(seed_idx) < 50:
        raise ValueError("Insufficient training history for FHS quantile")
    sigma_seed = np.sqrt(np.maximum(rv_realised.loc[seed_idx].to_numpy(), 1e-12))
    z_seed = log_returns.loc[seed_idx].to_numpy() / sigma_seed

    # Out-of-sample test period, standardised by the forecast (self-consistent).
    test_idx = log_returns.index.intersection(rv_forecasts.index)
    test_idx = test_idx[test_idx > train_end]
    if len(test_idx) == 0:
        raise ValueError("No out-of-sample period after train_end")
    r_test = log_returns.loc[test_idx].to_numpy()
    sigma_test = np.sqrt(np.maximum(rv_forecasts.loc[test_idx].to_numpy(), 1e-12))
    z_test = r_test / sigma_test

    var_t = np.empty(len(test_idx))
    q = float(np.quantile(z_seed, alpha))
    for i in range(len(test_idx)):
        if i % recalibrate_every == 0:
            pool = np.concatenate([z_seed, z_test[:i]]) if i > 0 else z_seed
            q = float(np.quantile(pool, alpha))
        var_t[i] = q * sigma_test[i]

    hits = (r_test <= var_t).astype(int)
    qloss = _quantile_loss(r_test, var_t, alpha)
    lr_unc, p_unc = _kupiec_unconditional(hits, alpha)
    lr_joint, p_joint = _christoffersen_conditional(hits, alpha)

    return VaRResult(
        var_forecast=pd.Series(var_t, index=test_idx, name=f"VaR_{int(alpha*100):02d}"),
        hits=pd.Series(hits, index=test_idx, name="hit"),
        alpha=alpha,
        quantile_loss=qloss,
        kupiec_lr=lr_unc,
        kupiec_pvalue=p_unc,
        christoffersen_lr=lr_joint,
        christoffersen_pvalue=p_joint,
        expected_hits=float(alpha * len(test_idx)),
        observed_hits=int(hits.sum()),
        n=int(len(test_idx)),
    )
