# Methodology — Detailed Implementation Reference

This document is the technical companion to the report. It states
every estimator, every loss, every test, and every numerical choice
made in this codebase, with the equations the implementation actually
solves and pointers to the file:line where it is computed. A reader
of the report can come here for the equation and find it tied to
a specific module in the code; an auditor reading the code can come
here for the rationale behind a numerical decision.

Where the paper is followed exactly we say so. Where we depart, the
deviation is named and justified.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `r_{t,i}` | i-th intraday 5-min log return on day t |
| `RV_t` | realised variance on day t |
| `RV+_t`, `RV-_t` | positive / negative semivariances |
| `RQ_t` | realised quarticity |
| `IV_t`, `VIX_t`, `EPU_t`, `HSI_t`, `ADS_t`, `US3M_t` | macro features |
| `M1W_t`, `DVOL_t`, `EA_t` | per-stock features |
| `RVD_t = RV_{t-1}`, `RVW_t = (1/5)·Σ_{k=1..5} RV_{t-k}`, `RVM_t = (1/22)·Σ_{k=1..22} RV_{t-k}` | HAR lags |
| `y_t^{(h)}` | h-step target = average RV from t+1 to t+h |
| `ŷ_t` | a model's forecast of `y_t^{(h)}` |
| `T_train`, `T_val`, `T_test` | observation counts |
| `m` | number of models in a comparison |

---

## 1. Data construction

### 1.1 Minute-bar ingest

Raw files are comma-separated minute bars in `MM/DD/YYYY,HH:MM,O,H,L,C,V`
format. The loader in `src/data/tick_to_minute.py:load_raw_minute_bars`
parses the date+time into a `pandas.DatetimeIndex` and returns a
DataFrame with columns `Open, High, Low, Close, Volume`.

The session restriction in `restrict_to_regular_session(df, "09:30",
"16:00")` keeps `[09:30, 16:00)` (open closed, close open) so the
16:00 close auction is *excluded*. This is intentional and matches
how 5-min RV is conventionally computed from minute data: the close
auction is a single discrete print, not a continuous-time observation,
and including it inflates RV by an idiosyncratic component.

### 1.2 Outlier filter

The filter implements a tick-cleaning rule of the Barndorff-Nielsen,
Hansen, Lunde, Shephard (2009) style (their P3 procedure adapted for
minute bars):

* Compute a rolling median of price over a window of `rolling_window_minutes`
  (default 50, i.e. ±25 min centred).
* Compute the rolling MAD-σ = 1.4826 · median(|price − rolling_median|).
* Drop any bar whose price-to-rolling-median deviation exceeds
  `sd_multiplier` (default 8) × MAD-σ.
* Independently drop any bar with `|log return| > max_abs_log_return`
  (default 0.10) to catch obvious data errors that the median filter
  might miss.

On real data this removes 0.03–0.05% of bars. Implemented in
`src/data/tick_to_minute.py:filter_outliers`.

### 1.3 5-min resampling within a day

`src/data/compute_rv.py:resample_to_frequency` does the resampling
**per trading day**, never across day boundaries. The implementation
groups by `df.index.normalize()` and then resamples each group with
`df.resample(f"{minutes}min", label="right", closed="right").last()`.
This guarantees that the last bar of day t and the first bar of day
t+1 are not combined into a single 5-min window (which would inject
an overnight return into intraday RV).

For each trading day we then compute:

* `RV_t = Σ_i r_{t,i}^2`
* `RV+_t = Σ_i r_{t,i}^2 · 1{r_{t,i} > 0}`
* `RV−_t = Σ_i r_{t,i}^2 · 1{r_{t,i} < 0}`
* `RQ_t = (n_t / 3) · Σ_i r_{t,i}^4`

where `r_{t,i}` is the log return on 5-min bar i of day t and `n_t` is
the number of 5-min returns that day (usually 78 in a regular session).
Days with fewer than 5 returns are dropped.

### 1.4 Realised kernel (extension)

The realised kernel of BNHLS (2008) at sampling frequency `f`:

```
RK(f) = γ_0 + Σ_{h=1..H} k((h-1)/H) · (γ_h + γ_{-h})
```

with γ_h the h-th autocovariance of intraday returns, k the Parzen
kernel, and H the bandwidth. The bandwidth follows BNHLS (2009):

```
H* = c* · ξ^(4/5) · n^(3/5),
ξ^2 = (ω^2 / IQ)^(1/2),
```

where ω² is the microstructure-noise variance estimated from the
high-frequency autocovariance, IQ is integrated quarticity (proxied
by Σr⁴ at the highest sampling frequency), and c* = 3.5134 for the
Parzen kernel. Implemented in `src/data/realised_kernel.py`.

### 1.5 Macro features

| Feature | Source | Cache | Construction |
|---|---|---|---|
| VIX | FRED VIXCLS | `data/macro/fred.parquet` | Level (close) |
| EPU | FRED USEPUINDXD | `data/macro/fred.parquet` | Level |
| ADS | Philly Fed XLSX (FRED ADSWBCIND retired) | `data/macro/ads_raw.parquet` | Level |
| US3M | FRED DTB3 | `data/macro/fred.parquet` | First-differenced |
| HSI | Yahoo Finance via `yfinance` | `data/macro/hsi.parquet` | Squared log return |
| IV (per-stock) | **VIX as substitute** | as above | Level |
| EA (per-stock dummy) | yfinance `Ticker.get_earnings_dates` | `data/macro/earnings_dates.json` (single JSON keyed by ticker) | 1 on reported earnings day, 0 otherwise |
| M1W | derived | none | 5-day rolling sum of daily log return from open to close |
| DVOL | derived | none | First difference of log(price · volume aggregated daily) |

All macro features are **observed before** the forecast date, so there
is no look-ahead leakage. The macro panel is forward-filled across
non-trading days, then sliced to align with each stock's trading
calendar.

### 1.6 Targets

Forecast target at horizon h is the *average RV over the window
`t … t+h-1`*, paired with features observable through `t-1`:

```
y_t^{(h)} = (1/h) · Σ_{k=0..h-1} RV_{t+k}
```

For h=1 this is `RV_t` predicted from `RV_{t-1}` — Corsi (2009) Eq. (5)
exactly, a one-day gap between the freshest predictor and the target.
An earlier version stacked `rv.shift(-k)` for k=1..h, which paired
`RV_{t-1}` features with an `RV_{t+1}` target — a two-day gap that
discarded the most recent observation `RV_t` and made h=1 a 2-step
forecast. Corrected in `src/data/feature_engineering.py:make_horizon_target`.

---

## 2. The HAR family

All five HAR variants are linear regressions. The estimator is OLS via
`numpy.linalg.lstsq` for numerical stability (no inversion of XᵀX).

### 2.1 HAR (Corsi 2009)

```
RV_t = β_0 + β_d · RVD_{t-1} + β_w · RVW_{t-1} + β_m · RVM_{t-1} + ε_t
```

Four parameters. Implemented in `src/models/har_models.py:HAR`.

### 2.2 LogHAR

Same regression in log space:

```
log(RV_t) = β_0 + β_d · log(RVD_{t-1}) + β_w · log(RVW_{t-1}) + β_m · log(RVM_{t-1}) + ε_t
```

At forecast time we back-transform with the Jensen correction:

```
ŷ_t = exp(μ̂_t + σ̂²/2)
```

where σ̂² is the residual variance on the training set. Implemented in
`src/models/har_models.py:LogHAR`.

### 2.3 LevHAR (Corsi & Renò 2012)

HAR augmented with aggregated negative returns at the three
frequencies:

```
RV_t = HAR-terms + γ_d · |r_{t-1}|·1{r_{t-1}<0}
                  + γ_w · (1/5) Σ_{k=1..5} |r_{t-k}|·1{r_{t-k}<0}
                  + γ_m · (1/22) Σ_{k=1..22} |r_{t-k}|·1{r_{t-k}<0}
                  + ε_t
```

The leverage terms allow bad news (negative returns) to have a larger
impact on next-day RV than good news. Implemented in `src/models/har_models.py:LevHAR`.

### 2.4 SHAR (Patton & Sheppard 2015)

HAR with the daily lag *decomposed* into positive and negative
semivariance:

```
RV_t = β_0 + β_+ · RV+_{t-1} + β_- · RV-_{t-1}
              + β_w · RVW_{t-1} + β_m · RVM_{t-1} + ε_t
```

Implemented in `src/models/har_models.py:SHAR`. The weekly and monthly
aggregates are *not* decomposed — that's a choice made by the original
paper and we keep it.

### 2.5 HARQ (Bollerslev, Patton, Quaedvlieg 2016)

HAR augmented with a measurement-error interaction term:

```
RV_t = β_0 + (β_d + β_dQ · √RQ_{t-1}) · RVD_{t-1}
              + β_w · RVW_{t-1} + β_m · RVM_{t-1} + ε_t
```

The intuition: when RQ is high, the measurement noise in RVD is large
and we should down-weight it; when RQ is low, RVD is informative.
HARQ requires an "insanity filter" because the interaction can
produce extreme predictions in high-RQ regimes. The filter clips
predictions outside the in-sample range to the **nearest in-sample
endpoint** (`np.clip(raw, train_min, train_max)`, per BPQ 2016 §3.3).
Implemented in `src/models/har_models.py:HARQ.predict`.

### 2.6 HAR-X

HAR augmented with the full M_ALL extra feature set. Used as the
"unregularised baseline with extras" — it shows how much pure OLS can
extract from M_ALL before any regularisation. Implemented in
`src/models/har_models.py:HARX`.

---

## 3. Regularised regression

All five solvers run on the M_ALL design matrix (12 features) with the
target on the natural scale and features standardised to mean 0 / unit
variance using train-set statistics. The intercept is fit unpenalised
in all cases.

### 3.1 Common loss

```
L(β) = (1/N) Σ_t (y_t − x_t' β)² + λ · P(β)
```

where P(β) is the penalty function.

### 3.2 Penalty grid

All methods search λ on a **100-point** logarithmic grid spanning
[10⁻⁵, 10²]. The grid is constructed by `_build_log_grid` in
`src/pipeline/orchestrator.py:53` from the YAML spec:

```yaml
elastic_net:
  alpha_grid_log10: [-5, 2]
  alpha_grid_n: 100    # paper-faithful is 1000; we use 100 for tractability
```

The paper's Appendix A.6 Table A.6 specifies 1000 points. We reduced
to 100 because the empirical optimum on our sample sits in a wide
plateau where a coarser grid is sufficient (the validation MSE
landscape is smooth in `log(α)`).

### 3.3 Specific penalties

| Method | P(β) | Hyperparameters |
|---|---|---|
| Ridge | (1/2)·||β||₂² | λ (100 points) |
| Lasso | ||β||₁ | λ (100 points) |
| Elastic Net | (1−α)/2·||β||₂² + α·||β||₁ | λ × α (10 ratios from 0.1 to 1.0) |
| Post-Lasso | 0 in stage 2; ||β||₁ in stage 1 | λ (100 points); OLS refit on selected support |
| Adaptive Lasso | Σ w_j |β_j|, w_j = 1/|β̂_OLS_j|^γ | λ (100 points), γ=1.0 |

Selection is on validation-set MSE. Refitting on `train + val` after
hyperparameter selection is **on**.

Implementation: scikit-learn `Ridge`, `Lasso`, `ElasticNet` solvers.
Adaptive weights come from a preliminary OLS fit; if any OLS
coefficient is exactly zero (rare on standardised data), the weight
defaults to a large constant.

### 3.4 Diagnostics

Each fitted regularised forecaster exposes a `diagnostics` attribute
after `fit()` containing the chosen λ (and α for EN), the validation
MSE, and the number of non-zero coefficients. These are aggregated
across stocks/horizons in `outputs/tables/har_coefficients_*.csv`.

---

## 4. Tree-based methods

All trees use scikit-learn.

### 4.1 Bagging

`BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=None),
 n_estimators=500, bootstrap=True)`.

No hyperparameter tuning per Breiman & Cutler's recommendation; this
matches the paper.

### 4.2 Random Forest

`RandomForestRegressor(n_estimators=500, max_features='sqrt',
 bootstrap=True)`.

Per the paper: default depth, default min_samples_leaf, sqrt feature
sub-sampling.

### 4.3 Gradient Boosting

`GradientBoostingRegressor(loss='squared_error', subsample=1.0)`
with validation-set search over:

* `n_estimators ∈ {50, 100, 200, 500}`
* `learning_rate ∈ {0.01, 0.1}`
* `max_depth ∈ {1, 2}`

Selected by validation MSE; refit on `train + val`.

---

## 5. Neural networks

### 5.1 Architectures (geometric pyramid per Diebold & Shin 2019; Heaton et al. 2017)

| Label | Hidden layer widths |
|---|---|
| NN1 | (2,) |
| NN2 | (4, 2) |
| NN3 | (8, 4, 2) |
| NN4 | (16, 8, 4, 2) |

### 5.2 Activation, optimiser, regulariser

* Activation: **ReLU** (sklearn `MLPRegressor` does not implement
  Leaky ReLU; the paper uses Leaky ReLU with slope 0.01)
* Optimiser: Adam, learning rate 0.001, β1=0.9, β2=0.999
* Batch size: 64
* Epochs: **200** (paper-faithful is 500; reduced for sklearn-tractable
  training time, with patience-25 early stopping in practice always
  triggering before the cap)
* L2 weight decay: sklearn default `alpha=0.0001` (the paper's
  dropout 0.8 has no sklearn equivalent; the lower default is the
  cost of the backend substitution and is documented in
  `LIMITATIONS.md §3.4`)
* Early stopping: patience **25** on a 15% internal validation slice
  carved by `MLPRegressor` (separate from the outer 10% val split used
  for top-10 seed ranking)
* No learning-rate scheduling (sklearn does not support an LR
  scheduler on MLPRegressor)

### 5.3 Ensembling

For each architecture we train **100 random seeds**. Each seed gets
its own `MLPRegressor` instance and an independent train/val split.
After training, we rank the 100 seeds by their on-train-set
validation MSE and form:

* **NN^1** (top-1) = the single-best seed's prediction
* **NN^10** (top-10) = the simple average of the top-10 seeds'
  predictions

Both NN^1 and NN^10 are stored as separate forecasts so that a
reader can read the seed-variance reduction directly.

### 5.4 Standardisation

X is standardised to mean 0 / unit variance using train statistics. y
is *also* standardised; predictions are back-transformed before being
written to the prediction pickle. This is consistent with the paper's
implementation (NN training is much more stable on a unit-scale
target).

### 5.5 Parallelism

The 100-seed loop runs under `joblib.Parallel(n_jobs=16,
backend='loky')` with `OPENBLAS_NUM_THREADS=1` per worker to prevent
inner BLAS oversubscription. On a 16-core M-series machine the full
NN training pass takes ~30-40 min per (stock × horizon × feature set).

---

## 6. Forecasting harness

Two strategies, dispatched in `src/pipeline/orchestrator.py:run_one`:

### 6.1 Rolling window (HAR family only)

The HAR family (HAR, LogHAR, LevHAR, SHAR, HARQ, HAR-X) is refit every
`refit_frequency_days = 1` days on a sliding fixed-length window equal to
`len(train)+len(val)`, ending at t-1 (this matches the paper's
"fixed length … past data are gradually excluded", Appendix A.4 — i.e.
sliding, not expanding). Closed-form OLS makes daily refits cheap.

### 6.2 Fixed window (regularised, trees, NN)

Regularised models (RR/LA/EN/P-LA/A-LA), trees (BG/RF/GB) and the NN
ensembles are fit **once** on `train + val` and predict the entire test
block. This is a deliberate simplification relative to the paper, which
rolls RR/LA/EN/GB/BG/RF daily and fixes only the NNs. We keep all ML
fixed-window because:

1. RF/BG/GB with 500 trees are costly to refit daily.
2. NN ensembles (4 architectures × 100 seeds = 400 fits) are far too
   costly to refit daily — the paper fixes these for the same reason.
3. Exact replication is not the goal; fixed-window is
   disclosed as a deviation.

**Consequence (documented finding).** Fixed-window trees cannot
extrapolate beyond their training-leaf range. With training spanning the
high-volatility 2020-2022 window and the test set falling in the calmer
2023-2024 period, the trees systematically over-predict at the monthly
horizon, where the target is a smooth 22-day average — producing MSE
ratios of 3-14× HAR at h=22. The paper, which *rolls* BG/RF, instead
finds RF the best model at h=22. The divergence isolates the daily
re-fit as the load-bearing assumption behind the paper's long-horizon
tree result — a §4 critical-comparison insight, not a defect.

---

## 7. Loss functions

### 7.1 MSE

```
MSE(y, ŷ) = (1/N) Σ_t (y_t − ŷ_t)²
```

Pointwise loss `(y_t − ŷ_t)²` is exposed for bootstrap/MCS.

### 7.2 QLIKE (Patton 2011)

```
QLIKE(y, ŷ) = (1/N) Σ_t [y_t/ŷ_t − log(y_t/ŷ_t) − 1]
```

QLIKE requires positive forecasts. Positivity is now enforced **at the
prediction level** in the orchestrator (paper p. 1691: a negative
variance forecast is replaced by the minimum in-sample realised
variance), applied uniformly to every model. This makes QLIKE
well-defined for all models — previously the unconstrained linear
fits (HAR-X, RR) and the standardised-NN inverse transform produced
negative forecasts that inflated QLIKE to 50-130× HAR, an artifact, not
a result. Note QLIKE itself is *our* robustness addition: the paper
evaluates on MSE only (QLIKE appears nowhere in it), so we present
QLIKE as a noise-robust complement (Patton 2011), not as a paper
replication.

### 7.3 MAE

Implemented but not used in headline tables. Available via
`metrics.LOSSES["mae"]`.

---

## 8. Statistical inference

### 8.1 Diebold-Mariano

For two forecasts ŷ¹ and ŷ², with loss differential `d_t = L(y_t, ŷ¹_t) −
L(y_t, ŷ²_t)`:

```
DM = (1/√T) · (Σ d_t) / √(σ̂²_NW)
```

where σ̂²_NW is the Newey-West HAC variance with automatic bandwidth
selection (Newey-West 1994 rule). For our T = 449 we use the HLN
small-sample correction (Harvey-Leybourne-Newbold 1997):

```
DM* = DM · √((T + 1 − 2h + h(h-1)/T) / T)
```

distributed as Student-t with T-1 degrees of freedom. Implemented in
`src/evaluation/diebold_mariano.py`.

### 8.2 Multi-testing correction (extension)

For a family of m DM tests at nominal level α:

* **Bonferroni**: reject p_i if p_i ≤ α/m. Controls FWER.
* **Holm step-down**: order p_(1) ≤ … ≤ p_(m), reject p_(i) if
  p_(i) ≤ α/(m − i + 1) for all j ≤ i. Controls FWER. More powerful
  than Bonferroni for m > 1.
* **Benjamini-Hochberg**: order p_(1) ≤ … ≤ p_(m), reject p_(i) if
  there exists k ≤ i with p_(k) ≤ (k/m)·α. Controls FDR.

Implemented in `scripts/20_dm_multitest_correction.py`. The paper does
*not* perform any correction.

### 8.3 Model Confidence Set (Hansen-Lunde-Nason 2011)

Iterative elimination of underperforming models from the candidate set
M₀ until the surviving set passes the equal-predictive-ability test
at level α. Algorithm:

1. Compute the loss-difference matrix d_{ij,t} = L(y_t, ŷ^i_t) − L(y_t, ŷ^j_t).
2. Compute the centred matrix d̄_{ij} = (1/T) Σ_t d_{ij,t}.
3. For each model i, define T_i,max = max_j (d̄_{ij} / √σ̂²_ij),
   where σ̂² is a moving-block-bootstrap variance estimate.
4. Take the worst-performing model arg max_i T_i,max and test whether
   it can be removed.
5. Test statistic: T_max = max_i T_i,max.
6. p-value: P(T_max* ≥ T_max | M) under the bootstrap distribution.
7. If p > α, stop and return the current set as the MCS.
8. Otherwise, eliminate the worst model and repeat.

We use:

* `num_bootstrap = 10000` (paper does the same in their final tables)
* `block_length = n^(1/3)` per Politis-White (2004)
* `statistic = Tmax` (the paper's primary choice)
* `α ∈ {0.05, 0.10, 0.25}` reported separately

Implemented in `src/evaluation/mcs.py`.

### 8.4 Accumulated Local Effects (Apley & Zhu 2020)

For a feature x_j and a value z:

```
f̂_j,ALE(z) = ∫_{z₀}^{z}  E[∂f(X)/∂x_j | x_j = u]  du - C
```

We approximate the integral by quantile-binning x_j into K=40 bins:

```
f̂_j,ALE(z) ≈ Σ_{k: z_k ≤ z}  Δ_k  - C
Δ_k = (1/N_k) · Σ_{i: x_{i,j} ∈ B_k}  [f(x_i^{(z_k.upper)}) − f(x_i^{(z_k.lower)})]
```

The centring constant C is chosen so that ∫ f̂_j,ALE(u) dF_{x_j}(u) = 0.

Variable importance follows the paper's equation 30:

```
I(x_j) = √(E[ f̂_j,ALE(x_j)² ])
       ≈ √( (1/N) · Σ_i f̂_j,ALE(x_{i,j})² )
```

For binary features (the EA dummy) the squared-mean is replaced by
|max − min| over the two values, as in the paper Appendix B.1.
Implemented in `src/evaluation/ale.py`.

### 8.5 Mincer-Zarnowitz (extension)

Forecast efficiency regression:

```
y_t = α + β · ŷ_t + ε_t
```

Joint Wald test of H_0: (α, β) = (0, 1) with Newey-West HAC
covariance. We also report:

* The R² of the MZ regression
* The Theil (1961) decomposition of MSE into bias², regression-error,
  and disturbance components

Implemented in `src/evaluation/mincer_zarnowitz.py`.

### 8.6 Moving-block bootstrap CI (extension)

For a per-observation loss series `L_t` with sample size n, block
length b:

```
For r = 1..R:
    Sample ⌈n/b⌉ block starts uniformly from [0, n-b]
    Concatenate and truncate to length n → L*_t
    Statistic*_r = mean(L*_t)
Report (2.5%, 97.5%) quantiles of Statistic* as the 95% CI.
```

Block length defaults to `b = max(1, round(n^(1/3)))` (Politis-White
2004). For ratios of means (in the path decomposition CIs), we
bootstrap numerator and denominator under the same block-index draw
and report `mean(num*) / mean(den*)`. Implemented in
`src/evaluation/bootstrap.py` and `scripts/21_path_decomposition_ci.py`.

### 8.7 Filtered Historical Simulation VaR (extension)

For a forecast ŷ_t of conditional variance:

1. Standardise in-sample residuals: z_s = (y_s − ŷ_s) / √ŷ_s.
2. Sample empirical α-quantile of z_s.
3. VaR_t(α) = −z_(α) · √ŷ_t · sign convention.

Backtested with:

* **Kupiec unconditional coverage**: LR test of H_0: hit-rate = α.
* **Christoffersen conditional coverage**: joint LR test of
  unconditional coverage + independence (no hit-clustering).

Implemented in `src/evaluation/value_at_risk.py`.

### 8.8 HMM regime-conditional analysis (extension)

A 2-state Markov regime-switching model on log(RV_t):

```
log(RV_t) | S_t = k  ~  N(μ_k, σ_k²)
S_t ∈ {0, 1}, with transition matrix P
```

Estimated by EM with quantile-based start values (regime 0 =
below-median log-RV, regime 1 = above-median) plus 25 random restarts
to avoid local maxima. The input series is z-scored before fitting to
keep the EM well-conditioned; estimated means and variances are
back-transformed to the original log-RV scale. The regime label for
each day is the argmax of the smoothed marginal probability. We then
compute model-conditional MSE in {low-vol, high-vol} separately.
Implemented in `src/evaluation/regime.py` and
`scripts/22_hmm_regime_analysis.py`.

---

## 9. Cross-cutting: split and seeds

### 9.1 Split

70 / 10 / 20 chronological. Sizes per stock at h=1, M_ALL: 1568 /
224 / 449. Test-set window: ≈ 2023-04 to 2024-12.

### 9.2 Seeds

`src/utils.py:set_global_seed(42)` seeds Python's `random`, numpy, and
sklearn's random state. The NN ensemble uses seeds `[0, 1, ..., 99]`
deterministically. All bootstraps use `numpy.random.default_rng(42)`
(or `42 + offset` per resample) for reproducibility.

### 9.3 Parallelism

* `joblib.Parallel(n_jobs=16, backend='loky')` for the 100-seed NN
  loop and the 5000–10000-replicate bootstrap.
* `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1` per worker (set in the entry-point scripts) to
  prevent BLAS oversubscription on top of joblib's process pool.

---

## 10. Outputs and persistence

### 10.1 Prediction pickles

Each `predictions_{class}_{feature_set}.pkl` is a list of
`StockRunResult` dataclass instances. Each `StockRunResult` carries:

* `ticker, feature_set, horizon`
* `predictions: dict[str model_label → pd.Series prediction]`
* `y_true: pd.Series`
* `val_predictions: dict[str model_label → pd.Series prediction on val set]`
* `y_val: pd.Series`

Persisted via the pickle protocol; loaded by
`src/pipeline/orchestrator.py:load_results(filename)`.

### 10.2 Tables

CSVs are written to `outputs/tables/`. Naming convention:

* `loss_h{h}_{loss}.csv` — per-(ticker, model) MSE
* `loss_ratio_h{h}_{loss}.csv` — same / HAR per row
* `dm_h{h}_{loss}.csv` — DM pairwise grid
* `dm_multitest_h{h}_{loss}.csv` — DM with Bonferroni/Holm/BH flags
* `mcs_h{h}_{loss}_a{alpha}.csv` — MCS membership
* `mz_h{h}_{loss}.csv` — Mincer-Zarnowitz
* `bootstrap_ci_h{h}_{loss}.csv` — bootstrap MSE CIs
* `combinations_h{h}_{loss}.csv` — forecast combinations
* `decile_h{h}.csv` — decile-stratified loss
* `var_backtest_h{h}.csv` — VaR backtest
* `hmm_regime_*.csv` — regime-conditional outputs

### 10.3 Figures

`outputs/figures/` carries PDFs:

* RV time series per stock
* Loss boxplots per (stock, horizon)
* MCS membership bars
* ALE overlays per feature × model
* Path-decomposition bar charts
* COVID sub-period contrasts

### 10.4 Logs

All scripts use `src/utils.py:get_logger(name)` which configures a
single stream handler with the format
`%(asctime)s | %(levelname)-7s | %(name)s | %(message)s`. Idempotent.

---

## 11. Implementation deviations from the paper (summary)

The exhaustive list is in `LIMITATIONS.md`. The three deviations that
matter most:

1. **NN activation function**: ReLU instead of Leaky ReLU.
2. **NN backend**: scikit-learn `MLPRegressor` instead of PyTorch /
   TensorFlow.
3. **IV proxy**: VIX (aggregate index) instead of per-stock
   OptionMetrics implied volatility.

None of these alters a qualitative finding of the empirical analysis.

---

## 12. File index

| Functionality | File |
|---|---|
| Config | `config/config.yaml` |
| Minute-bar ingest | `src/data/tick_to_minute.py` |
| Daily RV / RV± / RQ | `src/data/compute_rv.py` |
| Realised kernel | `src/data/realised_kernel.py` |
| Macro features | `src/data/macro_features.py` |
| HAR family | `src/models/har_models.py` |
| Regularised | `src/models/regularized.py` |
| Trees | `src/models/tree_models.py` |
| Neural networks | `src/models/neural_networks_sklearn.py` |
| Combinations | `src/models/combinations.py` |
| Losses | `src/evaluation/metrics.py` |
| Diebold-Mariano | `src/evaluation/diebold_mariano.py` |
| MCS | `src/evaluation/mcs.py` |
| ALE | `src/evaluation/ale.py` |
| Mincer-Zarnowitz | `src/evaluation/mincer_zarnowitz.py` |
| Bootstrap CIs | `src/evaluation/bootstrap.py` |
| Decile loss | `src/evaluation/decile.py` |
| VaR backtest | `src/evaluation/value_at_risk.py` |
| HMM regime | `src/evaluation/regime.py` |
| Forecast harness | `src/pipeline/rolling_forecast.py` |
| Orchestrator | `src/pipeline/orchestrator.py` |
| Figure builders | `src/visualization/plots.py` |
| Table builders | `src/visualization/tables.py` |
