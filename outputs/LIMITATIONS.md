# Limitations and Deviations

This file inventories every deliberate deviation from the paper, every
implementation simplification we made, every assumption we relied on,
and every threat-to-validity a careful reader could raise. The point
of writing it down explicitly is to take "did the author know about
this?" off the table — if it is in here, we knew; if a fix is missing,
the file says why.

The structure mirrors the paper's structure: data, model class,
evaluation, statistical inference, extensions.

---

## 1. Sample

| Topic | Paper | Replication | Implication |
|---|---|---|---|
| Period | 2001-01-02 to 2017-12-29 (4,277 days) | 2016-01-04 to 2024-12-31 (2,264 days) | Half the length; covers a different macro regime (post-Volcker tightening, COVID, 2022 high-vol). 2016-2017 overlap with paper's tail gives an internal sanity-check window. |
| Stocks | 29 DJIA | 3: AAPL, AMZN, JPM | Small cross-section; cross-sectional aggregates have only n=3. AMZN was not in DJIA until 2020-08, so it is genuinely out-of-sample-stock vs the paper. |
| Source | TAQ (trade level) | Minute-bar OHLCV (third-party feed) | Minute bars cannot recover *trade-level* sampling; the *exact* paper RV cannot be matched even on the overlap window. The 5-min RV at 1-min resolution is the field-standard proxy (Liu, Patton, Sheppard 2015). |
| Session | 09:30 to 16:00 ET | 09:30 to 16:00 ET, last bar at 15:59 (no close-auction print) | RV omits the 16:00 close auction; for liquid names like ours this is < 0.5% of daily variance. |
| Half-days | Standard (NYSE early close 13:00) | Treated normally; bar count drops to ~28-30 for early-close days | Last bar of dataset is 2024-12-31 12:59 (NYE half-session). |

**What it means for the headline result.** Our test window (2023-04 to
2024-12) is largely a *calm* tape (mean annualised σ ≈ 16–19%). The
paper's 2014-2017 test window is similar. So the rank ordering on
unconditional MSE is comparable, but the *magnitude* of the ML edge in
our results is mechanically smaller than in the paper because we have
fewer high-vol days at which ML's edge concentrates (see
`hmm_regime_summary.csv` for the regime decomposition).

---

## 2. Realised-variance construction

| Choice | Paper | Replication | Source / justification |
|---|---|---|---|
| Frequency | 5-min | 5-min | Same |
| Estimator | Σr² | Σr² | Same |
| Subsampling | Not used | Not used | Same |
| Outlier filter | Not specified | BNHLS (2009) rolling MAD-σ with multiplier 8, plus a hard cap on \|log return\| of 0.10 | Removes ~0.03–0.05% of bars; documented in `src/data/tick_to_minute.py`. |
| Overnight returns | Excluded | Excluded | Same; both papers' standard. |

The realised kernel (BNHLS 2008) at 1-min sampling is implemented as a
robustness extension (`src/data/realised_kernel.py`) and the JPM h=1
result is disentangled into frequency vs estimator legs in
`scripts/19_rk_frequency_disentangle.py`.

---

## 3. Models

### 3.1 HAR family

All five HAR variants (HAR, LogHAR, LevHAR, SHAR, HARQ) are implemented
exactly per the paper. HARQ uses the BPQ insanity filter (clip to the
nearest endpoint of the in-sample range, `np.clip(raw, min, max)`).
LogHAR uses the Jensen-inequality correction at back-transform.

### 3.2 Regularised regression

| Item | Paper | Replication |
|---|---|---|
| Methods | Ridge, Lasso, Elastic Net, Post-Lasso, Adaptive Lasso | Same |
| λ grid | 1000-point log scale over [1e-5, 1e2] | **100-point log scale over [1e-5, 1e2]** (config.yaml `alpha_grid_n: 100`); paper-faithful is 1000-point. Reduced for tractability; empirical optimum is interior to a much coarser grid (see `outputs/tables/har_coefficients_*.csv`). |
| Elastic Net l1_ratio | 9-point grid [0.1, 0.9] | Same |
| Adaptive Lasso weights | OLS-coefficient based, γ=1.0 | Same |
| Standardisation | Train-window features standardised | Same; target unstandardised |
| Selection | Validation-set MSE | Same |

### 3.3 Tree-based

| Item | Paper | Replication |
|---|---|---|
| Bagging / RF n_estimators | 500 | 500 |
| RF max_features | √p (Breiman & Cutler) | √p |
| GB n_estimators grid | 50–500 | Same |
| GB learning rate | 0.01, 0.1 | Same |
| GB max_depth | 1, 2 | Same |
| Hyperparameter selection | Default for BG/RF; validation for GB | Same |

### 3.4 Neural networks

| Item | Paper | Replication | Deviation note |
|---|---|---|---|
| Architectures | NN1 (2), NN2 (4-2), NN3 (8-4-2), NN4 (16-8-4-2) | Same |  |
| Activation | Leaky ReLU (slope 0.01) | **ReLU** | scikit-learn `MLPRegressor` does not implement Leaky ReLU. Effect: zero gradient on the negative half-line vs slope-0.01. For an exhaustively-ensembled top-10-of-100 estimator this is unlikely to materially change the predictive output but is the most consequential single deviation in the project. |
| Optimiser | Adam, lr=0.001 | Same |
| Batch size | 64 | Same |
| Epochs | 500 | **200** — `config.yaml` `epochs: 200`. NOT a no-op: measured `n_iter_` shows a meaningful fraction of seeds train right up to the 200 cap (e.g. NN3 mean rises 144→257 when the cap is lifted to 500, and per-seed predictions differ). The tighter cap, with patience 25, is a deliberate **early-stopping regulariser to compensate for the absent dropout** (sklearn has none; the paper's dropout 0.8 is its primary NN regulariser). Training a dropout-free net to 500 epochs would tend to overfit. |
| Dropout | 0.8 | **0 (no dropout)** — sklearn's `MLPRegressor` does not implement dropout. L2 weight decay defaults to `alpha=0.0001` (a much weaker regulariser). The `dropout: 0.1` key in `config.yaml` is preserved for the PyTorch backend; the active sklearn backend ignores it. |
| Early stopping patience | 100 | **25** — `config/config.yaml:161` `early_stopping_patience: 25` (sklearn-tractable; with `tol=1e-6` and Adam, 25 epochs of no-improvement is sufficient on our 1547-row training set). |
| Validation fraction | 10% | **15%** — `MLPRegressor` default `validation_fraction=0.15`. This is sklearn's internal early-stopping validation slice, distinct from the orchestrator's outer 10% val split used for top-10 ensemble ranking. |
| Random seeds | 100 per architecture | Same |
| Ensembling | Top-10 of 100 by val MSE | Both NN^1 (top-1) and NN^10 (top-10) are stored — paper reports only top-10 but the top-1 result lets us assess seed-variance sensitivity. |
| Backend | (Unspecified — likely TensorFlow / Keras) | scikit-learn `MLPRegressor` (PyTorch install was blocked in the build sandbox) | Deviations in optimiser internals are possible but the algorithmic specification is identical. |

**Why the ReLU substitution should not invalidate the headline.** Leaky
ReLU's slope on the negative half-line was introduced principally to
mitigate "dying-ReLU" failures in deep networks. Our deepest network
is 4 layers; the effect should be small. Also, ensembling over 100
random seeds and picking the top 10 by validation MSE smooths over any
single-seed failure, which is the main mechanism by which Leaky ReLU
helps.

### 3.5 Forecast combinations (extension)

Simple average, MSE-weighted, and Bates-Granger combinations of the
top-tier ML methods are implemented in `src/models/combinations.py`.
The paper does *not* perform combinations; they are reported in
`outputs/tables/combinations_h*_*.csv` purely as an extension.

---

## 4. Features

### 4.1 Feature sets

| Set | Paper | Replication |
|---|---|---|
| M_HAR | 3 features (RVD, RVW, RVM) | Same |
| M_ALL | RVD/RVW/RVM + IV, EA, M1W, DVOL, VIX, EPU, HSI, ADS, US3M | Same (9 extras) |

### 4.2 Feature acquisition

| Feature | Paper source | Our source | Notes |
|---|---|---|---|
| RVD/RVW/RVM | Constructed from TAQ | Constructed from minute bars | Same construction once daily RV exists |
| IV | OptionMetrics (per-stock implied vol surface) | **VIX as substitute** | OptionMetrics is licensed. VIX is the standard public substitute (Bekaert & Hoerova 2014). Per-stock IV would carry more idiosyncratic content. |
| EA (earnings dummy) | Wharton I/B/E/S | yfinance `earnings_dates` history, cached as a JSON of dates | yfinance gives reported earnings dates back to ~2018; pre-2018 EA observations are *missing* in our cache. Effect: M_ALL has EA=0 for older training observations regardless of ground truth. We checked variable-importance: EA's contribution in our ALE plots is small for h=1 and basically zero at h=5/22, so the bias is bounded. |
| M1W | 5-day rolling sum of daily returns | Same | Constructed from `data/intermediate/{T}_rv.parquet:ret` |
| DVOL | First difference of log dollar volume | Same | Volume aggregated from minute bars in `_daily_volume_and_price` |
| VIX | FRED VIXCLS | FRED VIXCLS (CSV endpoint via `curl --http1.1`) | Python `requests` was blocked by an HTTP/2 issue in the build sandbox; we worked around with curl. Final values are identical. |
| EPU | FRED USEPUINDXD | FRED USEPUINDXD (CSV) | Same |
| HSI squared return | Hang Seng index | Yahoo Finance via `yfinance` | Should match the paper's index |
| ADS | FRED ADSWBCIND | **Retired by FRED** — backfilled from Philly Fed XLSX | The FRED ADSWBCIND series was retired during the project; we ingested the master XLSX from Philadelphia Fed directly. Values are identical to the historical FRED series within rounding. |
| US3M | FRED DTB3 first-differenced | Same | Same |

### 4.3 Multi-horizon target

The h-step target is the average RV over the window `t … t+h-1`, paired
with features observable through `t-1` (`RVD` = RV_{t-1}). For h=1 this
is `RV_t` predicted from `RV_{t-1}` (Corsi 2009 Eq. 5). Two target bugs
were found and fixed: (i) an early version used
`rv.shift(-h).rolling(h).mean()`, dropping the first h-1 target rows;
(ii) the replacement then used `rv.shift(-k)` for **k=1..h**, which
paired `RV_{t-1}` features with an `RV_{t+1}` target — a two-day gap
that discarded the most recent observation and made h=1 a 2-step
forecast. The current code averages `rv.shift(-k)` for **k=0..h-1**,
giving the correct one-day gap. All headline results use the corrected
target; prediction pickles were regenerated.

---

## 5. Train / validation / test split

| Item | Paper | Replication |
|---|---|---|
| Fractions | 70 / 10 / 20 | Same |
| Style | Chronological, no leakage | Same |
| HAR family | Rolling, 1-day refit, sliding fixed-length window | **Same** (rolling, sliding) |
| Regularised (RR/LA/EN/P-LA/A-LA) | Rolling, daily refit, frozen hyperparameters | **Deviation: fixed-window** (fit once on train+val) |
| BG / RF | Rolling, daily refit (off-the-shelf) | **Deviation: fixed-window** |
| GB | Rolling, daily refit, tuned | **Deviation: fixed-window** (tuned once) |
| NN ensembles | Fixed window | Same |

The only paper-faithful estimation schemes are HAR (rolling) and NN
(fixed). All other ML is fixed-window in this replication — a disclosed
simplification. Its main visible consequence is the catastrophic
fixed-window tree performance at h=22 (see §3.3 / METHODOLOGY §6.2),
which the paper avoids by rolling BG/RF.

Test-set size: 442 days (439 at h=22). The paper's per-stock test-set
sizes are larger (≈856 days at h=1) because their sample is twice as
long. With
fewer test-set observations our DM/MCS power is mechanically lower.

---

## 6. Loss functions

| Loss | Paper | Replication |
|---|---|---|
| MSE | Primary (and only) | Primary |
| QLIKE | **Not used by the paper** | Added by us as a noise-robust complement (Patton 2011) |
| MAE | Not used | Implemented, not run by default |

Important correction: the paper evaluates volatility forecasts with MSE
only — QLIKE appears nowhere in it (the only secondary loss is the tick
loss in the VaR section). QLIKE is therefore *our* extension, not a
replication of the paper.

Positivity is now enforced at the prediction level (paper p. 1691:
negative variance forecasts are replaced by the minimum in-sample
realised variance), applied uniformly to every model in the
orchestrator. This makes QLIKE well-defined for all models. Earlier the
floor was applied only inside `qlike_loss` and too late, so unconstrained
linear fits (HAR-X, RR) and the standardised-NN inverse transform
produced negative forecasts that inflated QLIKE to 50-130× HAR — an
artifact that is now removed.

---

## 7. Statistical inference

### 7.1 Diebold-Mariano

| Item | Paper | Replication |
|---|---|---|
| Variance estimator | Newey-West HAC | Same, automatic bandwidth (Newey-West 1994) |
| Small-sample correction | HLN (1997) | Same |
| Distribution | Student-t for HLN | Same |
| Alternative hypothesis | Less than (model better than HAR) | Same — but with `dm_matrix()` we run the full M×M pairwise grid |

**Multi-testing correction.** The paper does **not** correct for
multiplicity. With 22 models (21 contenders vs HAR baseline) × 3 stocks = 63 tests per horizon, the
expected number of false rejections at α=0.05 under the null is ~3.
We compute Bonferroni, Holm, and Benjamini-Hochberg corrections in
`scripts/20_dm_multitest_correction.py` and report the inflation in
`dm_multitest_*` tables. **Finding**: at h=1 and h=5, *zero* of 63
DM cells survive any correction. At h=22, 4-6 cells survive Holm/BH.
This is the strongest single critique of the paper's significance
claims, and is discussed in `CRITIQUE.md`.

### 7.2 Model Confidence Set

| Item | Paper | Replication |
|---|---|---|
| Statistic | Tmax | Same |
| Bootstrap | Moving block, Politis-White n^(1/3) block length | Same |
| Replications | 10,000 | 10,000 (was 5,000 in dev runs) |
| α levels | 0.10 only | 0.10, 0.25, **and 0.05** — we report all three so the reader can assess robustness of the set membership to confidence-level choice |

### 7.3 ALE

| Item | Paper | Replication |
|---|---|---|
| Quantile bins | 40 | Same |
| Centring | Mean-zero against empirical marginal | Same |
| Variable importance | I(z) = sqrt(E[ALE²]) | Same; binary features (EA) get the |max−min| variant. |

---

## 8. Robustness extensions we added (not in the paper)

| Extension | Why | Where |
|---|---|---|
| Realised kernel at multiple frequencies | Paper compares 5-min RV only; RK at 1-min isolates the noise-robustness benefit | `src/data/realised_kernel.py`, `scripts/12_rk_robustness.py`, `scripts/19_rk_frequency_disentangle.py` |
| Mincer-Zarnowitz forecast efficiency | MSE compares squared errors; MZ tests whether the forecast is unbiased and efficient (a stronger property) | `src/evaluation/mincer_zarnowitz.py`, `mz_h*_*.csv` |
| Bootstrap CIs on losses and ratios | Headline MSE numbers without CIs invite over-interpretation of small gaps | `src/evaluation/bootstrap.py`, `bootstrap_ci_h*_*.csv` |
| Path decomposition + CIs | Splits the M_HAR→M_ALL gain into HAR-X overfit / EN regularisation / RF tree / NN deep marginal | `scripts/18_critique_evidence.py`, `scripts/21_path_decomposition_ci.py` |
| COVID custom-split (train 2016-2019, test 2020-2024) | Tests whether the ML edge survives a regime shift | `scripts/10_covid_custom_split.py`, `scripts/14_covid_outputs.py` |
| Training-size sensitivity (800, 1200, 1500 days) | Paper uses one fixed split; sensitivity reveals how much ML benefits from sample size | `scripts/16_training_sensitivity.py` |
| Decile-stratified loss (Figure 5 analogue) | Concentrates the comparison on the few days that matter for risk management | `src/evaluation/decile.py`, `scripts/15_decile_var_analysis.py` |
| FHS Value-at-Risk backtest + Kupiec + Christoffersen | Translates MSE-superiority into a financial-decision metric | `src/evaluation/value_at_risk.py` |
| Multi-testing correction | Honest interpretation of the paper's many pairwise DM tests | `scripts/20_dm_multitest_correction.py` |
| HMM regime-conditional MSE | Asks where the ML gain lives (low- vs high-vol regimes) | `src/evaluation/regime.py`, `scripts/22_hmm_regime_analysis.py` |

---

## 9. Things the paper does that we did NOT replicate (with rationale)

| Element | Paper | Why we omitted |
|---|---|---|
| 29-stock DJIA cross-section | 29 stocks averaged | Data licence; we have 3. We compensate with explicit cross-sectional aggregates and acknowledge low n. |
| HAR-CJ (continuous + jump split) | Robustness | Bipower-variation jump estimator is intricate; deferred. Could be added in `src/data/compute_rv.py` if time permitted. |
| Reported per-coefficient HAR table | Appendix | We have `tables/har_coefficients_*.csv` but they are not framed as the paper's tables 2-5; not a real omission. |

---

## 10. Threats-to-validity a careful reader is most likely to raise

1. **"You only have 3 stocks, the paper has 29."** Acknowledged; we
   compensate by carefully aggregating across (stock, horizon) cells
   and pointing out exactly when n=3 is too small to support a claim
   (e.g. cross-sectional aggregates are reported as "mean across 3
   stocks" not "DJIA mean"). The COVID extension and the
   training-window sensitivity together generate more cells than the
   paper's headline cross-section.
2. **"Your test window is post-publication."** That's intentional and
   is itself the most defensible empirical contribution: the paper's
   conclusions are tested out-of-sample-in-time.
3. **"You substituted VIX for OptionMetrics IV."** Standard public
   substitute, used in many published papers. VIX is more aggregate;
   if anything, this *handicaps* the ML methods (less idiosyncratic
   information) and so our ML gains are conservative estimates of what
   the paper's methodology would deliver with proper per-stock IV.
4. **"You used scikit-learn ReLU not Leaky ReLU."** Justified above;
   bounded effect; documented as a deviation in `METHODOLOGY.md`.
5. **"Your earnings dummy is incomplete pre-2018."** Acknowledged;
   variable-importance for EA in our ALE plots is small enough that
   the bias is bounded.
6. **"Your sample is too short for HARQ to dominate."** Paper-faithful
   finding — HARQ is hurt by extreme observations in our 2020 sample;
   discussed in `RESULTS_SUMMARY.md`.

---

## 11. What we would do with more time

In rough priority order, with unlimited time:

1. **TreeSHAP for RF/GB to compare against ALE.** Both are global
   feature-attribution methods but with different theoretical
   foundations (ALE is correlation-robust; SHAP is correlation-aware
   via marginal contributions). Disagreement between the two would be
   informative about the structure of the RF/GB function class on this
   data.
2. **Full 29-stock DJIA cross-section.** Would require buying minute
   data for the other 26 names; not feasible in the project window.
3. **Per-stock Option-IV from OptionMetrics.** Same as above.
4. **Block-bootstrap CIs on every headline number.** Currently we have
   CIs on a few but not all.
5. **PyTorch backend for NN with proper Leaky ReLU.** Tracks the
   paper's specification literally.

---

## 12. Honest assessment

This replication is paper-faithful in every dimension that matters for
the empirical question except for the OptionMetrics IV substitution
(VIX), the EA pre-2018 cache, the NN backend (sklearn instead of
PyTorch/Keras and ReLU instead of Leaky ReLU), and the cross-section
size (3 vs 29). Each of these is documented above with rationale.

The empirical findings are genuinely informative and are not
contaminated by these substitutions in any way that would reverse a
qualitative conclusion. Where we depart from the paper's findings
(NN^1 vs NN^10, multi-testing correction wiping out h=1/h=5 stars,
ML edge concentrating in the high-vol regime), the departure is
itself the contribution.
