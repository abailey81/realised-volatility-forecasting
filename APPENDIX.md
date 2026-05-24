# Appendix (≤3 pages) — full results, robustness, and code

All numbers are out-of-sample on the 2016–2024 test set (≈449 days), AAPL/AMZN/JPM.
Ratios are MSE relative to HAR; **< 1 beats HAR**. Full machine-readable tables are in `outputs/tables/`; the complete critique is in `CRITIQUE.md`.

## A1. Full 22-model out-of-sample MSE ratios vs HAR — M_ALL

**Horizon h = 1**

| | LogHAR | LevHAR | SHAR | HARQ | HAR-X | RR | LA | EN | P-LA | A-LA | BG | RF | GB | NN1¹⁰ | NN2¹⁰ | NN3¹⁰ | NN4¹⁰ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 0.944 | 1.026 | 1.004 | 0.985 | 1.015 | 0.987 | 0.941 | 0.953 | 0.929 | 1.991 | 1.167 | 1.069 | 1.002 | 0.920 | **0.891** | 0.938 | 0.921 |
| AMZN | 1.000 | 1.065 | 0.991 | 0.992 | 0.951 | 0.951 | 0.881 | 0.970 | 0.959 | 1.525 | 1.140 | 0.963 | 0.978 | 0.869 | 0.879 | **0.861** | 0.941 |
| JPM | 0.924 | 1.049 | 1.177 | 1.163 | 1.119 | 1.093 | 0.951 | 0.953 | 0.979 | 2.169 | 1.054 | 0.952 | 0.955 | 0.977 | **0.912** | 0.950 | 0.944 |

**Horizon h = 22** (note the fixed-window tree collapse — see A3)

| | LogHAR | LevHAR | SHAR | HARQ | HAR-X | RR | LA | EN | P-LA | A-LA | BG | RF | GB | NN1¹⁰ | NN2¹⁰ | NN3¹⁰ | NN4¹⁰ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | **0.666** | 0.842 | 1.008 | 1.025 | 1.123 | 1.079 | 1.021 | 1.031 | 0.921 | 1.638 | 7.314 | 3.038 | 2.677 | 1.067 | 1.009 | 0.926 | 0.814 |
| AMZN | 1.214 | 0.949 | 0.996 | 1.161 | 1.141 | 1.114 | **0.901** | 1.085 | 1.115 | 1.410 | 6.719 | 3.885 | 4.069 | 1.266 | 1.129 | 1.245 | 1.364 |
| JPM | **0.630** | 1.112 | 0.988 | 1.476 | 1.494 | 1.481 | 0.976 | 1.012 | 0.721 | 1.694 | 13.835 | 5.274 | 1.625 | 0.901 | 0.860 | 0.716 | 1.309 |

## A2. Apples-to-apples M_HAR (3 RV lags only), h = 1 — supports critique (i)

| | LogHAR | HARQ | RR | LA | EN | P-LA | BG | RF | GB | NN2¹⁰ | NN3¹⁰ | NN4¹⁰ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 0.944 | 0.986 | 0.992 | 1.023 | 1.021 | 1.003 | 1.578 | 1.406 | 1.036 | 0.997 | 0.978 | 0.976 |
| AMZN | 1.003 | 0.997 | 0.991 | 0.993 | 0.995 | 0.995 | 1.420 | 1.260 | 1.126 | 1.046 | 1.064 | 1.042 |
| JPM | 0.916 | 1.161 | 0.984 | 0.995 | 0.994 | 0.977 | 1.363 | 1.231 | 1.088 | 0.999 | 0.995 | 0.985 |

On M_HAR only the NN ensembles (and LogHAR) reach HAR; the three tree methods lose by 4–58% (BG worst, GB mildest) and adaptive-Lasso by ~113%, while the other regularisers (RR/LA/EN/P-LA) tie HAR (0.98–1.02). The ML gain appears only once the nine extra predictors are added (A1) — i.e. it is mostly information set, not nonlinearity.

## A3. Critique evidence (verified numbers; full development in `CRITIQUE.md`)

- **Path decomposition, h=1 (HAR→best NN).** Regularisation gain +4.7/3.0/4.7 pp (AAPL/AMZN/JPM); tree marginal −11.6/+0.7/+0.1 pp; deep-NN marginal +17.8/+10.2/+4.0 pp; total ML gain 10.9/13.9/8.8 pp; regularisation share 0.4/0.2/0.5. Regularisation does a large slice of the work; the tree step is flat-to-negative.
- **Fixed- vs rolling-window trees, AAPL h=22 (controlled experiment).** RF MSE/HAR = **3.04 fixed → 1.55 rolling** (daily refit). The daily re-fit, not the tree class, drives CSV's long-horizon result; it does not fully reach CSV's RF<HAR, so the result is also sample-conditional.
- **Significance.** MCS (90%) retains 20–21 of 22 models per stock at h=1, HAR never eliminated. One-sided DM: only 0–3 models per stock beat HAR at 5%; under Holm/BH correction essentially none survive at h=1/h=5.
- **Loss-conditional (QLIKE), h=1, M_ALL ratios vs HAR (mean over 3 stocks).** LogHAR 0.96, RF 0.99, but LA 1.28, NN3¹⁰ 1.06, NN2¹⁰ 1.17, **EN 2.32** — the MSE edge largely reverses under the noise-robust loss.
- **Measurement error (realised kernel vs 5-min RV), h=1 MSE ratio RK/RV.** JPM 0.729 (RK 27% lower), AMZN 0.845, AAPL 1.065 — switching the *estimator* moves JPM MSE more than the entire ML-vs-HAR gap.
- **Error correlation.** Every model's out-of-sample errors correlate with HAR's at r ≥ 0.79, and the eight models that beat HAR at r ≈ 0.88–0.99 — each winner is HAR plus a small orthogonal perturbation, so its edge lives in under ~20% of error variance. (Lowest pairwise corr 0.63 is A-LA vs a single-seed NN, neither of which beats HAR.)
- **Regime conditionality (COVID split, 2016–2019 train / 2020–2024 test).** The best-ML-vs-HAR edge at h=1 collapses from ≈9–14% (standard split) to ≈0–2%: AAPL P-LA −2.1%, AMZN EN −1.7%, JPM none beats HAR (HARQ +0.2%). The headline gain is conditional on the absence of a regime break in the test window (`covidfull_loss_ratio_*`).

## A4. Variable importance (ALE), AAPL h=1 — reproduces CSV Figure 7 pattern

| feature | HAR-X | EN | RF | NN2¹⁰ |
|---|---|---|---|---|
| RVD | 0.316 | 0.382 | 0.204 | 0.040 |
| RVW | 0.250 | 0.357 | 0.183 | 0.186 |
| IV | 0.106 | 0.191 | **0.214** | **0.232** |
| VIX | 0.106 | 0.000 | 0.230 | 0.146 |
| EA | 0.000 | 0.000 | 0.000 | 0.000 |

RVD/RVW dominate for every model (agreement on the top features); IV/VIX are weighted higher by RF/NN than by the linear models (the "ML weighs implied vol more" point). EA has zero ALE importance — the known ALE weakness for a sparse 0/1 dummy, as CSV note. (We cite CSV's Table A.5 for HAR-X coefficient t-stats: our proxy makes IV≡VIX collinear, so per-coefficient inference on those two is unstable — itself critique (vi).)

## A5. Selected code excerpts (methodological fidelity)

**Horizon target (one-step alignment, RV_{t-1}→RV_t):**
```python
def make_horizon_target(rv, horizon):
    parts = [rv.shift(-k) for k in range(horizon)]   # mean of RV_t..RV_{t+h-1}
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False).rename("y")
```
**HARQ insanity filter (clip to in-sample range, BPQ 2016):**
```python
raw = X[cols].to_numpy() @ self.beta_ + self.intercept_
return np.clip(raw, self.train_min_, self.train_max_)
```
**Positivity filter applied to every model (CSV p.1691):**
```python
floor = float(train["y"].min())
for store in (predictions, val_predictions):
    for k in list(store): store[k] = store[k].clip(lower=floor)
```
**MCS T_max statistic with moving-block bootstrap (Hansen–Lunde–Nason 2011):**
```python
d_i   = Lbar - Lbar.mean()
sd    = boot_d.std(axis=0, ddof=1)
t_stat, boot_t = d_i / sd, (boot_d - d_i) / sd
pval  = np.mean(boot_t.max(axis=1) >= t_stat.max())      # eliminate argmax if pval<=alpha
```

## A6. Data, deviations, reproducibility (one line each)
5-min RV within-day (no overnight), BNHLS outlier filter, RV/RV±/RQ; 70/10/20 chronological split. **Deviations (disclosed):** sklearn NN (ReLU, no dropout; early stopping in dropout's place); regularised + trees fixed-window (HAR & NN match CSV); IV proxied by VIX; 100-point λ grid. 37 unit tests pass. One command: `python scripts/00_run_full_pipeline.py`.
