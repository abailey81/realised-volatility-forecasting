# Realised-Variance Forecasting: HAR vs Machine Learning

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)
![status](https://img.shields.io/badge/build-replication%20%2B%20critique-informational)

A Python **replication and critical evaluation** of Christensen, Siggaard & Veliyev
(2023), *A Machine Learning Approach to Volatility Forecasting* (Journal of
Financial Econometrics 21(5), 1680–1727). The paper asks whether off-the-shelf
machine learning beats the Heterogeneous Autoregressive (HAR) benchmark at
forecasting daily realised variance, and where any gain comes from. This
repository reproduces the methodology end to end and tests it **out of sample on
AAPL, AMZN and JPM over 2016–2024** — a window seven years later than the original,
spanning the COVID shock and the 2022 tightening. The contribution is a critique:
where the data allow it agrees with the paper, but it shows the machine-learning
edge is **fragile** — mostly information set plus regularisation rather than
nonlinearity, loss- and regime-dependent, and statistically thin. It does **not**
claim ML beats HAR overall.

> AAPL and JPM are in the paper's Dow Jones cross-section; AMZN is not (it joined
> the index only in 2020), so it serves as out-of-sample-stock evidence. The aim
> is a methodologically faithful replication and an honest comparison, not exact
> numerical reproduction — the data and sample period differ from the original.

---

## Contents

- [Headline result](#headline-result)
- [What is implemented](#what-is-implemented)
- [Repository map](#repository-map)
- [Reproducing](#reproducing)
- [Key findings and critique](#key-findings-and-critique)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Reference](#reference)
- [License](#license)

---

## Headline result

At the one-day horizon on the wider feature set (M_ALL), the best model beats HAR
by **9–14%** per stock — neural-net ensembles lead, lasso close behind — which
matches the original's magnitude and ordering. On the three-lag set (M_HAR) the
gain nearly disappears and only the NN ensembles beat HAR.

**Out-of-sample MSE relative to HAR, h = 1, M_ALL** (ratio < 1 beats HAR; bold = best per row):

| Stock | LogHAR | LA | EN | RF | BG | NN2¹⁰ | NN3¹⁰ |
|-------|:------:|:--:|:--:|:--:|:--:|:-----:|:-----:|
| AAPL  | 0.944  | 0.941 | 0.953 | 1.069 | 1.167 | **0.891** | 0.938 |
| AMZN  | 1.000  | 0.881 | 0.970 | 0.963 | 1.140 | 0.879     | **0.861** |
| JPM   | 0.924  | 0.951 | 0.953 | 0.952 | 1.054 | **0.912** | 0.950 |

Best per stock: AAPL 10.9% (NN2¹⁰), AMZN 13.9% (NN3¹⁰), JPM 8.8% (NN2¹⁰).
Full 22-model tables for h = 1 and h = 22 are in [`APPENDIX.md`](APPENDIX.md);
machine-readable versions in [`outputs/tables/`](outputs/tables/).

![One-day MSE relative to HAR](outputs/figures/headline_h1_ratios.png)

---

## What is implemented

Twenty-two forecasting models behind a single `Forecaster` interface (the count
matches the paper's comparison set used in every loss table):

| Family | Members |
|--------|---------|
| **HAR** | HAR, LogHAR (Jensen back-transform), LevHAR, SHAR, HARQ (insanity filter), HAR-X |
| **Regularised regression** | ridge, lasso, elastic net, adaptive lasso, post-lasso |
| **Tree ensembles** | bagging, random forest, gradient boosting |
| **Neural networks** | four geometric-pyramid architectures, each as a single network (NN¹) and a 10-of-100 validation-ranked ensemble (NN¹⁰) |

**Design axes**

| Axis | Values |
|------|--------|
| Feature sets | `M_HAR` (3 RV lags) · `M_ALL` (+ 9 macro/firm series: IV, earnings dummy, momentum, dollar-volume, VIX, EPU, Hang Seng, ADS, 3-month rate) |
| Horizons | 1, 5, 22 days |
| Losses | MSE (primary, as in the paper) · QLIKE (added as a noise-robust complement) |
| Inference | Diebold–Mariano with Newey–West HAC + HLN small-sample correction · Hansen–Lunde–Nason Model Confidence Set · Holm/BH multiple-testing correction |
| Interpretability | Accumulated Local Effects (ALE), TreeSHAP cross-check |

Realised variance, signed semivariances (RV±) and realised quarticity (RQ) are
constructed from 5-minute returns within each trading day (no overnight returns).
The critical path is covered by **37 unit tests**.

---

## Repository map

```
src/
  data/          minute-bar cleaning, 5-min RV / RV± / RQ, realised kernel, macro features
  models/        HAR family, regularised, trees, neural networks, forecast combinations
  evaluation/    MSE/QLIKE, Diebold–Mariano, MCS, Mincer–Zarnowitz, ALE, bootstrap, VaR, regimes
  pipeline/      rolling/fixed-window forecast harness and the orchestrator
  visualization/ figure and table builders
scripts/         numbered pipeline stages (01 preprocess → 09 outputs) + extensions (10–27)
tests/           unit tests for the critical path (37 passing)
config/          all hyperparameters, paths and switches (config.yaml)
outputs/
  tables/        result tables (CSV/TeX) — the tracked empirical evidence
  figures/       curated figures (PNG tracked; per-stage *.pdf renders gitignored)
  *.md           methodology / results / limitations / reproducibility notes
data/            inputs and caches — gitignored, not shipped (see data/README.md)
REPORT.pdf · APPENDIX.pdf · CRITIQUE.md   the write-up
```

The write-up is the three-page [`REPORT.pdf`](REPORT.pdf), the curated
[`APPENDIX.pdf`](APPENDIX.pdf), and the deep critique reference
[`CRITIQUE.md`](CRITIQUE.md).

---

## Reproducing

Requires **Python 3.11+**. The minute-bar price data is **not redistributed**
here (see [`data/README.md`](data/README.md) for the expected layout); the macro
inputs are downloaded from public sources (FRED, Yahoo Finance) by stage 2.
The `data/` tree and per-stage figure PDFs are gitignored.

```bash
pip install -r requirements.txt
python -m pytest -q                    # 37 passing
python scripts/00_run_full_pipeline.py # stages 1–9: preprocess → outputs
```

The end-to-end driver runs the core stages (preprocess, macro download, build
features, train each model family, run the DM/MCS tests, compute ALE, generate
outputs). The numbered extension scripts (`scripts/10_*` … `scripts/27_*` — COVID
split, training-size sensitivity, realised-kernel robustness, multiple-testing
correction, path decomposition, HMM regimes, rolling-tree experiment, VaR
back-test, TreeSHAP, selected-λ, and the 2020–2024 VaR significance and
expanding-quantile coverage tests) are run individually; see the docstring at
the top of each.
Everything is driven from [`config/config.yaml`](config/config.yaml) and runs are
seeded (`seed: 42`) for reproducibility.

**Notable implementation choices.** The neural networks run on scikit-learn's
`MLPRegressor` (ReLU, with early stopping standing in for the paper's dropout, as
PyTorch was unavailable). The HAR family rolls daily; the regularised and tree
models are estimated fixed-window for tractability. Every departure from the
paper is documented in [`outputs/LIMITATIONS.md`](outputs/LIMITATIONS.md) and
[`outputs/METHODOLOGY.md`](outputs/METHODOLOGY.md).

---

## Key findings and critique

Where the data allow, the replication agrees with the paper: LogHAR is the
strongest HAR variant, NN ensembles are best at the daily horizon, and
regularisation helps once the extra predictors are added. The substantive
contribution is where it **qualifies** the original — each number below is
reproduced from [`outputs/tables/`](outputs/tables/):

- **Mostly information set + regularisation, not nonlinearity.** A path
  decomposition splits the h = 1 M_ALL gain into a regularisation step (+3–5pp)
  and a deep-NN step (+4–18pp), with the tree step flat-to-negative; a regularised
  linear model captures roughly half the headline.
- **Loss-dependent.** Under QLIKE — a noise-robust loss the paper does not report
  — the edge largely reverses (mean h = 1 ratios vs HAR: LA 1.28, NN2¹⁰ 1.17,
  EN 2.32), and HAR becomes hard to beat.
- **Regime-dependent.** On a 2016–2019 / 2020–2024 split the best-ML edge at h = 1
  collapses from ≈9–14% to ≈0–2% (AAPL −2.1%, AMZN −1.7%), with **no model
  beating HAR on JPM**.
- **The long-horizon tree result depends on daily re-fitting, not the tree class.**
  Fixed-window trees blow up at the monthly horizon (bagging up to ~14× HAR);
  rolling random forests daily roughly halves the gap (AAPL h = 22: 3.04 → 1.55).
- **Statistically thin.** The Model Confidence Set retains 20–21 of 22 models at
  α = 0.10 on every stock (HAR never eliminated); one-sided DM finds only 0–3
  models per stock beat HAR at 5%, and almost nothing survives a Holm/BH
  multiple-testing correction at short horizons.

![Critique evidence: error correlation, horizon divergence, regime collapse](outputs/figures/appendix_evidence_board.png)

The full development, with page/table references to the paper and a verified
number for every claim, is in [`CRITIQUE.md`](CRITIQUE.md).

---

## Limitations

This is an honest, caveated replication. Three stocks (vs the paper's 29), a
post-publication sample, a VIX proxy for licensed OptionMetrics implied
volatility, an incomplete pre-2018 earnings dummy, a scikit-learn NN backend
(ReLU, no dropout), and fixed-window estimation for the regularised/tree models
are all deliberate, disclosed deviations. Each is inventoried with its rationale
and threat-to-validity in [`outputs/LIMITATIONS.md`](outputs/LIMITATIONS.md).

---

## Documentation

| File | Contents |
|------|----------|
| [`REPORT.pdf`](REPORT.pdf) / [`REPORT.md`](REPORT.md) | Three-page write-up: problem, replication, results, critique |
| [`APPENDIX.pdf`](APPENDIX.pdf) / [`APPENDIX.md`](APPENDIX.md) | Full 22-model tables, critique evidence, ALE, code excerpts |
| [`CRITIQUE.md`](CRITIQUE.md) | Deep critique reference, each point anchored to the paper and the data |
| [`outputs/METHODOLOGY.md`](outputs/METHODOLOGY.md) | Equation-by-equation implementation reference, tied to file:line |
| [`outputs/LIMITATIONS.md`](outputs/LIMITATIONS.md) | Every deviation, simplification and threat-to-validity |
| [`outputs/RESULTS_SUMMARY.md`](outputs/RESULTS_SUMMARY.md) | Auto-generated full result tables (MSE, QLIKE, DM, MCS) |
| [`outputs/REPRODUCIBILITY.md`](outputs/REPRODUCIBILITY.md) | How to regenerate every table and figure |

---

## Reference

> Christensen, K., Siggaard, M., and Veliyev, B. (2023). A Machine Learning
> Approach to Volatility Forecasting. *Journal of Financial Econometrics*,
> 21(5), 1680–1727. https://doi.org/10.1093/jjfinec/nbac020

---

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The license
covers this code only — not the underlying paper or any price/macro data, which
are not redistributed.
