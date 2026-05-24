# Critique of Christensen, Siggaard & Veliyev (2023) — deep reference

*"A Machine Learning Approach to Volatility Forecasting", Journal of Financial Econometrics 21(5), 1680–1727 ("CSV").*

This is the full critique behind the report's §4. Every claim is anchored to a
specific page/table in CSV (read directly) and, where possible, to a verified
number from our 2016–2024 replication (`outputs/tables/`). Critiques are grouped
by **mode** — internal contradiction, statistical inference, econometric depth,
design, robustness, reproducibility, framing — and each ends with the
financial, econometric or ML insight it carries.

A note on reading CSV's tables: each cell is the **column** model's MSE relative
to the **row** model, so in the HAR row a value **< 1 means the column model
beats HAR**.

**Claims deliberately NOT made** (they are falsifiable against the paper and
would damage credibility): that "the ML numbers are not shown under the shorter
training split" — Tables A.2/A.4 show every ML model at 1,000- and 2,000-day
training; that the paper "uses QLIKE" — it uses MSE only; that the MCS
statistic/bootstrap is the paper's "stated choice" — it states neither.

---

## Mode 1 — Internal contradictions (the paper's own evidence undercuts its framing)

### C1. "Machine learning beats HAR" is really "a wider feature set, exploited with shrinkage, beats unregularised HAR; nonlinearity adds a smaller increment." [ML]
- **CSV evidence.** On the apples-to-apples M_HAR set (Table 2, h=1, HAR row): only the ensembled NNs beat HAR (best NN3¹⁰ = 0.954, ≈5%); trees *lose* (BG = 1.147); regularisation *ties* (EN = 0.999, RR = 1.000). On M_ALL (Table 3, h=1, HAR row): HAR-X — plain OLS on the wider set — already gets 0.966; Elastic Net 0.916; best NN 0.885. So the step HAR-X→EN (≈5pp) is *regularisation* and EN→NN (≈3pp) is *nonlinearity*.
- **Our evidence.** Path decomposition at h=1 (`critique_path_decomposition_h1.csv`): regularisation contributes +3–5pp, deep-NN +4–18pp, the tree step is flat-to-negative; total ML gain 9–14pp. On M_HAR our pattern matches CSV — only NN ensembles beat HAR, trees lose, regularisation ties.
- **Relevance (ML).** The share of the headline attributable to genuine ML nonlinearity is a minority of the gain; a regularised linear model on more predictors captures much of it. The title's emphasis on "machine learning" overstates a result that is largely "regularisation on a richer information set."

### C2. CSV's own "the basic HAR is superior to its offspring" is split-dependent and quietly indicts the HAR-extension literature. [econometric]
- **CSV evidence.** p.1695, verbatim: *"the basic HAR is superior to its offspring. Compared with the extant literature, this is surprising."* Their Appendix A.1 (Tables A.1–A.4) then shows the ranking *flips* when training is shortened to 1,000 days — with the crisis pushed out of the in-sample window the offspring and ML improve markedly. So the surprising headline is conditional on the 70/10/20 split.
- **Our evidence.** LogHAR is the strongest HAR variant on our sample too (h=1: 0.92–1.00 vs HAR; h=22: 0.63–0.67 on AAPL/JPM), while LevHAR/SHAR/HARQ sit at or above HAR — the same "offspring don't help" pattern; our training-window sensitivity (`trainsize_*`) shows the ranking is split-sensitive, as CSV's own appendix admits.
- **Relevance (econometric).** If established extensions (Patton–Sheppard SHAR, BPQ HARQ, Corsi–Renò LevHAR) underperform plain HAR under the paper's design, that design is not neutral for ranking volatility models; conclusions are path-dependent on the split, which the paper notes only in a footnote.

---

## Mode 2 — Statistical inference is weaker than the prose

### C3. A between-means result is presented as a between-distributions one; the MCS never rejects HAR. [econometric]
- **CSV evidence.** Figure 4: at 90% confidence the MCS retains most models, and **LogHAR is the only HAR variant retained at a high rate** — i.e. a HAR model sits inside the "best set." The DM evidence is reported as vote-counting ("rejected for >50% of stocks"), a heuristic rather than a joint test, and is uncorrected for the ≈136 pairwise comparisons per stock.
- **Our evidence.** MCS at h=1, α=0.10 retains **20–21 of 22 models** on every stock (`mcs_h1_mse_a10.csv`); HAR is never eliminated. One-sided DM finds only **0–3 models per stock** beat HAR at 5% (`dm_h1_mse.csv`); under a Holm/Benjamini–Hochberg correction essentially nothing survives at h=1/h=5 (`dm_multitest_*`).
- **Relevance (econometric).** "ML beats HAR" is a statement about *average* loss. The MCS — the device the paper itself invokes to make a formal claim — keeps HAR in the elite set, i.e. it does *not* establish distinguishable predictive accuracy.

### C4. MSE-only evaluation: the noise-robust loss tells a different story. [econometric / financial]
- **CSV evidence.** The forecast comparison uses MSE only; QLIKE appears nowhere. Patton (2011) shows that with a *noisy* volatility proxy only MSE and QLIKE remain "robust" loss functions, so reporting one without the other leaves the ranking's robustness untested.
- **Our evidence.** Under QLIKE the ML edge largely evaporates: at h=1, HAR-X (2.3–3.9×), Ridge (2.2–4.1×), Elastic Net (1.0–4.4×) and the NNs (≈1.0–1.4×) are all QLIKE-*worse* than HAR on most stocks (`loss_ratio_h1_qlike.csv`). MSE rewards fit on a few high-variance days; QLIKE weights proportional errors and so weights the many low-variance days — and there HAR is hard to beat.
- **Relevance (financial/econometric).** The loss function is not innocuous. The ML advantage is loss-conditional; for applications where proportional accuracy at normal volatility matters, the paper's preferred loss flatters ML.

---

## Mode 3 — Econometric depth

### C5. "ML captures long memory" confuses persistence with long memory, and in-sample fit with out-of-sample forecast. [econometric]
- **CSV evidence.** p.1709 / Figure 8: RF and NN have higher *in-sample fitted* ACF at long lags, read as "better approximating an underlying long-memory structure."
- **Critique.** (i) In-sample fitted persistence ≠ out-of-sample forecast persistence. (ii) Higher ACF is not *correct* ACF: a forecast that over-smooths toward the unconditional mean is also highly autocorrelated *and* mechanically lowers long-horizon MSE, mimicking the same picture. (iii) Long memory is a specific hyperbolic decay rate (a fractional-integration parameter d), not "ACF still positive at lag 200." The decisive test — estimate d on forecast vs realised series, or compare OOS forecast ACF to realised ACF — is absent.
- **Our evidence.** The OOS ACF figure (`critique_oos_acf_h*.pdf`) shows ML forecast ACF can exceed the realised-RV ACF at long lags — consistent with over-smoothing, not with "correctly captured" long memory.
- **Relevance (econometric).** The mechanism the paper credits for its long-horizon gain is asserted, not demonstrated; an over-smoothing explanation fits the same evidence and has opposite economic meaning.

### C6. Generated-regressor / measurement-error asymmetry. [econometric]
- **CSV evidence.** RV is a noisy estimator of integrated variance (Eq. 2); HARQ exists precisely to correct the attenuation this induces in the daily coefficient (BPQ 2016). The ML methods are fed the same noisy RV regressors with no measurement-error correction.
- **Critique.** The horse-race therefore partly rewards whichever function class best fits *estimator noise*, not true conditional variance; the comparison is not on a level measurement footing.
- **Our evidence.** Realised-kernel robustness (`rk_robustness.csv`): replacing 5-min RV with the noise-robust kernel moves JPM h=1 MSE by ≈27% (ratio 0.729) — larger than the entire ML-vs-HAR gap — so a meaningful slice of "model improvement" on noisier names is estimator noise the kernel removes.
- **Relevance (econometric).** Measurement error in target and regressors is first-order here; part of the measured ML gain is attributable to noise a better estimator would eliminate.

---

## Mode 4 — Methodological design

### C7. "Minimal hyperparameter tuning" is asymmetric and favours the methods it crowns. [ML]
- **CSV evidence.** Table A.6: RR/LA/EN search a **1,000-point** λ grid; GB is validation-tuned over depth/trees/learning-rate; the NNs train **100 seeds and keep the best 10** (selection is itself tuning). Bagging/RF use 2004 Breiman–Cutler defaults with no tuning.
- **Critique.** The methods that win (NN, EN) are the most heavily searched; the "off-the-shelf" RF gets the least attention. The stated stance ("we under-tune ML to be conservative for HAR") is the opposite of what the implementation does for NN and the regularised linear models.
- **Relevance (ML).** A like-for-like tuning budget would change the relative standings; the comparison embeds a tuning asymmetry that flatters the chosen winners.

### C8. The estimation scheme confounds model class with re-fit frequency — and our controlled experiment isolates it. [econometric / financial]
- **CSV evidence.** CSV roll HAR, BG, RF and the tuned linear models daily, but estimate the NNs **once** and freeze them (p.1693: "outside our budget", and "strictly disadvantageous to the NNs"). So when RF overtakes the NNs at the monthly horizon (Table 7), part of that gap is "a rolling model beats a frozen one," not "trees beat networks."
- **Our evidence (controlled experiment).** With *all* ML fixed-window, our trees collapse at h=22 (bagging up to ~14× HAR, RF 3–5×). Refitting RF *daily* — CSV's scheme — roughly **halves** the AAPL h=22 ratio (3.04 → 1.55), a large recovery that directly isolates the daily re-fit (not "trees capture long memory") as the load-bearing assumption behind CSV's headline long-horizon result. It does *not* fully reproduce CSV's RF < HAR at h=22, which is consistent with our calmer 3-stock post-COVID test set differing from their 29-stock 2001–2017 sample — itself evidence that the long-horizon tree result is sample-conditional, not structural. (`rolling_vs_fixed_trees.csv`.)
- **Relevance (financial).** A forecaster that needs constant re-estimation to survive is precisely the one that fails at a regime change — the moment risk management most needs it. The paper's most striking finding depends on an operational assumption it never stress-tests.

---

## Mode 5 — Robustness & generalisation (our post-publication sample adds these)

### C9. The edge is regime-conditional. [financial]
- CSV's test set ends in 2017, before COVID and the 2022 tightening. On a 2016–2019-train / 2020–2024-test split (`covidfull_*`, the test spans the COVID crash and 2022) the ML-vs-HAR edge **collapses from ≈9–14% to ≈0–2%** (best ML at h=1: AAPL P-LA −2.1%, AMZN EN −1.7%, JPM none beats HAR — HARQ +0.2%). The headline is conditional on the absence of a structural break in the out-of-sample window — something CSV structurally cannot observe.

### C10. Cross-sectional averaging hides large heterogeneity. [financial]
- CSV report 29-stock means with no cross-stock dispersion. Our three stocks alone span tree MSE ratios of ≈1.6× (GB) to ≈14× (BG) at h=22. A "7% average improvement" can hide a method helping one name and badly hurting another — and tail behaviour per name is exactly what risk management cares about.

### C11. The 22-model space is far narrower than the model count suggests. [ML]
- CSV hold every forecast series but never publish the cross-model error-correlation matrix. Ours (`critique_error_correlation_h1.csv`) shows every model's out-of-sample errors correlate with HAR's at **r ≥ 0.79**, and the eight models that actually beat HAR (NN ensembles, LA, EN, P-LA, LogHAR) at **r ≈ 0.88–0.99**: each winner is HAR plus a small orthogonal perturbation, so its ≈10% MSE reduction lives in **under ~20% of the error variance**. (The lowest pairwise correlation, 0.63, is between A-LA and a single-seed NN — two models that do not beat HAR.) This narrowness also explains why forecast combinations add little.

---

## Mode 6 — Reproducibility / data

### C12. The M_ALL headline leans on a licensed, unverifiable predictor that is redundant once proxied. [reproducibility / financial]
- Table A.5: model-free implied volatility is the **strongest** extra predictor (β = 0.198, t = 6.39), while VIX is **insignificant once IV is included** (t = 0.09). IV is per-stock OptionMetrics — licensed and unreproducible. The standard public proxy (VIX) is market-wide and collinear with the VIX column, so the wider-feature-set gain rests on the one input no external researcher can verify and that, when proxied, adds nothing beyond VIX.

---

## Mode 7 — Framing & scope

### C13. The economic VaR application is decoupled and weakly discriminating; scope is narrow. [financial]
- Filtered historical simulation re-attaches a *fixed* empirical residual quantile to each variance forecast; once standardised residuals are roughly i.i.d., coverage tests pass for any reasonable σ̂ — which is why CSV themselves find the VaR differences "smaller and less statistically significant" (p.1681). A genuinely discriminating economic test (direct quantile regression on RV, or the P&L of a volatility-timing strategy) is not run. Scope is also one asset class (US mega-cap equities, survivorship-tilted), one frequency, and point forecasts only.

---

## How the report uses this

§4 of the report (the heaviest section, per the marking emphasis on critique) leads with the four strongest, each tied to a perspective: **C1** (decomposition — ML), **C3** (MCS permissiveness — econometric), **C8** (the controlled rolling experiment — econometric/financial), **C4** (loss-conditional edge — financial). C2, C5, C6, C7, C9–C13 are developed here in the appendix. Every number is reproduced from `outputs/tables/`; every paper claim carries a page/table reference.
