"""
Render the 3-page report to PDF with ReportLab Platypus (project-preferred
engine for text-heavy documents). Main text (sections 1-4 + one table + one
figure) is kept within 3 pages; references and the appendix follow after a
page break and are excluded from the 3-page limit.
"""
from __future__ import annotations
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "REPORT.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading2"], fontName="Times-Bold",
                    fontSize=11, spaceBefore=6, spaceAfter=2, leading=13)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontName="Times-Roman",
                      fontSize=9.6, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=4)
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontName="Times-Bold",
                       fontSize=14, alignment=TA_CENTER, spaceAfter=2, leading=16)
META = ParagraphStyle("META", parent=styles["Normal"], fontName="Times-Italic",
                      fontSize=8.5, alignment=TA_CENTER, spaceAfter=6, leading=10)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, leading=9.6)
CAP = ParagraphStyle("CAP", parent=SMALL, fontName="Times-Italic", alignment=TA_CENTER)

def P(t): return Paragraph(t, BODY)

story = []
story.append(Paragraph("A Machine Learning Approach to Volatility Forecasting", TITLE))
story.append(Paragraph("Replication and critical discussion of Christensen, Siggaard &amp; Veliyev (2023, "
                       "<i>Journal of Financial Econometrics</i> 21(5)) — AAPL, AMZN, JPM 5-minute data, 2016–2024.", META))

# --- 1 ---
story.append(Paragraph("1. The problem and the original paper", H1))
story.append(P("Realised variance (RV), the sum of squared intraday returns, is the ex-post measure of a day's "
  "return variation (Andersen &amp; Bollerslev 1998) and the input to risk management, option pricing and "
  "allocation; unlike returns it is strongly predictable because volatility is persistent. The field benchmark "
  "is the Heterogeneous Autoregressive model (HAR; Corsi 2009): a linear regression of RV on its daily, weekly "
  "(5-day) and monthly (22-day) lagged averages — four parameters that approximate the long memory of volatility "
  "so well that decades of richer models struggle to beat it (Hansen &amp; Lunde 2005)."))
story.append(P("Christensen, Siggaard &amp; Veliyev (2023, “CSV”) ask whether machine learning (ML) beats HAR, and "
  "where any gain comes from. They run a 22-model horse-race — the HAR family (HAR, LogHAR, LevHAR, SHAR, HARQ), "
  "regularised regression (Ridge, Lasso, Elastic Net, adaptive- and post-Lasso), tree ensembles (bagging, random "
  "forest, gradient boosting) and feed-forward neural nets (four geometric-pyramid architectures, each a single net "
  "NN<super>1</super> and a 10-of-100 ensemble NN<super>10</super>) — over two information sets: M_HAR (three RV lags) and M_ALL "
  "(plus nine macro/firm predictors). On 29 Dow Jones stocks, 2001–2017, they conclude ML beats HAR; the gain is "
  "small and class-specific on M_HAR but clear on M_ALL (Elastic Net ≈−8%, random forest ≈−10%, "
  "best NN ≈−11% at h=1), grows with horizon, and is led by random forests at one month "
  "(≈40% MSE reduction). Significance uses Diebold–Mariano (DM; 1995) and the Hansen–Lunde–Nason Model "
  "Confidence Set (MCS; 2011); interpretability uses Accumulated Local Effects (Apley &amp; Zhu 2020)."))
story.append(P("We replicate the full methodology on the three stocks with available 5-minute data over 2016–2024 — "
  "seven years later than CSV, spanning the COVID shock and 2022 regime. AAPL and JPM are in CSV's cross-section "
  "(a check); AMZN joined the DJIA only in 2020, giving out-of-sample-stock evidence. We target a "
  "sound, transparent implementation and a critical comparison, not exact numbers."))

# --- 2 ---
story.append(Paragraph("2. Implementation and replication", H1))
story.append(P("<b>Data.</b> From 1-minute bars we keep the 09:30–16:00 session, apply a Barndorff-Nielsen et al. "
  "(2009)-style outlier filter, resample to 5-minute returns within each day (no overnight returns), and compute "
  "RV, signed semivariances and realised quarticity. The split is 70/10/20 chronological, as CSV. The daily lag "
  "RV<sub>t-1</sub> predicts RV<sub>t</sub> (a true one-step forecast)."))
story.append(P("<b>Models.</b> All 22 CSV models sit behind one interface. HAR is closed-form OLS; LogHAR carries "
  "the Jensen back-transform; HARQ uses the Bollerslev–Patton–Quaedvlieg (2016) insanity filter (clip to the "
  "in-sample range). Regularised models use a 100-point log-λ grid (CSV 1,000; the optimum is interior, so this "
  "is immaterial). Bagging/RF use Breiman–Cutler defaults; gradient boosting is validation-tuned. Following CSV "
  "(p.1691) any negative variance forecast is replaced by the in-sample minimum RV, applied uniformly."))
story.append(P("<b>Honest deviations.</b> The NNs run on scikit-learn (PyTorch unavailable): ReLU not leaky-ReLU and "
  "no dropout, so early stopping is the regulariser in dropout's place. For tractability the regularised, tree and "
  "NN models are fixed-window; CSV roll the regularised models and trees and fix only the NNs — our fixed-window "
  "trees are the main deviation, exploited in §4. Implied volatility is proxied by VIX (CSV use licensed "
  "OptionMetrics). <b>Evaluation:</b> MSE (primary, as CSV) plus QLIKE (Patton 2011) as a noise-robust complement; "
  "DM with Newey–West HAC and the Harvey–Leybourne–Newbold correction; the HLN MCS (10,000 moving-block bootstraps) "
  "at α∈{0.25,0.10,0.05}; ALE for variable importance. Thirty-seven unit tests cover the critical path."))

# --- 3 ---
story.append(Paragraph("3. Results", H1))
story.append(P("We select three results as most important — the M_ALL comparison at h=1, its behaviour across "
  "horizons, and the significance picture — because these are CSV's three headline claims."))
# table
_hd = ParagraphStyle("hd", fontName="Times-Bold", fontSize=8, alignment=TA_CENTER, leading=9)
tbl_data = [
 [Paragraph(x, _hd) for x in ["h","Stock","LogHAR","EN","LA","RF","BG","NN2<super>10</super>","NN3<super>10</super>"]],
 ["1","AAPL","0.944","0.953","0.941","1.069","1.167","0.891","0.938"],
 ["1","AMZN","1.000","0.970","0.881","0.963","1.140","0.879","0.861"],
 ["1","JPM","0.924","0.953","0.951","0.952","1.054","0.912","0.950"],
 ["22","AAPL","0.666","1.031","1.021","3.038","7.314","1.009","0.926"],
 ["22","AMZN","1.214","1.085","0.901","3.885","6.719","1.129","1.245"],
 ["22","JPM","0.630","1.012","0.976","5.274","13.835","0.860","0.716"],
]
t = Table(tbl_data, hAlign="LEFT", colWidths=[0.7*cm,1.1*cm]+[1.7*cm]*7)
t.setStyle(TableStyle([
 ("FONT",(0,0),(-1,-1),"Times-Roman",8),
 ("FONT",(0,0),(-1,0),"Times-Bold",8),
 ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
 ("LINEBELOW",(0,0),(-1,0),0.6,colors.black),
 ("LINEBELOW",(0,3),(-1,3),0.4,colors.grey),
 ("ALIGN",(2,1),(-1,-1),"CENTER"),
 ("TOPPADDING",(0,0),(-1,-1),1.2),("BOTTOMPADDING",(0,0),(-1,-1),1.2),
]))
story.append(t)
story.append(Paragraph("Table 1. Out-of-sample MSE relative to HAR, M_ALL (ratio &lt; 1 beats HAR). Full 22-model "
  "tables, DM stars and MCS sets in the Appendix.", CAP))
story.append(Spacer(1,3))
story.append(P("<b>At h=1 on M_ALL</b> the best model beats HAR by ≈9–14% on every stock (AAPL 10.9%, AMZN 13.9%, JPM 8.8%) — NN ensembles lead "
  "(0.86–0.91), Lasso close (0.88–0.95) — reproducing CSV's magnitude (≈11%) and ordering (NNs best, "
  "regularisation helping, bagging weakest). On the apples-to-apples M_HAR set the gain shrinks to a few percent "
  "and only the NN ensembles beat HAR, with trees losing — again matching CSV."))
img = ROOT/"outputs/figures/headline_h1_ratios.png"
if img.exists():
    story.append(Image(str(img), width=14.6*cm, height=6.3*cm))
    story.append(Paragraph("Figure 1. M_ALL MSE ratios vs HAR at h=1; bars below the dashed HAR line beat HAR.", CAP))
story.append(P("<b>Across horizons</b> the tree picture diverges sharply and is our most informative result. CSV "
  "find ML's edge grows with horizon, RF leading at h=22; we find the opposite for fixed-window trees — at h=22 "
  "bagging is 7–14× HAR and RF 3–5× — while LogHAR dominates the HAR family (0.63–0.67 on AAPL/JPM) and "
  "the NN ensembles stay competitive. <b>Significance is thin:</b> at h=1 the MCS at 90% retains 20–21 of 22 models "
  "on every stock and never eliminates HAR; one-sided DM finds only 0–3 models per stock beat HAR at 5%, and under "
  "a Holm/BH multiple-testing correction almost nothing survives. Under QLIKE the ML advantage largely disappears."))

# --- 4 ---
story.append(Paragraph("4. Critical comparison and discussion", H1))
story.append(P("The replication matches CSV where the data allow (LogHAR strongest in the HAR family, NN ensembles "
  "best at h=1, regularisation helping on M_ALL), so the substantive contribution is the critique. Five points, each "
  "tied to a financial, econometric or ML reason; full development with page/table references in the Appendix."))
story.append(P("<b>(i) The “machine learning” headline is mostly information set and regularisation, not "
  "nonlinearity [ML].</b> CSV's own tables separate the effects their title bundles. On M_HAR (Table 2, h=1) only "
  "the NN ensembles beat HAR, by ≈5%; trees lose (bagging +15%) and regularisation ties (EN 0.999). The gain "
  "appears only on M_ALL, where plain OLS on the wider set (HAR-X) already earns 3.4%, Elastic Net 8.4% and the best "
  "NN 11.5% — regularisation does roughly half the work, nonlinearity the rest. Our path decomposition reproduces "
  "the split (regularisation +3–5pp, deep-NN +4–18pp, trees flat-to-negative). “Regularisation on a richer "
  "information set” would describe the result more honestly than “machine learning”."))
story.append(P("<b>(ii) CSV's long-horizon tree result depends on the daily re-fit, not the model class "
  "[econometric/financial].</b> CSV roll bagging/RF daily but fix the NNs once (p.1693, conceded “strictly "
  "disadvantageous to the NNs”), so “RF best at h=22” (Table 7, ≈40% reduction) confounds "
  "<i>trees</i> with <i>re-estimated daily</i>. We isolate the two: with all ML fixed-window the trees collapse at "
  "h=22; refitting RF daily — CSV's scheme — roughly halves the AAPL ratio (3.04→1.55), pinning the daily re-fit, "
  "not “trees capture long memory”, as the load-bearing assumption (it does not fully reach CSV's RF&lt;HAR, "
  "evidence the result is also sample-conditional). A model needing constant re-estimation to survive is the one that "
  "fails at a regime change, when forecasts matter most."))
story.append(P("<b>(iii) The significance machinery does not support the prose [econometric].</b> CSV's claim is a "
  "between-means statement (lower average MSE via DM). The MCS — the test of <i>distinguishable</i> accuracy — keeps "
  "HAR in the best set: their Figure 4 retains LogHAR at a high rate, and ours retains 20–21 of 22 models at "
  "α=0.10 with HAR never eliminated. DM finds 0–3 models per stock beat HAR at 5%, and a multiple-testing "
  "correction (CSV apply none, despite ≈136 pairwise tests per stock) removes essentially every h=1/h=5 star. "
  "ML lowers average MSE a few percent but is not statistically distinguishable from HAR at the daily horizon."))
story.append(P("<b>(iv) The edge is loss- and regime-conditional [financial].</b> CSV evaluate on MSE only; under "
  "QLIKE (Patton's noise-robust loss) the advantage largely vanishes — HAR-X, Ridge, Elastic Net and the NNs are all "
  "QLIKE-worse than HAR on most stocks at h=1. MSE rewards fit on a few high-variance days; QLIKE weights the many "
  "ordinary days, where HAR is hard to beat. And on a 2016–2019-train / 2020–2024-test split (spanning the COVID "
  "crash and 2022) the edge collapses from ≈9–14% to ≈0–2% — no model beats HAR on JPM. Since proportional "
  "accuracy and turbulent regimes are what risk management cares about, the practical value of the edge is smaller "
  "than an MSE-only, calm-period comparison implies."))
fig2 = ROOT/"outputs/figures/mse_vs_qlike_h1.png"
if fig2.exists():
    story.append(Image(str(fig2), width=13.6*cm, height=5.5*cm))
    story.append(Paragraph("Figure 2. Models that beat HAR under MSE (left bars &lt; 1) are mostly worse under the "
      "noise-robust QLIKE loss (right bars &gt; 1); only LogHAR is robust to both. h=1, M_ALL, mean over three stocks.", CAP))
story.append(P("<b>(v) Long-memory interpretation and licensed-data dependence [econometric/reproducibility].</b> CSV "
  "credit “better approximating long memory”, but their evidence is <i>in-sample fitted</i> ACF (Figure 8), "
  "which cannot distinguish captured long memory from over-smoothing toward the unconditional mean — our "
  "<i>out-of-sample</i> ACF shows ML forecasts more persistent than realised RV, the over-smoothing signature. And "
  "the M_ALL gain leans on per-stock OptionMetrics implied volatility, the strongest extra predictor (Table A.5, "
  "t=6.4) yet licensed, unreproducible, and redundant once proxied by VIX (insignificant alongside IV, t=0.09). "
  "Where we match CSV the agreement is reassuring; where we differ, the difference is the contribution — enabled by a "
  "regime-spanning sample CSV could not observe."))
story.append(P("<b>Further critiques (full development in <font face='Courier'>CRITIQUE.md</font>).</b> "
  "<b>(vi)</b> The comparison runs on a noisy proxy: RV estimates integrated variance with error and HARQ exists to "
  "correct it (BPQ 2016), yet the ML methods receive the same noisy regressors uncorrected — switching to the "
  "noise-robust realised kernel moves JPM h=1 MSE by ≈27%, larger than the entire ML-vs-HAR gap, so part of the "
  "“gain” is estimator noise, not model skill [econometric]. <b>(vii)</b> “Minimal tuning” is asymmetric: CSV "
  "search a 1,000-point λ grid and keep the best 10 of 100 NN seeds but leave RF at 2004 defaults — the methods that "
  "win are the most heavily searched [ML]. <b>(viii)</b> The 22-model space is narrow: every model's out-of-sample "
  "errors correlate with HAR at r ≥ 0.79 (≈0.88–0.99 for the models that beat HAR), so each winner is HAR plus a small "
  "orthogonal perturbation and its gain lives in under ~20% of error variance — which is also why combinations add "
  "little [ML]. <b>(ix)</b> “The basic "
  "HAR is superior to its offspring” (CSV p.1695) is split-dependent: their own Appendix A.1 flips it under a shorter "
  "window, so the headline ranking is conditional on the train/test design rather than structural [econometric]."))

# --- references + appendix (after page break; excluded from 3-page limit) ---
story.append(PageBreak())
story.append(Paragraph("References", H1))
refs = ("Andersen &amp; Bollerslev (1998, <i>Int. Econ. Rev.</i>); Apley &amp; Zhu (2020, <i>JRSS-B</i>); "
 "Barndorff-Nielsen &amp; Shephard (2002, <i>JRSS-B</i>); Barndorff-Nielsen, Hansen, Lunde &amp; Shephard (2008, "
 "<i>Econometrica</i>); Bollerslev, Patton &amp; Quaedvlieg (2016, <i>J. Econometrics</i>); Christensen, Siggaard "
 "&amp; Veliyev (2023, <i>J. Financial Econometrics</i> 21(5), 1680–1727); Corsi (2009, <i>J. Financial "
 "Econometrics</i>); Diebold &amp; Mariano (1995, <i>JBES</i>); Hansen &amp; Lunde (2005, <i>J. Applied "
 "Econometrics</i>); Hansen, Lunde &amp; Nason (2011, <i>Econometrica</i>); Patton (2011, <i>J. Econometrics</i>); "
 "Patton &amp; Sheppard (2015, <i>Rev. Econ. Stat.</i>); Zou (2006, <i>JASA</i>).")
story.append(Paragraph(refs, SMALL))
story.append(Paragraph("Appendix (≤3 pages — full content in APPENDIX.md)", H1))
story.append(Paragraph("A1 full 22-model MSE tables (M_ALL h=1 &amp; h=22); A2 the apples-to-apples M_HAR table; "
 "A3 critique evidence (path decomposition, fixed-vs-rolling tree experiment, MCS/DM significance, QLIKE reversal, "
 "realised-kernel, error correlation); A4 ALE variable importance; A5 code excerpts; A6 data construction and "
 "disclosed deviations. Set the wide A1 tables landscape (as CSV do). Machine-readable tables in "
 "<font face='Courier'>outputs/</font>; full critique in <font face='Courier'>CRITIQUE.md</font>.", SMALL))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=1.5*cm, bottomMargin=1.4*cm,
                        leftMargin=1.7*cm, rightMargin=1.7*cm,
                        title="ML Approach to Volatility Forecasting — Replication")
doc.build(story)
print("Built", OUT)
