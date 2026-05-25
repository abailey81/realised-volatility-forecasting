"""Render the curated appendix (APPENDIX.md content) to a landscape PDF to
verify it fits within the 3-page appendix limit. Landscape A4 is used because
the A1 tables are 18 columns wide (CSV themselves set their wide tables
landscape)."""
from __future__ import annotations
from pathlib import Path
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "APPENDIX.pdf"
styles = getSampleStyleSheet()
H = ParagraphStyle("H", fontName="Times-Bold", fontSize=10, spaceBefore=6, spaceAfter=2, leading=12)
B = ParagraphStyle("B", fontName="Times-Roman", fontSize=8.5, leading=10.3, alignment=TA_JUSTIFY, spaceAfter=3)
C = ParagraphStyle("C", fontName="Courier", fontSize=7.2, leading=8.4, spaceAfter=3)
CAP = ParagraphStyle("CAP", fontName="Times-Italic", fontSize=7.5, leading=9, spaceAfter=4)
hdr = ParagraphStyle("hdr", fontName="Times-Bold", fontSize=7, alignment=1, leading=8)

def tbl(data, w):
    t = Table(data, colWidths=w, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT",(0,0),(-1,-1),"Times-Roman",7),
        ("FONT",(0,0),(-1,0),"Times-Bold",7),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("LINEBELOW",(0,0),(-1,0),0.5,colors.black),
        ("ALIGN",(1,1),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1)]))
    return t

MODELS=["LogHAR","LevHAR","SHAR","HARQ","HAR-X","RR","LA","EN","P-LA","A-LA","BG","RF","GB","NN1","NN2","NN3","NN4"]
A1h1=[["AAPL",".944","1.026","1.004",".985","1.015",".987",".941",".953",".929","1.991","1.167","1.069","1.002",".920",".891",".938",".921"],
      ["AMZN","1.000","1.065",".991",".992",".951",".951",".881",".970",".959","1.525","1.140",".963",".978",".869",".879",".861",".941"],
      ["JPM",".924","1.049","1.177","1.163","1.119","1.093",".951",".953",".979","2.169","1.054",".952",".955",".977",".912",".950",".944"]]
A1h22=[["AAPL",".666",".842","1.008","1.025","1.123","1.079","1.021","1.031",".921","1.638","7.314","3.038","2.677","1.067","1.009",".926",".814"],
       ["AMZN","1.214",".949",".996","1.161","1.141","1.114",".901","1.085","1.115","1.410","6.719","3.885","4.069","1.266","1.129","1.245","1.364"],
       ["JPM",".630","1.112",".988","1.476","1.494","1.481",".976","1.012",".721","1.694","13.835","5.274","1.625",".901",".860",".716","1.309"]]
A2=[["AAPL",".944",".986",".992","1.023","1.021","1.003","1.578","1.406","1.036",".997",".978",".976"],
    ["AMZN","1.003",".997",".991",".993",".995",".995","1.420","1.260","1.126","1.046","1.064","1.042"],
    ["JPM",".916","1.161",".984",".995",".994",".977","1.363","1.231","1.088",".999",".995",".985"]]
A2cols=["LogHAR","HARQ","RR","LA","EN","P-LA","BG","RF","GB","NN2","NN3","NN4"]

s=[]
s.append(Paragraph("Appendix — full results, robustness and code (companion to the 3-page report)", H))
s.append(Paragraph("Out-of-sample on the 2016–2024 test set; ratios are MSE relative to HAR (&lt;1 beats HAR). "
  "Machine-readable tables in <font face='Courier'>outputs/</font>; full critique in <font face='Courier'>CRITIQUE.md</font>.", B))

s.append(Paragraph("A1. Full 22-model out-of-sample MSE ratios vs HAR — M_ALL, h=1 (NN columns are the 10-net ensembles)", H))
w=[1.3*cm]+[1.45*cm]*17
s.append(tbl([["Stock"]+MODELS]+A1h1, w))
s.append(Paragraph("A1 (cont.) — M_ALL, h=22 (note the fixed-window tree collapse — A3)", H))
s.append(tbl([["Stock"]+MODELS]+A1h22, w))
s.append(Spacer(1,3))
s.append(Paragraph("A2. Apples-to-apples M_HAR (3 RV lags only), h=1 — only NN ensembles/LogHAR reach HAR; trees lose 4–58%; RR/LA/EN/P-LA tie", H))
s.append(tbl([["Stock"]+A2cols]+A2, [1.3*cm]+[1.6*cm]*12))
s.append(Spacer(1,3))

s.append(Paragraph("A3. Critique evidence (verified; full development in CRITIQUE.md)", H))
for t in [
 "<b>Path decomposition, h=1 (HAR→best NN).</b> regularisation +4.7/3.0/4.7pp (AAPL/AMZN/JPM); tree marginal −11.6/+0.7/+0.1pp; deep-NN +17.8/+10.2/+4.0pp; total ML gain 10.9/13.9/8.8pp.",
 "<b>Fixed- vs rolling-window trees (AAPL h=22).</b> RF MSE/HAR = 3.04 fixed → 1.55 rolling: the daily re-fit, not the tree class, drives CSV's long-horizon result.",
 "<b>Significance.</b> MCS (90%) retains 20–21/22 models per stock at h=1, HAR never eliminated; DM finds 0–3 models/stock beating HAR at 5%; under Holm/BH essentially none survive at h=1/h=5.",
 "<b>Loss-conditional (QLIKE, h=1, mean over stocks).</b> LogHAR 0.96, RF 0.99, but LA 1.28, NN3 1.06, NN2 1.17, EN 2.32 — the MSE edge largely reverses under the noise-robust loss.",
 "<b>Measurement error (realised kernel vs 5-min RV, JPM h=1).</b> RK/RV MSE = 0.729 (−27%) — larger than the entire ML-vs-HAR gap.",
 "<b>Error correlation.</b> Every model correlates with HAR's errors at r≥0.79 (winners ≈0.88–0.99): each winner is HAR + a small orthogonal perturbation (&lt;~20% of error variance).",
 "<b>Regime conditionality (COVID split 2016–2019/2020–2024).</b> Best-ML edge collapses from ≈9–14% to ≈0–2% (AAPL P-LA −2.1%, AMZN EN −1.7%, JPM none beats HAR)."]:
    s.append(Paragraph("• "+t, B))

s.append(Paragraph("A4. Variable importance (ALE), AAPL h=1 — reproduces CSV Fig 7: RVD/RVW dominate; IV/VIX weighted higher by RF/NN; EA≈0", H))
vi=[["feature","HAR-X","EN","RF","NN2"],["RVD",".316",".382",".204",".040"],["RVW",".250",".357",".183",".186"],
    ["IV",".106",".191",".214",".232"],["VIX",".106",".000",".230",".146"],["EA",".000",".000",".000",".000"]]
s.append(tbl(vi,[2.2*cm]+[2.0*cm]*4))
s.append(Spacer(1,3))

s.append(Paragraph("A5. Selected code excerpts (methodological fidelity)", H))
s.append(Paragraph("Horizon target (one-step alignment RV_{t-1}→RV_t):", B))
s.append(Paragraph("def make_horizon_target(rv, h):<br/>&nbsp;&nbsp;return pd.concat([rv.shift(-k) for k in range(h)],axis=1).mean(axis=1).rename('y')", C))
s.append(Paragraph("HARQ insanity filter (clip to in-sample range, BPQ 2016); positivity filter (CSV p.1691):", B))
s.append(Paragraph("return np.clip(raw, self.train_min_, self.train_max_)<br/>"
  "floor=float(train['y'].min()); pred = pred.clip(lower=floor)  # every model", C))
s.append(Paragraph("MCS T_max with moving-block bootstrap (Hansen–Lunde–Nason 2011):", B))
s.append(Paragraph("d_i=Lbar-Lbar.mean(); t=d_i/sd; boot_t=(boot_d-d_i)/sd<br/>"
  "pval=np.mean(boot_t.max(axis=1) &gt;= t.max())  # eliminate argmax if pval&lt;=alpha", C))

s.append(Paragraph("A6. Data &amp; disclosed deviations", H))
s.append(Paragraph("5-min RV within-day (no overnight), BNHLS outlier filter, RV/RV±/RQ; 70/10/20 chronological split. "
  "Deviations: sklearn NN (ReLU, no dropout; early stopping in dropout's place); regularised+trees fixed-window "
  "(HAR &amp; NN match CSV); IV proxied by VIX; 100-point λ grid (resolution immaterial, but several selected "
  "penalties sit at a grid bound — see selected_lambda.csv). 37 unit tests pass; "
  "<font face='Courier'>python scripts/00_run_full_pipeline.py</font>.", B))

_board = ROOT/"outputs/figures/appendix_evidence_board.png"
if _board.exists():
    s.append(Paragraph("A7. Critique evidence board", H))
    s.append(Image(str(_board), width=25.8*cm, height=12.2*cm))
    s.append(Paragraph("(a) Out-of-sample error correlation across the 22 models — every model correlates with HAR at "
      "r≥0.79; only A-LA and the single-seed NNs diverge, so the model space is far narrower than the count suggests. "
      "(b) Horizon divergence — fixed-window trees (BG, RF) explode to 5–9× HAR at h=22 (log axis) while LogHAR and "
      "NN3 stay below HAR; the paper avoids this by rolling BG/RF daily. (c) The best-ML edge over HAR collapses "
      "from 9–14% on the standard 2023–24 test to 0–2% on the 2016–2019 / 2020–2024 regime-spanning split.", CAP))

doc=SimpleDocTemplate(str(OUT), pagesize=landscape(A4), topMargin=1.1*cm, bottomMargin=1.0*cm,
                      leftMargin=1.2*cm, rightMargin=1.2*cm, title="Appendix")
doc.build(s)
print("Built", OUT)
