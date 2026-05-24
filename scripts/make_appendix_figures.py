"""Build a single, journal-grade 'critique evidence board' for the appendix:
(left) the 22-model OOS error-correlation heatmap; (top-right) the horizon
divergence — trees explode at h=22 while LogHAR/NN stay low; (bottom-right) the
COVID regime collapse of the ML edge. Serif type, 300 dpi."""
from __future__ import annotations
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","Times","DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False, "legend.fontsize": 7.5,
})

ABBR = {"HAR":"HAR","LogHAR":"LogH","LevHAR":"LevH","SHAR":"SHAR","HARQ":"HARQ","HAR-X":"HAR-X",
 "RR":"RR","LA":"LA","EN":"EN","P-LA":"P-LA","A-LA":"A-LA","BG":"BG","RF":"RF","GB":"GB",
 "NN1_ensemble":"NN1e","NN1_top1":"NN1t","NN2_ensemble":"NN2e","NN2_top1":"NN2t",
 "NN3_ensemble":"NN3e","NN3_top1":"NN3t","NN4_ensemble":"NN4e","NN4_top1":"NN4t"}

fig = plt.figure(figsize=(11.0, 5.2))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1],
                       wspace=0.28, hspace=0.55)

# ---- Panel A: error-correlation heatmap ----
axA = fig.add_subplot(gs[:, 0])
ec = pd.read_csv("outputs/tables/critique_error_correlation_h1.csv", index_col=0)
labels = [ABBR.get(c, c) for c in ec.columns]
im = axA.imshow(ec.values, cmap="RdYlBu_r", norm=TwoSlopeNorm(vmin=0.55, vcenter=0.85, vmax=1.0))
axA.set_xticks(range(len(labels))); axA.set_xticklabels(labels, rotation=90, fontsize=5.6)
axA.set_yticks(range(len(labels))); axA.set_yticklabels(labels, fontsize=5.6)
axA.set_title("(a) Pairwise OOS forecast-error correlation (h=1)")
axA.tick_params(length=0)
for sp in axA.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6.5)
cb.outline.set_linewidth(0.5)

# ---- Panel B: horizon divergence (mean over stocks) ----
axB = fig.add_subplot(gs[0, 1])
hs = [1, 5, 22]
series = {"LogHAR":"#23507a","RF":"#2f8f4e","BG":"#c0504d","NN3_ensemble":"#9467bd"}
disp = {"LogHAR":"LogHAR","RF":"RF","BG":"BG","NN3_ensemble":"NN3$^{10}$"}
for m,c in series.items():
    ys=[pd.read_csv(f"outputs/tables/loss_ratio_h{h}_mse.csv",index_col=0)[m].mean() for h in hs]
    axB.plot(hs, ys, marker="o", ms=4, lw=1.3, color=c, label=disp[m])
axB.axhline(1.0, color="0.15", lw=1.0, ls=(0,(5,2)))
axB.set_yscale("log"); axB.set_yticks([0.7,1,2,5,10]); axB.set_yticklabels(["0.7","1","2","5","10"])
axB.set_xticks(hs); axB.set_xlabel("forecast horizon h (days)"); axB.set_ylabel("MSE / HAR (log)")
axB.set_title("(b) Horizon divergence: fixed-window trees explode at h=22")
axB.grid(axis="y", color="0.88", lw=0.5); axB.set_axisbelow(True)
axB.legend(ncol=2, loc="upper left", columnspacing=1.0, handlelength=1.4)

# ---- Panel C: COVID regime collapse ----
axC = fig.add_subplot(gs[1, 1])
std=pd.read_csv("outputs/tables/loss_ratio_h1_mse.csv",index_col=0)
cov=pd.read_csv("outputs/tables/covidfull_loss_ratio_h1_mse.csv",index_col=0)
stocks=["AAPL","AMZN","JPM"]
def edge(df,tk):
    r=df.loc[tk].drop(labels=[c for c in ["HAR"] if c in df.columns]); return max((1-r.min())*100,0)
se=[edge(std,t) for t in stocks]; ce=[edge(cov,t) for t in stocks]
x=np.arange(len(stocks)); w=0.38
axC.bar(x-w/2, se, w, color="#23507a", label="standard test (2023–24)", edgecolor="white", lw=0.4)
axC.bar(x+w/2, ce, w, color="#c0504d", label="COVID split (2020–24)", edgecolor="white", lw=0.4)
for xi,v in zip(x-w/2,se): axC.text(xi,v+0.2,f"{v:.0f}%",ha="center",va="bottom",fontsize=7)
for xi,v in zip(x+w/2,ce): axC.text(xi,v+0.2,f"{v:.0f}%",ha="center",va="bottom",fontsize=7)
axC.set_xticks(x); axC.set_xticklabels(stocks); axC.set_ylabel("best-ML edge over HAR (%)")
axC.set_ylim(0, max(se)*1.25); axC.set_title("(c) The ML edge collapses under a regime shift")
axC.grid(axis="y", color="0.88", lw=0.5); axC.set_axisbelow(True)
axC.legend(loc="upper right", handlelength=1.2)

fig.savefig("outputs/figures/appendix_evidence_board.png")
print("Built appendix evidence board (3 panels, 300 dpi).")
