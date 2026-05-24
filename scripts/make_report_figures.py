"""Build the two headline report figures to a clean, journal-quality standard:
serif type matching the report, de-spined axes, a shaded 'beats-HAR' band, a
colour-blind-safe palette, and 300 dpi. Regenerates the PNGs embedded in the PDF."""
from __future__ import annotations
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False, "legend.fontsize": 8.5,
})
STOCK_C = {"AAPL": "#23507a", "AMZN": "#7fb8e0", "JPM": "#2f8f4e"}  # cb-safe-ish

def beats_band(ax, lo):
    ax.axhspan(lo, 1.0, color="#2f8f4e", alpha=0.05, zorder=0)   # beats HAR (green tint)
    ax.axhline(1.0, color="0.15", lw=1.1, ls=(0,(5,2)), zorder=3)
    ax.text(0.004, 1.0, " HAR", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=7.5, color="0.3")

# ---- Figure 1: headline MSE ratios, M_ALL, h=1 ----
d = pd.read_csv("outputs/tables/loss_ratio_h1_mse.csv", index_col=0)
models = ["LogHAR","EN","LA","RF","BG","NN2_ensemble","NN3_ensemble"]
labels = ["LogHAR","EN","LA","RF","BG","NN2$^{10}$","NN3$^{10}$"]
stocks = list(d.index); x = np.arange(len(models)); w = 0.25
fig, ax = plt.subplots(figsize=(7.0, 3.0))
lo = 0.80; beats_band(ax, lo)
for i, tk in enumerate(stocks):
    ax.bar(x + (i-1)*w, [d.loc[tk, m] for m in models], w, label=tk,
           color=STOCK_C.get(tk, "#888"), edgecolor="white", linewidth=0.4, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("out-of-sample MSE / HAR")
ax.set_ylim(lo, 1.22); ax.set_yticks([0.8,0.9,1.0,1.1,1.2])
ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10),
          columnspacing=1.6, handlelength=1.1, handletextpad=0.5)
fig.savefig("outputs/figures/headline_h1_ratios.png")
plt.close(fig)

# ---- Figure 2: MSE vs QLIKE reversal ----
mse = pd.read_csv("outputs/tables/loss_ratio_h1_mse.csv", index_col=0)
ql  = pd.read_csv("outputs/tables/loss_ratio_h1_qlike.csv", index_col=0)
m2 = ["LogHAR","EN","LA","RF","NN2_ensemble","NN3_ensemble"]
l2 = ["LogHAR","EN","LA","RF","NN2$^{10}$","NN3$^{10}$"]
mse_m = [mse[m].mean() for m in m2]; ql_m = [ql[m].mean() for m in m2]
x = np.arange(len(m2)); w = 0.40
fig, ax = plt.subplots(figsize=(7.0, 3.0))
lo = 0.80; beats_band(ax, lo)
b1 = ax.bar(x - w/2, mse_m, w, label="MSE",   color="#23507a", edgecolor="white", linewidth=0.4, zorder=2)
b2 = ax.bar(x + w/2, ql_m,  w, label="QLIKE", color="#c0504d", edgecolor="white", linewidth=0.4, zorder=2)
# annotate the QLIKE bars that breach HAR (the reversal)
for xi, v in zip(x + w/2, ql_m):
    if v > 1.04:
        ax.text(xi, min(v, 2.35)+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#c0504d")
ax.set_xticks(x); ax.set_xticklabels(l2)
ax.set_ylabel("loss / HAR  (mean over 3 stocks)")
ax.set_ylim(lo, 2.45); ax.set_yticks([0.8,1.0,1.5,2.0])
ax.grid(axis="y", color="0.85", lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper right", handlelength=1.1, handletextpad=0.5)
fig.savefig("outputs/figures/mse_vs_qlike_h1.png")
plt.close(fig)
print("Built both figures at 300 dpi.")
print(f"  installed serif: {'Times New Roman' in {f.name for f in font_manager.fontManager.ttflist} or 'Times' in {f.name for f in font_manager.fontManager.ttflist}}")
