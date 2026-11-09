"""
Figure of the replay results, from results/summary.json and results/sweep_*.csv
(run analyse.py, then analyse_detail.py, then this).  Writes results/fig_variants.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
S = json.load(open(os.path.join(RES, "summary.json")))

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 of the palette used in gpedm_py_docs
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7.5,
})

main, hub = S["main"], S["hub"]
rows = [
    ("Production model, as now", "prod", None),
    ("    exact phi gradient", "prod_exactgrad", None),
    ("    + time of year as input", "prod_season", None),
    ("    + log scale", "prod_log", None),
    ("    + log scale + time of year", "prod_log_season", None),
    ("One GP per horizon", "direct", None),
    ("    + time of year", "direct_season", None),
    ("    + log scale", "direct_log", None),
    ("    + log scale + time of year", "direct_log_season", None),
    ("    + the same, change as response", "direct_log_diff_season", None),
    ("    mix of E = 3 and E = 52", "direct_mix3+52_qavg", "sweep_direct"),
    ("    mix of E = 3 and E = 52, log scale", "direct_log_diff_mix3+52_qavg", "sweep_direct_log_diff"),
]
base = main["prod"]["rank_score"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.3))

y = np.arange(len(rows))[::-1]
for yy, (label, key, src) in zip(y, rows):
    r = (S[src] if src else main)[key]
    ax.plot([base + r["lo"], base + r["hi"]], [yy, yy], color=S1, lw=1.6, solid_capstyle="butt")
    ax.plot(base + r["diff"], yy, "o", color=S1, ms=6, mec="white", mew=1.2, zorder=5)
    ax.text(0.868, yy, f'{base + r["diff"]:.2f}', va="center", ha="right", color=INK2, fontsize=7.5)
for val, name in ((hub["baseline"], "hub baseline"), (hub["ensemble"], "hub ensemble")):
    ax.axvline(val, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(val, len(rows) - 0.35, name, ha="center", va="bottom", color=INK2, fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], ha="left")
ax.tick_params(axis="y", pad=178)
ax.set_ylim(-0.6, len(rows) - 0.2)
ax.set_xlim(0.63, 0.87)
ax.grid(axis="x")
ax.set_xlabel("mean hub rank score (1 = best model on every forecast)")
ax.set_title(f'Replay of {hub["origin_dates"]} forecast rounds: score of each variant', loc="left", fontsize=9, color=INK, pad=16)

es = [1, 2, 3, 4, 6, 8, 13, 26, 52]
xpos = np.arange(len(es))
for basekey, color, label in (("direct_log_diff", S1, "log scale"), ("direct", S2, "incidence scale")):
    sw = S[f"sweep_{basekey}"]
    vals = [sw[f"{basekey}_E{e}"]["rank_score"] for e in es]
    bx.plot(xpos, vals, color=color, lw=1.6, marker="o", ms=5, mec="white", mew=1.0, label=label, zorder=4)
    mix = sw[f"{basekey}_mix3+52_qavg"]["rank_score"]
    bx.plot([xpos[-1] + 1.1], [mix], marker="D", ms=6, color=color, mec="white", mew=1.0, zorder=5)
    bx.annotate(label, (xpos[3], vals[3]), xytext=(0, 7), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
bx.set_xticks(list(xpos) + [xpos[-1] + 1.1])
bx.set_xticklabels([str(e) for e in es] + ["3 + 52\nmix"])
bx.set_xlabel("embedding dimension E (weeks of history)")
bx.set_ylabel("mean hub rank score")
bx.grid(axis="y")
bx.set_title("One GP per horizon: score against E", loc="left", fontsize=9, color=INK, pad=16)
fig.text(0.012, 0.012, f'{hub["n_ranked"]:,} forecasts ranked against the other hub models, origin dates {hub["first"]} to {hub["last"]}.\n'
         "Bars: 95% interval of the difference from the production model (block bootstrap over origin dates). "
         "Diamonds: mix of E = 3 and E = 52.", fontsize=7, color=INK2, linespacing=1.5)
fig.subplots_adjust(left=0.245, right=0.985, top=0.88, bottom=0.2)
fig.savefig(os.path.join(RES, "fig_variants.png"), dpi=200)
fig.savefig(os.path.join(RES, "fig_variants.pdf"))
print("written", os.path.join(RES, "fig_variants.png"))
