"""
The two data figures of gpedm_py_math.pdf, drawn from numbers.json and the trace files
that make_numbers.py writes.  Run after it, from ~/git/Respi:
    python3 gpedm_py_docs/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
NUM = json.load(open(os.path.join(HERE, "numbers.json")))

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, validated as a set
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7,
})


def panel(ax, title):
    ax.grid(axis="y")
    ax.set_title(title, loc="left", fontsize=8, color=INK, pad=6)
    ax.minorticks_off()


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


# ---- figure: how the optimiser runs on one series -----------------------------------------
t = pd.read_csv(os.path.join(HERE, "trace_ARI_BE.csv"))
d = NUM["ARI_BE"]["d"]
VMIN, VMAX = 1e-4, 4.9999
squash = lambda p: (VMAX - VMIN) / (1 + np.exp(-p)) + VMIN

fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1), constrained_layout=True)

ax = axes[0]
panel(ax, "Objective F")
ax.plot(t.iter, t.nllpost, color=S1, lw=1.5, solid_capstyle="round")
ax.plot(t.iter.iloc[-1], t.nllpost.iloc[-1], "o", color=S1, ms=5.5, mec="white", mew=1.2, zorder=5)
ax.set_yscale("symlog", linthresh=100, linscale=0.7)
ax.set_yticks([-100, 0, 100, 1000, 10000])
ax.set_yticklabels(["−100", "0", "100", "1K", "10K"])
ax.set_ylim(-160, 30000)
ax.annotate(f"{t.nllpost.iloc[-1]:.1f}".replace("-", "−"), (t.iter.iloc[-1], t.nllpost.iloc[-1]),
            xytext=(-4, 7), textcoords="offset points", ha="right", color=INK2, fontsize=7)

ax = axes[1]
panel(ax, "Relative change of F")
ax.plot(t.iter[1:], t.relchange[1:], color=S1, lw=1.5, solid_capstyle="round")
ax.plot(t.iter.iloc[-1], t.relchange.iloc[-1], "o", color=S1, ms=5.5, mec="white", mew=1.2, zorder=5)
ax.set_yscale("log")
ax.set_ylim(1e-9, 30)
ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
ax.axhline(1e-7, color=INK2, lw=0.6)
ax.text(0.5, 1.6e-7, "stop when below $10^{-7}$", color=INK2, fontsize=7, va="bottom")

ax = axes[2]
panel(ax, "Parameters")
series = [(r"$\varphi_1$", np.exp(t.parst_phi1), S1), (r"$V_e$", squash(t.parst_ve), S2),
          (r"$\sigma^2$", squash(t.parst_sigma2), S3)]
for label, y, c in series:
    ax.plot(t.iter, y, color=c, lw=1.5, solid_capstyle="round", label=label)
    ax.annotate(label, (t.iter.iloc[-1], y.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", color=INK2, fontsize=7.5)
ax.set_yscale("log")
ax.set_ylim(5e-4, 12)
ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax.legend(loc="lower right", ncol=3, handlelength=1.1, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)

for ax in axes:
    ax.set_xlabel("iteration")
    ax.set_xlim(0, t.iter.iloc[-1] + (5 if ax is axes[2] else 1.5))
    ax.set_xticks([0, 10, 20, 30])
    ax.minorticks_off()
save(fig, "fig_trace")

# ---- figure: two forecasts, one that depends on the horizon and one that does not -----------
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55), constrained_layout=True, sharex=True)
for ax, key, title in zip(axes, ("ARI_BE", "ARI_SK"), ("Belgium", "Slovakia")):
    o = NUM[key]
    panel(ax, title)
    obs = np.array([np.nan if v is None else v for v in o["last_obs"]])[-24:]
    ax.plot(np.arange(-len(obs) + 1, 1), obs, color=INK2, lw=1.5, solid_capstyle="round")
    h = np.arange(1, 5)
    mu, sd = np.array(o["predmean"]), np.array(o["predsd"])
    for z, lw in ((1.959964, 1.0), (0.674490, 3.4)):
        for hh, lo, hi in zip(h, np.clip(mu - z * sd, 0, None), mu + z * sd):
            ax.plot([hh, hh], [lo, hi], color=S1, lw=lw, solid_capstyle="butt")
    ax.plot(h, mu, "o", color=S1, ms=5.5, mec="white", mew=1.2, zorder=5)
    ax.axhline(o["ymean"], color=AXIS, lw=0.7, zorder=1)
    ax.text(-11, o["ymean"], "series mean", color=INK2, fontsize=7, va="bottom", ha="center")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.03)
    ax.set_xlim(-len(obs) + 0.5, 4.8)
    ax.set_xticks([-20, -10, 0, 4])
    ax.set_xticklabels(["−20", "−10", "0", "+4"])
    ax.set_xlabel("weeks from the last observation")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
axes[0].set_ylabel("ARI incidence per 100,000")
handles = [Line2D([], [], color=INK2, lw=1.5, label="observed"),
           Line2D([], [], color=S1, lw=0, marker="o", ms=5.5, mec="white", mew=1.2, label="forecast median"),
           Line2D([], [], color=S1, lw=3.4, label="50% interval"),
           Line2D([], [], color=S1, lw=1.0, label="95% interval")]
fig.legend(handles=handles, loc="outside upper left", ncol=4, handlelength=1.4, columnspacing=1.4)
save(fig, "fig_forecast")

# ---- figure: the same two series, forecast submitted up to 14 October against the one from 21 October
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55), constrained_layout=True, sharex=True)
for ax, key, title in zip(axes, ("ARI_BE", "ARI_SK"), ("Belgium", "Slovakia")):
    old, new = NUM[key], NUM["direct_" + key]
    panel(ax, title)
    obs = np.array([np.nan if v is None else v for v in new["last_obs"]])[-24:]
    ax.plot(np.arange(-len(obs) + 1, 1), obs, color=INK2, lw=1.5, solid_capstyle="round")
    h = np.arange(1, 5)
    mu, sd = np.array(old["predmean"]), np.array(old["predsd"])
    bands = {S2: (np.clip(mu, 0, None), np.clip(mu - 1.959964 * sd, 0, None), mu + 1.959964 * sd,
                  np.clip(mu - 0.674490 * sd, 0, None), mu + 0.674490 * sd, -0.17),
             S1: (np.array(new["median"]), np.array(new["q025"]), np.array(new["q975"]),
                  np.array(new["q25"]), np.array(new["q75"]), 0.17)}
    for c, (med, lo95, hi95, lo50, hi50, dx) in bands.items():
        for hh, a, b, a5, b5 in zip(h + dx, lo95, hi95, lo50, hi50):
            ax.plot([hh, hh], [a, b], color=c, lw=1.0, solid_capstyle="butt")
            ax.plot([hh, hh], [a5, b5], color=c, lw=3.4, solid_capstyle="butt")
        ax.plot(h + dx, med, "o", color=c, ms=5, mec="white", mew=1.1, zorder=5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.03)
    ax.set_xlim(-len(obs) + 0.5, 4.8)
    ax.set_xticks([-20, -10, 0, 4])
    ax.set_xticklabels(["−20", "−10", "0", "+4"])
    ax.set_xlabel("weeks from the last observation")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
axes[0].set_ylabel("ARI incidence per 100,000")
handles = [Line2D([], [], color=INK2, lw=1.5, label="observed"),
           Line2D([], [], color=S2, lw=3.4, marker="o", ms=5, mec="white", mew=1.1, label="variable-step model (to 14 Oct)"),
           Line2D([], [], color=S1, lw=3.4, marker="o", ms=5, mec="white", mew=1.1, label="one GP per horizon (from 21 Oct)")]
fig.legend(handles=handles, loc="outside upper left", ncol=3, handlelength=1.6, columnspacing=1.4)
save(fig, "fig_direct")
print("figures written to", FIG)
