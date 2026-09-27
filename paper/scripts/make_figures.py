#!/usr/bin/env python3
"""Figures for the ICTS-TIFR eccentric/precessing SpEC catalog paper.

Palette: Okabe-Ito, colourblind-safe, validated (adjacent-pair CVD dE >= 11,
normal-vision dE >= 21).  Categorical hues are assigned in a fixed order and are
never cycled; every multi-series panel also carries a legend and, where it fits,
direct labels, so identity is never colour-alone.
"""
import json
import os
import csv
import ast
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_dir import FIGURES as FIG

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
WF = os.path.join(HERE, "..", "waveforms")
os.makedirs(FIG, exist_ok=True)

# ---- fixed categorical order (validated) -----------------------------------
C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#d9d9d9"
SEQ = "viridis"          # single perceptually-uniform ramp for magnitude

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.7,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ONECOL, TWOCOL = 3.4, 7.0


def gridify(ax):
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- load data
cat = json.load(open(os.path.join(DATA, "catalog.json")))
tim = json.load(open(os.path.join(DATA, "timings.json")))
ts = json.load(open(os.path.join(DATA, "timeseries.json")))
orb = json.load(open(os.path.join(DATA, "orbit_counts.json")))["summary"]


def orbits_of(e):
    """Number of ORBITS.  The website tables list GW cycles (= 2 x orbits);
    prefer the value measured directly from the (2,2) phase where available."""
    o = orb.get(e["name"])
    if o:
        return o["orbits"]
    nc = e.get("ncycles_max")
    return nc / 2.0 if nc else None

SERIES = ["ICTSEccParallel", "EccContPrecDiff", "EccPrecDiff"]
SLABEL = {"ICTSEccParallel": "ICTSEccParallel", "EccContPrecDiff": "EccContPrecDiff",
          "EccPrecDiff": "EccPrecDiff"}
SCOL = {s: C[i] for i, s in enumerate(SERIES)}
SMRK = {"ICTSEccParallel": "o", "EccContPrecDiff": "s", "EccPrecDiff": "^"}

rows = [e for e in cat.values() if e.get("q") and orbits_of(e)]


# =============================================================== Figure 1
def fig_parameter_space():
    fig, axes = plt.subplots(1, 3, figsize=(TWOCOL, 2.45))

    # (a) q vs chi_eff, colour = eccentricity
    ax = axes[0]
    q = np.array([e["q"] for e in rows])
    ce = np.array([e.get("chi_eff", 0.0) for e in rows])
    ec = np.array([e.get("ecc", 0.0) for e in rows])
    sc = ax.scatter(q, ce, c=ec, cmap=SEQ, s=34, edgecolor="white", linewidth=0.6,
                    vmin=0, vmax=max(ec.max(), 0.25), zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"mass ratio $q$")
    ax.set_ylabel(r"$\chi_{\rm eff}$")
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "10"])
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"eccentricity $e$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    gridify(ax)
    ax.set_title("(a) mass ratio – aligned spin", loc="left")

    # (b) chi_p vs eccentricity, marker = series
    ax = axes[1]
    for s in SERIES:
        sub = [e for e in rows if e["series"] == s]
        if not sub:
            continue
        ax.scatter([e.get("ecc", 0) for e in sub], [e.get("chi_p", 0) for e in sub],
                   c=SCOL[s], marker=SMRK[s], s=34, edgecolor="white",
                   linewidth=0.6, label=SLABEL[s], zorder=3)
    ax.set_xlabel(r"eccentricity $e$")
    ax.set_ylabel(r"precession parameter $\chi_{\rm p}$")
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2,
              bbox_to_anchor=(0.0, 1.02))
    gridify(ax)
    ax.set_title("(b) eccentricity – precession", loc="left")

    # (c) number of orbits
    ax = axes[2]
    for s in SERIES:
        sub = [e for e in rows if e["series"] == s]
        if not sub:
            continue
        ax.scatter([e["q"] for e in sub], [orbits_of(e) for e in sub],
                   c=SCOL[s], marker=SMRK[s], s=34, edgecolor="white",
                   linewidth=0.6, label=SLABEL[s], zorder=3)
    best = max(rows, key=orbits_of)
    ax.annotate("EccPrecDiff001/002\n80 orbits",
                xy=(best["q"], orbits_of(best)),
                xytext=(1.5, 68), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xscale("log")
    ax.set_xlabel(r"mass ratio $q$")
    ax.set_ylabel("orbits to merger")
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "10"])
    gridify(ax)
    ax.set_title("(c) waveform length", loc="left")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIG, "fig_parameter_space.pdf"))
    plt.close(fig)
    print("  fig_parameter_space.pdf")


# =============================================================== Figure 2
def fig_performance():
    """Evolution speed and cost."""
    fig, axes = plt.subplots(1, 2, figsize=(TWOCOL, 2.6))

    # (a) instantaneous evolution speed dt/dT vs t for representative runs
    ax = axes[0]
    show = [("EccPrecDiff002|Lev3", "EccPrecDiff002 (q=1, 80 orbits)"),
            ("ICTSEccParallel02|Lev3", "ICTSEccParallel02 (q=2.5)"),
            ("ICTSEccParallel13|Lev3", "ICTSEccParallel13 (q=9.5)")]
    for i, (k, lab) in enumerate(show):
        if k not in ts:
            continue
        a = np.array(ts[k])
        t, sp = a[:, 0], a[:, 6]
        good = (sp > 0) & np.isfinite(sp)
        # median filter to tame per-checkpoint jitter
        x, y = t[good], sp[good]
        if len(y) > 25:
            w = 9
            y = np.convolve(y, np.ones(w) / w, mode="valid")
            x = x[w // 2: len(x) - (w - 1 - w // 2)]
        ax.plot(x, y, color=C[i], lw=1.3, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, 400)
    ax.set_xlabel(r"evolution time $t/M$")
    ax.set_ylabel(r"speed $\mathrm{d}t/\mathrm{d}T\;[M\,\mathrm{h}^{-1}]$")
    ax.legend(loc="lower left", handlelength=1.4)
    gridify(ax)
    ax.set_title("(a) instantaneous evolution speed", loc="left")

    # (b) cost vs orbits
    ax = axes[1]
    byname = {}
    for r in tim:
        byname.setdefault(r["sim"], []).append(r)
    xs, ys, cs, ms_, names = [], [], [], [], []
    for name, e in cat.items():
        if not e.get("ncycles_max"):
            continue
        rr = byname.get(name)
        if not rr:
            continue
        cpu = max(r["cpu_h"] for r in rr)
        xs.append(e["ncycles_max"])
        ys.append(cpu)
        cs.append(SCOL.get(e["series"], C[4]))
        ms_.append(SMRK.get(e["series"], "o"))
        names.append(name)
    for s in SERIES:
        idx = [i for i, n in enumerate(names) if cat[n]["series"] == s]
        if not idx:
            continue
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], c=SCOL[s],
                   marker=SMRK[s], s=34, edgecolor="white", linewidth=0.6,
                   label=SLABEL[s], zorder=3)
    # reference scaling
    xx = np.array([4.5, 85.0])
    ax.plot(xx, 760 * xx, color=INK2, lw=0.8, ls="--", zorder=2)
    ax.text(26, 760 * 26 * 1.45, r"$\propto N_{\rm orb}$", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("orbits to merger")
    ax.set_ylabel("CPU-hours (highest level)")
    ax.legend(loc="upper left", handletextpad=0.3)
    gridify(ax)
    ax.set_title("(b) cost per simulation", loc="left")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIG, "fig_performance.pdf"))
    plt.close(fig)
    print("  fig_performance.pdf")


# =============================================================== Figure 3
def fig_optimization():
    """Compile-time optimisation speedups measured on sonic."""
    fig, axes = plt.subplots(1, 2, figsize=(TWOCOL, 2.5))

    # (a) dgemm throughput
    ax = axes[0]
    labels = ["gnu +\nnetlib", "aocc +\naocl", "gcc +\naocl", "gcc +\nmkl",
              "gcc + mkl\n(AMD fix)"]
    gflops = [54.71, 54.42, 55.10, 53.95, 75.13]
    bars = ax.bar(range(len(labels)), gflops, color=[C[0]] * 4 + [C[1]],
                  width=0.68, zorder=3)
    for b, v in zip(bars, gflops):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"dgemm throughput [GFLOP s$^{-1}$]")
    ax.set_ylim(0, 88)
    gridify(ax)
    ax.set_title(r"(a) BLAS, $5004^2$ matrix", loc="left")

    # (b) speedup summary
    ax = axes[1]
    labels2 = ["ID solver", "Evolution", "glibc $\\geq$ 2.34\n(of evolution)"]
    vals = [90, 40, 19]
    bars = ax.barh(range(len(labels2))[::-1], vals,
                   color=[C[1], C[0], C[2]], height=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"+{v}%",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(labels2))[::-1])
    ax.set_yticklabels(labels2, fontsize=8)
    ax.set_xlabel("speedup over reference build [%]")
    ax.set_xlim(0, 108)
    gridify(ax)
    ax.set_title("(b) SpEC speedup after optimisation", loc="left")

    fig.tight_layout(w_pad=1.8)
    fig.savefig(os.path.join(FIG, "fig_optimization.pdf"))
    plt.close(fig)
    print("  fig_optimization.pdf")


if __name__ == "__main__":
    print("figures ->", FIG)
    fig_parameter_space()
    fig_performance()
    fig_optimization()
