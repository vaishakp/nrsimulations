#!/usr/bin/env python3
"""Per-run constraint-violation time series, and the approach to merger.

Two figures:

  fig_constraint_gallery.pdf -- one small panel per simulation, the full
      ||C||_2 history at the finest resolution, with merger marked.

  fig_constraint_merger.pdf -- all runs overlaid on a time axis measured from
      merger, each normalised to its own late-inspiral level, so that the
      question "does the violation grow into merger" can be read off directly.

Merger time is taken from the waveform, `data/orbit_counts.json` (`t_peak`).
That is a retarded time at null infinity while the constraints are recorded in
simulation coordinate time, so the two axes are not guaranteed to agree; in
practice the constraint record ends within a few M of `t_peak` for nearly every
run, which is itself a useful consistency check and is reported below.
"""
import os
import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_dir import FIGURES as FIG

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#d9d9d9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 160,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
TWOCOL = 7.0


def gridify(ax):
    ax.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load_runs():
    """{sim: (lev, t, L2, Linf, t_merger or None)} at the finest resolution."""
    d = json.load(open(os.path.join(DATA, "constraints_harvest.json")))
    oc = json.load(open(os.path.join(DATA, "orbit_counts.json")))["summary"]
    best = {}
    for k, v in d.items():
        sim, _, lev = k.split("|")
        lev = int(lev[3:])
        if "norm" not in v:
            continue
        cur = best.get(sim)
        if cur is None or v["norm"]["t_end"] > cur[1]["norm"]["t_end"]:
            best[sim] = (lev, v)
    out = {}
    for sim, (lev, v) in best.items():
        a = np.asarray(v["norm_series"], dtype=float)
        tpk = (oc.get(sim) or {}).get("t_peak")
        out[sim] = (lev, a[:, 0], a[:, 1], a[:, 2], tpk)
    return out


def gallery(runs):
    names = sorted(runs)
    n = len(names)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TWOCOL, 1.45 * nrow),
                             sharey=True)
    axes = np.atleast_2d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for i, name in enumerate(names):
        lev, t, c2, ci, tpk = runs[name]
        ax = axes[i]
        s = t > t[0] + 50.0
        ax.semilogy(t[s] / 1000.0, c2[s], color=C[0], lw=0.5)
        if tpk is not None:
            ax.axvline(tpk / 1000.0, color=C[1], lw=0.7, ls="--")
        ax.set_title(f"{name}  L{lev}", fontsize=5.6, loc="left", pad=2)
        ax.tick_params(labelsize=5.5)
        ax.set_ylim(1e-7, 1e-1)
        gridify(ax)
    for i in range(n):
        if i // ncol == nrow - 1 or i >= n - ncol:
            axes[i].set_xlabel(r"$t\,[10^3M]$", fontsize=6)
    for r in range(nrow):
        axes[r * ncol].set_ylabel(r"$\|\mathcal{C}\|_2$", fontsize=6)
    fig.tight_layout(h_pad=0.5, w_pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_constraint_gallery.pdf"))
    plt.close(fig)
    print("  fig_constraint_gallery.pdf")


def merger_view(runs):
    """Overlay, normalised to each run's late-inspiral level."""
    fig, axes = plt.subplots(1, 2, figsize=(TWOCOL, 2.9))
    grid = np.linspace(-1500.0, 150.0, 900)
    stats = []
    for ax, (col, lab) in zip(axes, [(2, r"$\|\mathcal{C}\|_2$"),
                                     (3, r"$\|\mathcal{C}\|_\infty$")]):
        stack = []
        for name in sorted(runs):
            lev, t, c2, ci, tpk = runs[name]
            if tpk is None:
                continue
            y = c2 if col == 2 else ci
            tt = t - tpk
            # local baseline: inspiral well before merger, avoiding both the
            # junk transient and the merger itself
            base = (tt > -6000) & (tt < -1000)
            if base.sum() < 20:
                base = (tt < -500) & (tt > tt[0] + 300)
            if base.sum() < 10:
                continue
            b = np.median(y[base])
            if b <= 0:
                continue
            # NaN beyond the end of this run's record, so a record that stops
            # before merger does not masquerade as a drop to zero
            r = np.interp(grid, tt, y / b, left=np.nan, right=np.nan)
            r[(grid < tt[0]) | (grid > tt[-1])] = np.nan
            stack.append(r)
            if col == 2:
                late = (tt > -300) & (tt < 500)
                if late.sum() > 3:
                    stats.append((name, float(np.max(y[late]) / b),
                                  float(tt[-1])))
        S = np.array(stack)
        with np.errstate(all="ignore"):
            med = np.nanmedian(S, axis=0)
            lo = np.nanpercentile(S, 10, axis=0)
            hi = np.nanpercentile(S, 90, axis=0)
        ax.fill_between(grid, lo, hi, color=C[0], alpha=0.22, lw=0,
                        label="10--90th percentile")
        ax.semilogy(grid, med, color=C[0], lw=1.3, label="median")
        for nm, cc in [("ICTSEccParallel13", C[1]), ("EccPrecDiff002", C[2])]:
            if nm not in runs:
                continue
            lev, t, c2, ci, tpk = runs[nm]
            y = c2 if col == 2 else ci
            tt = t - tpk
            base = (tt > -6000) & (tt < -1000)
            b = np.median(y[base]) if base.sum() > 10 else None
            if not b:
                continue
            w = (tt > -1500) & (tt < 150)
            ax.semilogy(tt[w], y[w] / b, color=cc, lw=0.8, label=nm)
        ax.axvline(0.0, color=INK2, lw=0.8, ls="--")
        ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel(r"$(t-t_{\rm merger})/M$")
        ax.set_ylabel(lab + " / late-inspiral level")
        ax.set_xlim(-1500, 150)
        ax.set_ylim(3e-2, 3e2)
        gridify(ax)
    axes[0].legend(loc="upper left", fontsize=6.4)
    axes[0].set_title("(a) volume $L^2$ norm", loc="left")
    axes[1].set_title(r"(b) pointwise $L^\infty$ norm", loc="left")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIG, "fig_constraint_merger.pdf"))
    plt.close(fig)
    print("  fig_constraint_merger.pdf")

    stats.sort(key=lambda r: -r[1])
    g = np.array([s[1] for s in stats])
    print(f"\n  growth of ||C||_2 in [-300, +500]M about merger, vs late inspiral "
          f"(n={len(stats)}):")
    print(f"    median x{np.median(g):.2f}   "
          f">2x: {(g > 2).sum()}/{len(g)}   >10x: {(g > 10).sum()}/{len(g)}")
    for nme, r, tend in stats[:6]:
        print(f"      {nme:22s} x{r:8.2f}   record ends {tend:+.0f} M "
              f"from merger")
    print("      ...")
    for nme, r, tend in stats[-3:]:
        print(f"      {nme:22s} x{r:8.2f}   record ends {tend:+.0f} M "
              f"from merger")
    return stats


def growth_stats(runs):
    """Peak growth over the late-inspiral level, in several windows."""
    out = {}
    for lab, (a, b) in [("500M", (-500, 150)), ("100M", (-100, 150)),
                        ("30M", (-30, 150))]:
        g2, gi = [], []
        for nm, (lev, t, c2, ci, tpk) in runs.items():
            if tpk is None:
                continue
            tt = t - tpk
            base = (tt > -6000) & (tt < -1000)
            if base.sum() < 20:
                base = (tt < -500) & (tt > tt[0] + 300)
            if base.sum() < 10:
                continue
            w = (tt > a) & (tt < b)
            if w.sum() < 2:
                continue
            g2.append(float(np.max(c2[w]) / np.median(c2[base])))
            gi.append(float(np.max(ci[w]) / np.median(ci[base])))
        out[lab] = dict(n=len(g2), L2_median=float(np.median(g2)),
                        L2_max=float(np.max(g2)),
                        Linf_median=float(np.median(gi)),
                        Linf_max=float(np.max(gi)))
    junk = []
    for nm, (lev, t, c2, ci, tpk) in runs.items():
        tt = t - t[0]
        early, late = tt < 300, tt > 2000
        if early.sum() < 3 or late.sum() < 50:
            continue
        junk.append(float(np.max(c2[early]) / np.median(c2[late])))
    out["junk"] = dict(n=len(junk), median=float(np.median(junk)))
    return out


def main():
    runs = load_runs()
    print(f"{len(runs)} simulations")
    gaps = [(n, v[1][-1] - v[4]) for n, v in runs.items() if v[4] is not None]
    g = np.array([x[1] for x in gaps])
    print(f"  constraint record ends {np.median(g):+.0f} M from waveform "
          f"merger (median over {len(g)}); "
          f"|gap| < 50M for {(np.abs(g) < 50).sum()}/{len(g)}")
    gallery(runs)
    merger_view(runs)
    stats = growth_stats(runs)
    stats["gap"] = dict(n=len(g), median=float(np.median(g)),
                        n_close=int((np.abs(g) < 50).sum()))
    json.dump(stats, open(os.path.join(DATA, "constraint_merger.json"), "w"),
              indent=1)
    print("  wrote data/constraint_merger.json")


if __name__ == "__main__":
    main()
