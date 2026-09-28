#!/usr/bin/env python3
"""Reduce the sonic constraint harvest and make the constraint-violation figure.

`data/constraints_harvest.json` is produced by `harvest_constraints.py`, which
runs on the cluster (see the paper's data-availability section).  Each entry is
one (simulation, eccentricity-reduction directory, resolution) and carries a
downsampled time series of the normalised generalised-harmonic constraint,

    ||C||_2 = L2(NormalizedGhCe),   ||C||_inf = Linf(NormalizedGhCe),

joined across evolution segments with checkpoint-restart overlap removed.

The two things we want from it are the level at which the constraints are
satisfied, and whether that level improves with resolution -- the latter being
the real test, since an absolute constraint norm has no meaning on its own.
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
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load():
    p = os.path.join(DATA, "constraints_harvest.json")
    if not os.path.exists(p):
        print("  [skip] no constraints_harvest.json")
        return None
    return json.load(open(p))


def index(d):
    """{sim: {lev: record}} keeping the Ecc directory with the longest run."""
    out = {}
    for k, v in d.items():
        sim, eccdir, lev = k.split("|")
        lev = int(lev.replace("Lev", ""))
        if "norm" not in v:
            continue
        cur = out.setdefault(sim, {}).get(lev)
        if cur is None or v["norm"]["t_end"] > cur["norm"]["t_end"]:
            v = dict(v)
            v["_ecc_dir"] = eccdir
            out[sim][lev] = v
    return out


def series(rec):
    a = np.asarray(rec["norm_series"], dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]


def main():
    d = load()
    if d is None:
        return
    idx = index(d)
    cat = json.load(open(os.path.join(DATA, "catalog.json")))

    # The harvest walks the whole run tree, which also holds the PSUTest /
    # PSUTestHR commissioning runs.  Those are not catalogue entries and must
    # not enter any statistic quoted in the paper.
    dropped = sorted(s for s in idx if s not in cat)
    for s in dropped:
        del idx[s]
    if dropped:
        print(f"  [filter] not catalogue entries, dropped: {', '.join(dropped)}")

    # ------------------------------------------------ summary + convergence
    summary, ratios = {}, []
    for sim, levs in sorted(idx.items()):
        entry = {}
        for lev, rec in sorted(levs.items()):
            n = rec["norm"]
            entry[lev] = dict(L2_median=n["L2_median"], L2_max=n["L2_max"],
                              Linf_max=n["Linf_max"], t_end=n["t_end"],
                              n_segments=rec["n_segments"])
        # Resolution trend, compared on the time window the two levels share.
        # Taking the ratio of each level's median over its own span would
        # compare different stretches of the evolution: the levels often stop
        # at very different times, and the constraints grow towards merger.
        ks = sorted(k for k in entry if isinstance(k, int))
        if len(ks) >= 2:
            lo, hi = ks[-2], ks[-1]
            tl, cl, _ = series(levs[lo])
            th, ch, _ = series(levs[hi])
            t0 = max(tl[0], th[0]) + 200.0
            t1 = min(tl[-1], th[-1])
            if t1 > t0 + 500.0:
                g = np.linspace(t0, t1, 2000)
                a = np.interp(g, tl, cl)
                b = np.interp(g, th, ch)
                good = (a > 0) & (b > 0)
                if good.sum() > 100:
                    r = float(np.median(a[good] / b[good]))
                    entry["ratio_lo_hi"] = r
                    entry["ratio_levels"] = [lo, hi]
                    entry["overlap_M"] = float(t1 - t0)
                    ratios.append((sim, lo, hi, r))
        summary[sim] = entry

    json.dump(summary, open(os.path.join(DATA, "constraints.json"), "w"),
              indent=1)

    med = np.array([v[max(k for k in v if isinstance(k, int))]["L2_median"]
                    for v in summary.values()])
    print(f"{len(summary)} simulations")
    print(f"  ||C||_2 median over catalogue (finest level): "
          f"{np.median(med):.2e}   range {med.min():.2e} - {med.max():.2e}")
    if ratios:
        rr = np.array([r[3] for r in ratios])
        print(f"  resolution improvement Lev(lo)/Lev(hi): median {np.median(rr):.2f}"
              f"  ({(rr > 1).sum()}/{len(rr)} improve with resolution)")
        for s, lo, hi, r in sorted(ratios, key=lambda x: -x[3])[:5]:
            print(f"      {s:22s} Lev{lo}->Lev{hi}  x{r:.2f}")
        print("      ...")
        for s, lo, hi, r in sorted(ratios, key=lambda x: x[3])[:3]:
            print(f"      {s:22s} Lev{lo}->Lev{hi}  x{r:.2f}")

    # --------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(TWOCOL, 2.8))

    # (a) time series for a few representative runs at the finest level
    ax = axes[0]
    want = ["EccPrecDiff002", "ICTSEccParallel12", "ICTSEccParallel01",
            "ICTSEccParallel08"]
    n = 0
    for name in want:
        if name not in idx:
            continue
        lev = max(idx[name])
        t, l2, _ = series(idx[name][lev])
        # drop the initial-data transient, which spikes three decades above
        # the evolution and would set the whole vertical scale
        s = t > t[0] + 200.0
        ax.semilogy(t[s] / 1000.0, l2[s], color=C[n % len(C)], lw=0.7,
                    label=f"{name} L{lev}")
        n += 1
    ax.set_xlabel(r"$t\,[10^3 M]$")
    ax.set_ylabel(r"$\|\mathcal{C}\|_2$")
    ax.set_ylim(3e-7, 3e-2)
    ax.legend(loc="upper center", fontsize=6.2, handlelength=1.3,
              labelspacing=0.25, ncol=2)
    ax.set_title("(a) constraint violation", loc="left")
    gridify(ax)

    # (b) resolution comparison for the runs that have two levels
    ax = axes[1]
    xs, ys, cs, lens = [], [], [], []
    for sim, e in summary.items():
        if "ratio_lo_hi" not in e:
            continue
        lo, hi = e["ratio_levels"]
        tl, cl, _ = series(idx[sim][lo])
        th, ch, _ = series(idx[sim][hi])
        t0 = max(tl[0], th[0]) + 200.0
        t1 = min(tl[-1], th[-1])
        g = np.linspace(t0, t1, 2000)
        xs.append(float(np.median(np.interp(g, tl, cl))))
        ys.append(float(np.median(np.interp(g, th, ch))))
        cs.append(bool(cat.get(sim, {}).get("precessing")))
        lens.append(float(e[hi]["t_end"]))
    xs, ys = np.array(xs), np.array(ys)
    ln = np.array(lens)
    sc = ax.scatter(xs, ys, c=np.log10(ln), cmap="viridis", s=34,
                    edgecolor="white", linewidth=0.6, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ (run length $/M$)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = [min(xs.min(), ys.min()) * 0.5, max(xs.max(), ys.max()) * 2]
    ax.plot(lim, lim, "-", color=INK2, lw=0.8, zorder=2)
    ax.text(0.96, 0.06, "below the line:\nimproves with resolution",
            transform=ax.transAxes, fontsize=6.4, color=INK2, ha="right")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"median $\|\mathcal{C}\|_2$, coarser level")
    ax.set_ylabel(r"median $\|\mathcal{C}\|_2$, finer level")
    ax.set_title("(b) resolution dependence", loc="left")
    gridify(ax)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIG, "fig_constraints.pdf"))
    plt.close(fig)
    print("  fig_constraints.pdf")


if __name__ == "__main__":
    main()
