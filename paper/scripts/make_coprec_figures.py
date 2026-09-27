#!/usr/bin/env python3
"""Figures for the co-precessing-frame, first-law and detectability sections."""
import os
import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import (WF, DATA, sim_path, load_modes, to_waveform,
                          coprecessing, euler_angles, quadrupole_concentration,
                          mirror_asymmetry)

from paper_dir import FIGURES as FIG

HERE = os.path.dirname(os.path.abspath(__file__))

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
ONECOL, TWOCOL = 3.4, 7.0


def gridify(ax):
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(fname):
    p = os.path.join(DATA, fname)
    return json.load(open(p)) if os.path.exists(p) else {}


# ============================================ Figure: co-precessing validity
def fig_coprecessing(example="ICTSEccParallel01", lev=3):
    surv = load("coprec_survey.json")
    ecc = load("eccentricity.json")
    if not surv:
        print("  [skip] no coprec_survey.json")
        return

    fig = plt.figure(figsize=(TWOCOL, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)

    # (a) mirror asymmetry through the inspiral.  Runs differ in length by a
    # factor of ten, so the x-axis is the fraction of the inspiral elapsed;
    # the aligned-spin runs are shown as a band, being the noise floor.
    ax = fig.add_subplot(gs[0, :])
    show_prec = [("EccPrecDiff002", 3), ("EccContPrecDiff008", 3),
                 (example, lev), ("EccContPrecDiff007", 3)]
    show_align = [("ICTSEccParallel12", 3), ("ICTSEccParallel04", 3)]
    bmap = {r["name"]: r["beta_med"] for r in surv}

    def curve(nm, lv):
        p = sim_path(nm, lv)
        if not os.path.exists(p):
            return None
        t, m = load_modes(p, ell_max=4)
        W = to_waveform(t, m, ell_max=4,
                        decimate=max(1, int(round(0.5 / np.median(np.diff(t))))))
        Wc, _ = coprecessing(W)
        tpk = W.t[np.argmax(np.abs(W.data[:, W.index(2, 2)]))]
        tt = Wc.t - tpk
        sel = (tt > tt[0] + 300) & (tt < -50)
        x = (tt[sel] - tt[sel][0]) / (tt[sel][-1] - tt[sel][0])
        return x, mirror_asymmetry(Wc)[sel]

    lo, hi, xg = None, None, np.linspace(0, 1, 400)
    for nm, lv in show_align:
        c = curve(nm, lv)
        if c is None:
            continue
        y = np.interp(xg, c[0], c[1])
        lo = y if lo is None else np.minimum(lo, y)
        hi = y if hi is None else np.maximum(hi, y)
    if lo is not None:
        ax.fill_between(xg, lo, hi, color=C[1], alpha=0.30, lw=0,
                        label="aligned-spin runs (noise floor)", zorder=2)
    for i, (nm, lv) in enumerate(show_prec):
        c = curve(nm, lv)
        if c is None:
            continue
        b = bmap.get(nm)
        ax.semilogy(c[0], c[1], color=C[i % len(C)], lw=0.8, zorder=3,
                    label=rf"{nm}" + (rf"  ($\beta={b:.0f}^\circ$)" if b else ""))
    ax.set_xlabel("fraction of inspiral elapsed")
    ax.set_ylabel(r"mirror asymmetry $\mu$")
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-7, 1)
    ax.legend(loc="center left", ncol=2, fontsize=6.6, handlelength=1.6,
              bbox_to_anchor=(0.005, 0.30))
    ax.set_title(r"(a) violation of $h_{\ell,-m}=(-1)^\ell h^{*}_{\ell m}$ "
                 r"in the co-precessing frame", loc="left")
    gridify(ax)

    # (b) mirror asymmetry vs opening angle
    ax = fig.add_subplot(gs[1, 0])
    pr = [r for r in surv if r["prec"]]
    npr = [r for r in surv if not r["prec"]]
    b = np.array([r["beta_med"] for r in pr])
    mu = np.array([r["mu_cp"] for r in pr])
    ax.loglog(b, mu, "o", color=C[0], ms=5, mec="white", mew=0.6,
              label="precessing", zorder=3)
    bb = np.logspace(np.log10(b.min()), np.log10(b.max()), 50)
    c = np.polyfit(np.log(b), np.log(mu), 1)
    ax.loglog(bb, np.exp(c[1]) * bb ** c[0], "-", color=INK2, lw=0.9,
              label=rf"$\mu\propto\beta^{{{c[0]:.2f}}}$")
    mu_n = np.array([r["mu_cp"] for r in npr])
    ax.axhspan(mu_n.min(), mu_n.max(), color=C[1], alpha=0.25, lw=0,
               label="aligned-spin runs", zorder=1)
    ax.set_ylim(0.3 * mu_n.min(), 3 * mu.max())
    ax.set_xlabel(r"median opening angle $\beta$ [deg]")
    ax.set_ylabel(r"$\mu$ (co-precessing)")
    ax.legend(loc="center right", fontsize=6.6)
    ax.set_title(r"(b) scaling with $\beta$", loc="left")
    gridify(ax)

    # (c) quadrupole power concentration, inertial vs co-precessing
    ax = fig.add_subplot(gs[1, 1])
    ci = np.array([r["conc_in"] for r in pr])
    cc = np.array([r["conc_cp"] for r in pr])
    ax.loglog(b, 1 - ci, "o", color=C[1], ms=5, mec="white", mew=0.6,
              label="inertial", zorder=3)
    ax.loglog(b, 1 - cc, "s", color=C[0], ms=5, mec="white", mew=0.6,
              label="co-precessing", zorder=3)
    for i in range(len(b)):
        ax.plot([b[i], b[i]], [1 - ci[i], 1 - cc[i]], "-", color=GRID, lw=0.8,
                zorder=1)
    ax.set_xlabel(r"median opening angle $\beta$ [deg]")
    ax.set_ylabel(r"$1-P_{2,\pm2}/P_{\ell=2}$")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(r"(c) $\ell=2$ power outside $m=\pm2$", loc="left")
    gridify(ax)

    fig.savefig(os.path.join(FIG, "fig_coprecessing.pdf"))
    plt.close(fig)
    print("  fig_coprecessing.pdf")


# ==================================================== Figure: twisting study
def fig_twist():
    tw = load("twist_study.json")
    if not tw:
        print("  [skip] no twist_study.json")
        return
    rows = []
    for n, r in tw.items():
        e = r.get("ecc")
        qc = r["results"].get("nrhybsur_qc") or r["results"].get("seob_qc")
        ec = r["results"].get("seob_ecc")
        if e is None or qc is None or "mismatch" not in qc:
            continue
        rows.append((n, e, qc["mismatch"],
                     ec["mismatch"] if ec and "mismatch" in ec else np.nan))
    if not rows:
        print("  [skip] no usable twist rows")
        return
    rows.sort(key=lambda r: r[1])
    e = np.array([r[1] for r in rows])
    mq = np.array([r[2] for r in rows])
    me = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(TWOCOL * 0.52, 2.9))
    ax.loglog(e, mq, "o", color=C[1], ms=5, mec="white", mew=0.6,
              label="quasi-circular model", zorder=3)
    good = ~np.isnan(me)
    ax.loglog(e[good], me[good], "s", color=C[0], ms=5, mec="white", mew=0.6,
              label="eccentric model", zorder=3)
    for i in np.where(good)[0]:
        ax.plot([e[i], e[i]], [mq[i], me[i]], "-", color=GRID, lw=0.8, zorder=1)
    ax.set_xlim(0.008, 0.9)
    ax.set_ylim(5e-4, 3.0)
    ax.axvspan(0.3, 0.9, color=C[4], alpha=0.12, zorder=0)
    ax.text(0.31, 6.5e-4, "beyond model\ncalibration", fontsize=6.2,
            color=INK2, va="bottom", ha="left")
    ax.set_xlabel(r"measured eccentricity $e$")
    ax.set_ylabel(r"$(2,2)$ mismatch vs NR $h^{\rm CP}$")
    ax.legend(loc="upper left", fontsize=7)
    gridify(ax)
    fig.savefig(os.path.join(FIG, "fig_twist.pdf"))
    plt.close(fig)
    print("  fig_twist.pdf")


# ================================================== Figure: first-law residue
def fig_first_law():
    fl = load("first_law.json")
    cat = load("catalog.json")
    if not fl:
        print("  [skip] no first_law.json")
        return
    P, N = [], []
    for k, v in fl.items():
        if v.get("ecc") is None or v["R_secular_med"] > 1.0:
            continue
        (P if cat.get(k, {}).get("precessing") else N).append(
            (v["ecc"], v["R_secular_med"]))
    fig, ax = plt.subplots(figsize=(ONECOL, 2.8))
    for S, c, lab, mk in [(N, C[0], "aligned spin", "o"),
                          (P, C[1], "precessing", "s")]:
        if not S:
            continue
        x = np.array([s[0] for s in S])
        y = np.array([s[1] for s in S])
        ax.loglog(x, y, mk, color=c, ms=5, mec="white", mew=0.6, label=lab,
                  zorder=3)
    allx = np.array([s[0] for s in P + N])
    ally = np.array([s[1] for s in P + N])
    c = np.polyfit(np.log(allx), np.log(ally), 1)
    xx = np.logspace(np.log10(allx.min()), np.log10(allx.max()), 50)
    ax.loglog(xx, np.exp(c[1]) * xx ** c[0], "-", color=INK2, lw=0.9,
              label=rf"$|R|\propto e^{{{c[0]:.2f}}}$")
    ax.set_xlabel(r"measured eccentricity $e$")
    ax.set_ylabel(r"orbit-averaged $|R|$")
    ax.legend(loc="upper left", fontsize=7)
    gridify(ax)
    fig.savefig(os.path.join(FIG, "fig_first_law.pdf"))
    plt.close(fig)
    print("  fig_first_law.pdf")


# ================================================== Figure: detector band
def fig_detectability():
    det = load("detectability.json")
    if not det:
        print("  [skip] no detectability.json")
        return
    fig, ax = plt.subplots(figsize=(ONECOL, 2.8))
    L = np.array([v["length_M"] for v in det.values()])
    M = np.array([v["M_max_Msun"] for v in det.values()])
    pr = np.array([v["precessing"] for v in det.values()])
    ax.loglog(L[pr], M[pr], "s", color=C[1], ms=5, mec="white", mew=0.6,
              label="precessing", zorder=3)
    ax.loglog(L[~pr], M[~pr], "o", color=C[0], ms=5, mec="white", mew=0.6,
              label="aligned spin", zorder=3)
    for k, v in det.items():
        if v["length_M"] > 4e4:
            ax.annotate(k, (v["length_M"], v["M_max_Msun"]), fontsize=6,
                        xytext=(-4, 6), textcoords="offset points", ha="right",
                        color=INK2)
    ax.set_xlabel(r"waveform length $[M]$")
    ax.set_ylabel(r"$M_{\rm max}$ for $f_{\rm start}\leq 5$ Hz $[M_\odot]$")
    ax.legend(loc="upper right", fontsize=7)
    gridify(ax)
    fig.savefig(os.path.join(FIG, "fig_detectability.pdf"))
    plt.close(fig)
    print("  fig_detectability.pdf")


if __name__ == "__main__":
    fig_coprecessing()
    fig_twist()
    fig_first_law()
    fig_detectability()
