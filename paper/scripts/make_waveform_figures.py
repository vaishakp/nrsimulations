#!/usr/bin/env python3
"""Waveform and convergence figures.

Convergence is quantified in the standard way for SpEC: the difference between
consecutive adaptive-mesh-refinement levels (Lev2 vs Lev3) after a time-and-phase
alignment over an early inspiral window, plus the difference between successive
polynomial orders of the radius extrapolation (N2 vs N3), which bounds the
extraction error.
"""
import json
import os
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import minimize_scalar

from paper_dir import FIGURES as FIG

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
WF = os.path.join(HERE, "..", "waveforms")

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


def load_mode(path, l=2, m=2):
    with h5py.File(path, "r") as f:
        key = f"Y_l{l}_m{m}.dat"
        if key not in f:
            key = f"Y_l{l}_m{m}"
        d = f[key][()]
    return d[:, 0], d[:, 1] + 1j * d[:, 2]


def load_all(path, ellmax=4):
    out, t = {}, None
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("Y_l"):
                continue
            b = k[:-4] if k.endswith(".dat") else k
            lp, mp = b.split("_")[1:3]
            l, mm = int(lp[1:]), int(mp[1:])
            if l > ellmax:
                continue
            d = f[k][()]
            if t is None:
                t = d[:, 0].copy()
            out[(l, mm)] = d[:, 1] + 1j * d[:, 2]
    return t, out


def amp_phase(t, h):
    return np.abs(h), np.unwrap(np.angle(h))


def align_and_diff(tA, hA, tB, hB, frac=(0.05, 0.25)):
    """Align B onto A by (dt, dphi) over an early window; return common t and dphi(t).

    The window is taken as a fraction of the inspiral, after the junk-radiation
    transient and well before merger.
    """
    ampA, phA = amp_phase(tA, hA)
    ampB, phB = amp_phase(tB, hB)
    tpkA = tA[np.argmax(ampA)]
    tpkB = tB[np.argmax(ampB)]
    # shift B so the merger times coincide to start with
    dt0 = tpkA - tpkB
    t0 = max(tA[0], tB[0] + dt0) + 200.0
    t1 = tpkA - 50.0
    if not (t1 > t0):
        return None
    w0, w1 = t0 + frac[0] * (t1 - t0), t0 + frac[1] * (t1 - t0)
    grid = np.linspace(t0, t1, 20000)
    win = (grid >= w0) & (grid <= w1)

    fA = np.interp(grid, tA, phA)

    def resid(dt):
        fB = np.interp(grid, tB + dt0 + dt, phB)
        d = fA[win] - fB[win]
        return float(np.var(d))

    r = minimize_scalar(resid, bounds=(-150.0, 150.0), method="bounded",
                        options={"xatol": 1e-3})
    dt = r.x
    fB = np.interp(grid, tB + dt0 + dt, phB)
    dphi0 = np.mean(fA[win] - fB[win])
    dphi = fA - (fB + dphi0)
    aA = np.interp(grid, tA, ampA)
    aB = np.interp(grid, tB + dt0 + dt, ampB)
    return dict(t=grid, dphi=dphi, damp_rel=(aA - aB) / np.maximum(aA, 1e-30),
                tpk=tpkA, dt=dt0 + dt, ampA=aA)


def sim_path(name, lev, kind="extrapolated", n=2):
    d = os.path.join(WF, f"{name}_waveforms_Lev{lev}")
    if kind == "extrapolated":
        return os.path.join(d, "extrapolated", f"rhOverM_Extrapolated_N{n}_CoM.h5")
    return os.path.join(d, "transformed", "rhOverM_extrapolated_CoM_transformed_N2.h5")


# ============================================================ Figure: waveform
def fig_record_waveform(name="EccPrecDiff002", lev=3):
    p = sim_path(name, lev)
    if not os.path.exists(p):
        print(f"  [skip] {p}")
        return
    t, modes = load_all(p, ellmax=4)
    h22 = modes[(2, 2)]
    amp = np.abs(h22)
    tpk = t[np.argmax(amp)]
    tt = t - tpk

    fig = plt.figure(figsize=(TWOCOL, 3.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.58, wspace=0.34)

    # full waveform
    ax = fig.add_subplot(gs[0, :])
    ax.plot(tt, h22.real, color=C[0], lw=0.35, zorder=3)
    ax.plot(tt, amp, color=C[1], lw=0.9, zorder=4)
    ax.plot(tt, -amp, color=C[1], lw=0.9, zorder=4)
    ax.set_xlim(tt[0], 120)
    ax.set_ylabel(r"$r\,h_{22}/M$")
    ax.set_title(f"(a) {name} Lev{lev}: full $(2,2)$ mode, "
                 r"$\sim$80 orbits (159 GW cycles)", loc="left")
    ax.text(0.015, 0.90, r"$\mathrm{Re}\,h_{22}$", color=C[0],
            transform=ax.transAxes, fontsize=8)
    ax.text(0.015, 0.06, r"$|h_{22}|$", color=C[1],
            transform=ax.transAxes, fontsize=8)
    gridify(ax)

    # early inspiral zoom (eccentricity visible)
    ax = fig.add_subplot(gs[1, 0])
    sel = (tt > tt[0]) & (tt < tt[0] + 4000)
    ax.plot(tt[sel], h22.real[sel], color=C[0], lw=0.5)
    ax.plot(tt[sel], amp[sel], color=C[1], lw=1.0)
    ax.set_xlabel(r"$(t-t_{\rm peak})/M$")
    ax.set_ylabel(r"$r\,h_{22}/M$")
    ax.set_title("(b) early inspiral", loc="left")
    gridify(ax)

    # merger zoom
    ax = fig.add_subplot(gs[1, 1])
    sel = (tt > -450) & (tt < 110)
    ax.plot(tt[sel], h22.real[sel], color=C[0], lw=0.8)
    ax.plot(tt[sel], amp[sel], color=C[1], lw=1.0)
    ax.set_xlabel(r"$(t-t_{\rm peak})/M$")
    ax.set_title("(c) merger and ringdown", loc="left")
    gridify(ax)

    # mode hierarchy
    ax = fig.add_subplot(gs[1, 2])
    order = [(2, 2), (2, 1), (3, 3), (4, 4)]
    for i, lm in enumerate(order):
        if lm not in modes:
            continue
        # plot the envelope: the raw |h_lm| oscillate too fast to read here
        y = np.abs(modes[lm])
        w = max(1, len(y) // 900)
        n = (len(y) // w) * w
        ye = y[:n].reshape(-1, w).max(axis=1)
        xe = tt[:n].reshape(-1, w).mean(axis=1)
        ax.semilogy(xe, ye, color=C[i % len(C)], lw=0.9,
                    label=rf"$({lm[0]},{lm[1]})$")
    ax.set_xlim(-25000, 120)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel(r"$(t-t_{\rm peak})/M$")
    ax.set_ylabel(r"$|h_{\ell m}|$ (envelope)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.12),
              handlelength=0.9, columnspacing=0.7, handletextpad=0.35)
    ax.set_title("(d) mode amplitudes", loc="left")
    gridify(ax)

    fig.savefig(os.path.join(FIG, "fig_record_waveform.pdf"))
    plt.close(fig)
    print("  fig_record_waveform.pdf")


# ========================================================= Figure: convergence
def fig_convergence():
    cands = []
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        b = os.path.basename(d)
        name, lev = b.rsplit("_waveforms_Lev", 1)
        cands.append((name, int(lev)))
    bylev = {}
    for n, l in cands:
        if os.path.exists(sim_path(n, l)):
            bylev.setdefault(n, []).append(l)
    pairs = [(n, sorted(v)) for n, v in bylev.items() if len(v) >= 2]

    results = []
    for name, levs in sorted(pairs):
        # the two highest levels for which data actually exist
        lo, hi = levs[-2], levs[-1]
        pa, pb = sim_path(name, hi), sim_path(name, lo)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        try:
            tA, hA = load_mode(pa)
            tB, hB = load_mode(pb)
            r = align_and_diff(tA, hA, tB, hB)
            if r is None:
                continue
            r["name"], r["lo"], r["hi"] = name, lo, hi
            # extrapolation-order error at the finer level: N2 vs N3
            p3 = sim_path(name, hi, n=3)
            if os.path.exists(p3):
                t3, h3 = load_mode(p3)
                r2 = align_and_diff(tA, hA, t3, h3)
                if r2 is not None:
                    r["extrap"] = r2
            results.append(r)
        except Exception as e:
            print(f"  [warn] {name}: {e}")

    if not results:
        print("  [skip] no convergence pairs available yet")
        return

    json.dump(
        [{"name": r["name"], "lo": r["lo"], "hi": r["hi"],
          "dphi_at_merger": float(np.interp(r["tpk"], r["t"], np.abs(r["dphi"]))),
          "dphi_max_inspiral": float(np.max(np.abs(
              r["dphi"][r["t"] < r["tpk"] - 200]))) if np.any(r["t"] < r["tpk"] - 200) else None,
          "time_shift": float(r["dt"]),
          "dphi_extrap": (float(np.max(np.abs(
              r["extrap"]["dphi"][r["extrap"]["t"] < r["extrap"]["tpk"] - 200])))
              if "extrap" in r else None)}
         for r in results],
        open(os.path.join(DATA, "convergence.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(TWOCOL, 2.7))

    # (a) phase difference vs time for a representative subset
    ax = axes[0]
    orb = json.load(open(os.path.join(DATA, "orbit_counts.json")))["summary"]
    pick = sorted(results, key=lambda r: -np.max(np.abs(r["dphi"])))
    show = []
    for want in ("EccPrecDiff002", "EccContPrecDiff001", "ICTSEccParallel02",
                 "ICTSEccParallel07", "EccContPrecDiff005"):
        for r in results:
            if r["name"] == want:
                show.append(r)
    if not show:
        show = pick[:4]
    for i, r in enumerate(show[:5]):
        tt = r["t"] - r["tpk"]
        o = orb.get(r["name"], {}).get("orbits")
        lab = rf"{r['name']} (L{r['lo']}$\to${r['hi']}"
        lab += rf", {o:.0f} orb)" if o else ")"
        ax.semilogy(tt, np.abs(r["dphi"]), color=C[i], lw=1.0, label=lab)
    ax.axhline(1.0, color=INK2, lw=0.7, ls=":")
    ax.text(0.30, 0.645, "1 rad", transform=ax.transAxes, fontsize=7,
            color=INK2, ha="left", va="bottom")
    ax.set_xlabel(r"$(t-t_{\rm peak})/M$")
    ax.set_ylabel(r"$|\Delta\phi_{22}|$ [rad]")
    ax.set_ylim(1e-5, 800)
    ax.legend(loc="upper left", handlelength=1.3, fontsize=6.4,
              labelspacing=0.25)
    gridify(ax)
    ax.set_title("(a) resolution convergence", loc="left")

    # (b) accumulated phase error vs waveform length
    ax = axes[1]
    cat = json.load(open(os.path.join(DATA, "catalog.json")))
    orb = json.load(open(os.path.join(DATA, "orbit_counts.json")))["summary"]
    xs, ys, ys2, nm = [], [], [], []
    for r in results:
        e = cat.get(r["name"], {})
        o = orb.get(r["name"])
        n = o["orbits"] if o else (e.get("ncycles_max") or 0) / 2.0
        if not n:
            continue
        pre = r["t"] < r["tpk"] - 200
        if not np.any(pre):
            continue
        xs.append(n)
        ys.append(np.max(np.abs(r["dphi"][pre])))
        ys2.append(np.max(np.abs(r["extrap"]["dphi"][r["extrap"]["t"]
                                                     < r["extrap"]["tpk"] - 200]))
                   if "extrap" in r else np.nan)
        nm.append(r["name"])
    ax.scatter(xs, ys, c=C[0], s=32, edgecolor="white", linewidth=0.6,
               label="resolution (Lev$\\to$Lev+1)", zorder=3)
    good = ~np.isnan(ys2)
    if good.any():
        ax.scatter(np.array(xs)[good], np.array(ys2)[good], c=C[1], marker="s",
                   s=32, edgecolor="white", linewidth=0.6,
                   label="extrapolation order ($N$=2 vs 3)", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("orbits to merger")
    ax.set_ylabel(r"max $|\Delta\phi_{22}|$ over inspiral [rad]")
    ax.legend(loc="upper left", fontsize=7, handletextpad=0.3)
    gridify(ax)
    ax.set_title("(b) phase error vs length", loc="left")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIG, "fig_convergence.pdf"))
    plt.close(fig)
    print("  fig_convergence.pdf")
    print(f"    {len(results)} simulations with >=2 levels")
    for r in sorted(results, key=lambda r: r["name"]):
        pre = r["t"] < r["tpk"] - 200
        v = np.max(np.abs(r["dphi"][pre])) if np.any(pre) else float("nan")
        print(f"      {r['name']:<20} L{r['lo']}->{r['hi']}  "
              f"max|dphi|_inspiral = {v:7.4f} rad   dt = {r['dt']:8.3f} M")


if __name__ == "__main__":
    fig_record_waveform()
    fig_convergence()
