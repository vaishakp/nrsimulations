#!/usr/bin/env python3
"""Figure: what the initial-data-frame convention costs.

Three panels, made to sit next to the reference-frame discussion:

  (a) angle between the tabulated reference spins and the same vectors rotated
      into the reference frame, over the public SXS catalogue;
  (b) the same for the remnant kick;
  (c) for this catalogue, the RMS mode difference between the two frames,
      divided by the RMS difference between two resolutions of the same
      simulation -- i.e. how the frame convention compares with the numerical
      error of the waveform it is applied to.

(a) and (b) come from data/sxs_frame_impact.json (metadata only; run
sxs_frame_impact.py first).  (c) is computed here from the local waveform
products: `extrapolated/` is in the initial-data frame, `transformed/` is the
same waveform rotated to t_ref, so their difference isolates the convention.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_dir import FIGURES as FIG
from coprecessing import load_modes

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
TWOCOL = 7.0


def gridify(ax):
    ax.grid(True, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def path_for(name, lev, kind):
    if kind == "transformed":
        return os.path.join(
            WF, f"{name}_waveforms_Lev{lev}", "transformed",
            "rhOverM_extrapolated_CoM_transformed_N2.h5")
    return os.path.join(WF, f"{name}_waveforms_Lev{lev}", "extrapolated",
                        "rhOverM_Extrapolated_N2_CoM.h5")


def rms_diff(tA, mA, tB, mB):
    """RMS over the shared inspiral of the ell=2 complex mode difference."""
    t0, t1 = max(tA[0], tB[0]), min(tA[-1], tB[-1])
    if not (t1 > t0 + 100):
        return None
    g = np.linspace(t0 + 50, t1 - 50, 4000)
    tot = 0.0
    for key in mA:
        if key not in mB:
            continue
        a = np.interp(g, tA, mA[key].real) + 1j * np.interp(g, tA, mA[key].imag)
        b = np.interp(g, tB, mB[key].real) + 1j * np.interp(g, tB, mB[key].imag)
        tot += np.mean(np.abs(a - b) ** 2)
    return float(np.sqrt(tot)) if tot > 0 else None


def panel_c_data():
    levs = {}
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
        levs.setdefault(name, []).append(int(lev))
    rows = []
    for name, ls in sorted(levs.items()):
        ls = sorted(ls)
        if len(ls) < 2:
            continue
        lo, hi = ls[-2], ls[-1]
        try:
            tI, mI = load_modes(path_for(name, hi, "extrapolated"), ell_max=2)
            tR, mR = load_modes(path_for(name, hi, "transformed"), ell_max=2)
            tL, mL = load_modes(path_for(name, lo, "transformed"), ell_max=2)
        except Exception:
            continue
        frame = rms_diff(tI, mI, tR, mR)
        res = rms_diff(tR, mR, tL, mL)
        if not frame or not res:
            continue
        rows.append((name, frame / res))
        print(f"  {name:22s} frame/resolution = {frame / res:10.3g}")
    return rows


def main():
    p = os.path.join(DATA, "sxs_frame_impact.json")
    if not os.path.exists(p):
        print("  [skip] run sxs_frame_impact.py first")
        return
    sx = json.load(open(p))

    fig, axes = plt.subplots(1, 3, figsize=(TWOCOL, 2.35))

    # (a) spins
    ax = axes[0]
    bins = np.linspace(0, 180, 60)
    for tag, lab, c in (("chiA", r"$\vec\chi_1$", C[0]),
                        ("chiB", r"$\vec\chi_2$", C[3])):
        d = np.degrees([r[tag] for r in sx if r.get(tag) is not None])
        ax.hist(d, bins=bins, color=c, alpha=0.6, label=lab)
    ax.set_xlabel(r"$\delta\theta$ [deg]")
    ax.set_ylabel("simulations")
    # aligned-spin systems pile up at zero: there the rotation is about
    # z-hat, which leaves a spin parallel to L unchanged.  Log scale so that
    # spike does not flatten the tail, which is the part that matters.
    ax.set_yscale("log")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(loc="upper right")
    gridify(ax)
    ax.set_title("(a) spins, SXS catalogue", loc="left")

    # (b) kick
    ax = axes[1]
    d = np.degrees([r["kick"] for r in sx if r.get("kick") is not None])
    ax.hist(d, bins=bins, color=C[1], alpha=0.75)
    ax.axvline(np.median(d), color=INK2, lw=0.8, ls="--")
    ax.text(np.median(d) + 5, ax.get_ylim()[1] * 0.35,
            f"median {np.median(d):.0f}$^\\circ$", fontsize=7, color=INK2)
    ax.set_xlabel(r"$\theta_{\rm kick}$ [deg]")
    ax.set_yscale("log")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    gridify(ax)
    ax.set_title("(b) remnant kick, SXS catalogue", loc="left")

    # (c) frame error against numerical error, this catalogue
    ax = axes[2]
    rows = panel_c_data()
    if rows:
        r = np.array([x[1] for x in rows])
        ax.hist(r, bins=np.logspace(np.log10(max(r.min(), 1e-2)),
                                    np.log10(r.max()), 18),
                color=C[2], alpha=0.75)
        ax.set_xscale("log")
        ax.axvline(1.0, color=INK2, lw=0.9, ls=":")
        ax.text(1.25, ax.get_ylim()[1] * 0.85, "frame error\n= numerical",
                fontsize=6.4, color=INK2, va="top")
        print(f"\n  panel (c): n={len(r)}, median ratio {np.median(r):.3g}, "
              f"min {r.min():.3g}, max {r.max():.3g}")
    ax.set_xlabel("frame diff. / resolution diff.")
    gridify(ax)
    ax.set_title("(c) this catalogue", loc="left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_frame_impact.pdf"))
    plt.close(fig)
    print("  fig_frame_impact.pdf")
    emit_macros(sx, rows)




def emit_macros(sx, rows):
    """Numbers quoted in the text, so they regenerate with the figure."""
    from paper_dir import PAPER
    sp = np.degrees([r[t] for r in sx for t in ("chiA", "chiB")
                     if r.get(t) is not None])
    kk = np.degrees([r["kick"] for r in sx if r.get("kick") is not None])
    ratios = np.array([x[1] for x in rows]) if rows else np.array([])
    M = {
        "SXSNFrame": f"{len(sx)}",
        "SXSFrameSpinMed": f"{np.median(sp):.0f}",
        "SXSFrameKickMed": f"{np.median(kk):.0f}",
        "SXSFrameSpinBigFrac": f"{100 * (sp > 90).mean():.0f}",
        "SXSFrameKickBigFrac": f"{100 * (kk > 90).mean():.0f}",
        "NFrameTot": f"{len(ratios)}",
        "NFrameAboveRes": f"{int((ratios > 1).sum())}",
        "FrameResRatioMed": f"{np.median(ratios):.0f}",
    }
    with open(os.path.join(PAPER, "macros_frame.tex"), "w") as f:
        f.write("% autogenerated by scripts/frame_impact_figure.py\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("  wrote macros_frame.tex")
    for k, v in M.items():
        print(f"    {k:22} {v}")


if __name__ == "__main__":
    main()
