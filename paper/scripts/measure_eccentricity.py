#!/usr/bin/env python3
"""Measure the eccentricity of every catalogue waveform directly.

The `ecc` column of the published parameter table is an initial-data input and
does not always describe the waveform that came out.  This script measures
eccentricity from the waveform itself, using `gw_eccentricity` (Shaikh et al.,
arXiv:2302.11257), so the table can be regenerated from data.

For the precessing simulations the measurement is made on the co-precessing
frame (2, +-2) combination of Eqs. (48)-(49) of arXiv:1701.00550,

    amp_gw   = (|h_22| + |h_2-2|) / 2
    phase_gw = (arg h_22 - arg h_2-2) / 2

which `gw_eccentricity` constructs internally when `precessing=True`.  This is
the same frame used by the co-precessing-frame study in `coprecessing.py`.

Several methods are run per simulation: agreement between them is the error
estimate, and disagreement flags a measurement that should not be quoted.
"""
import os
import sys
import json
import glob
import warnings
import numpy as np
import h5py

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(HERE, "..", "waveforms")
DATA = os.path.join(HERE, "..", "data")

# Frequency/Amplitude need no model; the Residual* methods need a quasi-circular
# counterpart we do not have for these configurations.
METHODS = ["Amplitude", "Frequency", "AmplitudeFits", "FrequencyFits"]


def load_modes(path, ell_max=2):
    out, t = {}, None
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("Y_l"):
                continue
            b = k[:-4] if k.endswith(".dat") else k
            lp, mp = b.split("_")[1:3]
            l, m = int(lp[1:]), int(mp[1:])
            if l > ell_max:
                continue
            d = f[k][()]
            if t is None:
                t = d[:, 0].copy()
            out[(l, m)] = d[:, 1] + 1j * d[:, 2]
    return t, out


def uniform_resample(t, modes, dt=None):
    """gw_eccentricity wants a uniformly sampled time array."""
    if dt is None:
        dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    out = {}
    for lm, h in modes.items():
        out[lm] = (np.interp(tu, t, h.real) + 1j * np.interp(tu, t, h.imag))
    return tu, out


def trim_junk(t, modes, t_junk=250.0):
    s = t > t[0] + t_junk
    return t[s], {lm: h[s] for lm, h in modes.items()}


def coprecessing_h_eff(t, modes):
    """The precessing amp_gw/phase_gw combination, packaged as a (2,2) mode.

    `gw_eccentricity`'s own `precessing=True` path builds exactly this, but it
    first runs `check_and_filter_spin_induced_oscillations`, whose PN secular
    fit raises "Initial guess is outside of provided bounds" on every waveform
    in this catalogue (gw_eccentricity 2.1.0, `spin_filter.py:140`).  We
    therefore construct the combination ourselves from the co-precessing modes
    and hand it over as a non-precessing waveform, which reaches the identical
    estimator by a code path that works.
    """
    import sys as _sys
    _sys.path.insert(0, HERE)
    from coprecessing import to_waveform, coprecessing as to_cp

    W = to_waveform(t, modes, ell_min=2, ell_max=2)
    Wc, _ = to_cp(W)
    h22 = Wc.data[:, Wc.index(2, 2)]
    h2m2 = Wc.data[:, Wc.index(2, -2)]
    amp_gw = 0.5 * (np.abs(h22) + np.abs(h2m2))
    phase_gw = 0.5 * (np.unwrap(np.angle(h22)) - np.unwrap(np.angle(h2m2)))
    # Decide the sense from the inspiral, not from the endpoints: the final
    # samples are ringdown, where the co-precessing phase is not monotonic.
    if np.median(np.gradient(phase_gw)) < 0:
        phase_gw = -phase_gw
    # gw_eccentricity takes phase_gw = -unwrap(angle(h22)) internally
    # (eccDefinition.py:692) and then demands it increase, so hand back the
    # conjugate convention.
    return amp_gw * np.exp(-1j * phase_gw)


def measure_one(name, lev, path, precessing):
    import gw_eccentricity as ge

    ell_max = 2
    t, modes = load_modes(path, ell_max=ell_max)
    t, modes = trim_junk(t, modes)
    t, modes = uniform_resample(t, modes)
    if precessing:
        h_eff = coprecessing_h_eff(t, modes)
    else:
        h_eff = modes[(2, 2)]
    dataDict = {"t": t, "hlm": {(2, 2): h_eff}}

    res = {}
    for meth in METHODS:
        try:
            g = ge.measure_eccentricity(
                tref_in=t, method=meth, dataDict=dataDict,
                precessing=False, frame="inertial",
                num_orbits_to_exclude_before_merger=2)
            e = np.atleast_1d(g["eccentricity"])
            tr = np.atleast_1d(g["tref_out"])
            res[meth] = dict(
                ecc_early=float(e[0]), ecc_late=float(e[-1]),
                t_early=float(tr[0]), t_late=float(tr[-1]),
                n_peri=int(len(g.get("pericenters_location", []) or [])),
            )
        except Exception as e:
            res[meth] = dict(error=f"{type(e).__name__}: {e}"[:160])
    return res, t


def main():
    cat = json.load(open(os.path.join(DATA, "catalog.json")))
    best = {}
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
        p = os.path.join(d, "transformed",
                         "rhOverM_extrapolated_CoM_transformed_N2.h5")
        if os.path.exists(p) and int(lev) > best.get(name, (0, None))[0]:
            best[name] = (int(lev), p)

    out = {}
    hdr = f"{'name':22s} {'L':>2s} {'prec':>5s} " + "".join(
        f"{m[:9]:>10s}" for m in METHODS) + f"{'spread':>9s}{'table':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, (lev, p) in sorted(best.items()):
        prec = bool(cat.get(name, {}).get("precessing"))
        try:
            res, t = measure_one(name, lev, p, prec)
        except Exception as e:
            print(f"{name:22s} L{lev}  FAILED {type(e).__name__}: {e}"[:120])
            continue
        vals = [r["ecc_early"] for r in res.values() if "ecc_early" in r]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        out[name] = dict(lev=lev, precessing=prec, methods=res,
                         ecc_median=float(np.median(vals)) if vals else None,
                         ecc_spread=float(spread) if vals else None,
                         ecc_table=cat.get(name, {}).get("ecc"))
        cells = "".join(
            (f"{res[m]['ecc_early']:>10.4f}" if "ecc_early" in res[m]
             else f"{'--':>10s}") for m in METHODS)
        te = cat.get(name, {}).get("ecc")
        print(f"{name:22s} L{lev} {str(prec):>5s} {cells}{spread:>9.4f}"
              f"{(f'{te:.2f}' if te is not None else '--'):>7s}")

    json.dump(out, open(os.path.join(DATA, "eccentricity.json"), "w"), indent=1)
    print(f"\nwrote data/eccentricity.json ({len(out)} simulations)")


if __name__ == "__main__":
    main()
