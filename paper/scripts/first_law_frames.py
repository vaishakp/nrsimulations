#!/usr/bin/env python3
"""First-law residue in the inertial *and* co-precessing frames.

`first_law.py` reports the co-precessing residue only, because that is the one
we quote.  This script computes both for every run so the two can be compared
and contrasted: it is the quantitative version of the single example in the
text (ICTSEccParallel01, 0.1% -> 4.7%).

The difference is entirely in how the angular momentum flux is projected and
where the orbital frequency is read off:

    co-precessing :  Jdot . zhat_cp(t)  /  (omega_22^cp / 2)
    inertial      :  Jdot . zhat        /  (omega_22^in / 2)

Edot is the same in both -- the energy flux is a physical, inertial quantity,
and differentiating in a time-dependent rotating frame would add spurious
terms to hdot.  For an aligned-spin binary the two frames coincide (the
co-precessing axis is zhat to ~1e-5 deg), so those runs double as a null test:
the ratio must come out at 1.
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import (WF, DATA, sim_path, load_modes, to_waveform,
                          coprecessing)
from first_law import fluxes, coprecessing_axis, omega_gw, summarise


def residues_both(name, lev, ell_max=8, decimate=None):
    """(co-precessing, inertial) residue dicts, from one load of the modes."""
    t, modes = load_modes(sim_path(name, lev), ell_max=ell_max)
    if decimate is None:
        decimate = max(1, int(round(0.5 / np.median(np.diff(t)))))
    W = to_waveform(t, modes, ell_max=ell_max, decimate=decimate)
    Wc, rot = coprecessing(W)

    # merger reference and window from the co-precessing amplitude, so that
    # both frames are averaged over exactly the same stretch of inspiral.
    amp = np.abs(Wc.data[:, Wc.index(2, 2)])
    tpk = Wc.t[np.argmax(amp)]

    out = {}
    for tag, axis, Wf in (("cp", coprecessing_axis(rot), Wc),
                          ("in", None, W)):
        edot, jdot = fluxes(W, axis=axis)
        om22 = np.abs(omega_gw(Wf.t, Wf.data[:, Wf.index(2, 2)]))
        om_orb = om22 / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            om_fl = np.abs(edot / jdot)
        sel = (Wf.t > Wf.t[0] + 400) & (Wf.t < tpk - 300) & np.isfinite(om_fl)
        out[tag] = dict(t=Wf.t, R=om_fl / om_orb - 1.0, sel=sel,
                        om_orb=om_orb, om_fl=om_fl)
    return out


def main():
    cat = json.load(open(os.path.join(DATA, "catalog.json")))
    ecc_path = os.path.join(DATA, "eccentricity.json")
    ecc = json.load(open(ecc_path)) if os.path.exists(ecc_path) else {}

    best = {}
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
        if os.path.exists(sim_path(name, int(lev))):
            best[name] = max(best.get(name, 0), int(lev))

    out = {}
    print(f"{'name':22s}{'L':>3s}{'e':>7s}{'|R|_cp':>11s}{'|R|_in':>11s}"
          f"{'ratio':>8s}  prec")
    print("-" * 68)
    for name, lev in sorted(best.items()):
        try:
            both = residues_both(name, lev)
            s_cp, s_in = summarise(both["cp"]), summarise(both["in"])
            if s_cp is None or s_in is None:
                print(f"{name:22s}{lev:>3d}  [too short]")
                continue
            prec = bool(cat.get(name, {}).get("precessing"))
            e = (ecc.get(name) or {}).get("ecc_median")
            rec = dict(lev=lev, ecc=e, precessing=prec,
                       R_cp=s_cp["R_secular_med"], R_in=s_in["R_secular_med"],
                       R_cp_max=s_cp["R_secular_max"],
                       R_in_max=s_in["R_secular_max"],
                       osc_cp=s_cp["R_osc_rms"], osc_in=s_in["R_osc_rms"],
                       ratio=s_in["R_secular_med"] / s_cp["R_secular_med"])
            out[name] = rec
            print(f"{name:22s}{lev:>3d}{(f'{e:.3f}' if e else '--'):>7s}"
                  f"{rec['R_cp']:>11.3e}{rec['R_in']:>11.3e}"
                  f"{rec['ratio']:>8.2f}  {prec}")
        except Exception as ex:
            print(f"{name:22s}{lev:>3d}  [warn] {type(ex).__name__}: {ex}"[:110])

    json.dump(out, open(os.path.join(DATA, "first_law_frames.json"), "w"),
              indent=1)
    print(f"\nwrote data/first_law_frames.json ({len(out)} simulations)")


if __name__ == "__main__":
    main()
