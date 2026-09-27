#!/usr/bin/env python3
"""First-law residue for the catalogue waveforms.

The first law of binary black hole mechanics relates the binding energy and
angular momentum of a quasi-circular binary through the orbital frequency,

    dE / dJ = Omega .

On a shrinking orbit that becomes a statement about the radiated fluxes: the
energy and angular momentum carried off by the waves must be in the ratio
Omega.  `nrhjsurrogate` uses exactly this form (`_first_law_omega` in
`physics/action_extend.py`, `omega = |E'/J'|`) to anchor its inspiral seam.

We evaluate it as a diagnostic of the waveforms themselves:

    R(t) = Omega_dot_E_over_J(t) / Omega_gw(t) / 2 - 1,
           Omega_dot_E_over_J = (dE/dt) / (dJ_z/dt)

where Omega_gw is the (2,2) frequency, so that Omega_gw/2 is the orbital
frequency.  For a quasi-circular inspiral R -> 0.  For an eccentric binary the
first law holds only in an orbit-averaged sense, so R oscillates at the radial
period with an amplitude that grows with eccentricity; that oscillation is
physics, not error, and we report the orbit-averaged residue separately from
the oscillation amplitude.

Fluxes are computed from the strain with `scri`, which uses

    dE/dt   = (1/16pi) sum_lm |hdot_lm|^2
    dJ_z/dt = (1/16pi) Im[ sum_lm m hbar_lm hdot_lm ] .
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import (WF, DATA, sim_path, load_modes, to_waveform,
                          coprecessing)

HERE = os.path.dirname(os.path.abspath(__file__))


def fluxes(W, axis=None):
    """(dE/dt, dJ/dt . axis) on W's time grid.

    `axis` is the direction conjugate to the orbital phase.  For a
    non-precessing binary that is a fixed z; for a precessing one it is the
    instantaneous co-precessing z-axis, and using inertial z instead mixes in
    the precession of the orbital plane and spoils the residue.
    """
    edot = np.asarray(W.energy_flux())
    jdot = np.asarray(W.angular_momentum_flux())
    if axis is None:
        return edot, jdot[:, 2]
    return edot, np.einsum("ij,ij->i", jdot, axis)


def coprecessing_axis(R):
    """The image of z-hat under the frame rotor R(t), as an (N,3) array."""
    import quaternion
    z = np.quaternion(0, 0, 0, 1)
    return quaternion.as_float_array(R * z * np.conjugate(R))[:, 1:]


def omega_gw(t, h22):
    return np.gradient(np.unwrap(np.angle(h22)), t)


def residue(name, lev, ell_max=8, decimate=None):
    t, modes = load_modes(sim_path(name, lev), ell_max=ell_max)
    if decimate is None:
        decimate = max(1, int(round(0.5 / np.median(np.diff(t)))))
    W = to_waveform(t, modes, ell_max=ell_max, decimate=decimate)
    Wc, R = coprecessing(W)

    edot, jzdot = fluxes(W, axis=coprecessing_axis(R))
    # co-precessing (2,2) frequency -> orbital frequency
    om22 = np.abs(omega_gw(Wc.t, Wc.data[:, Wc.index(2, 2)]))
    om_orb = om22 / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        om_fl = np.abs(edot / jzdot)

    amp = np.abs(Wc.data[:, Wc.index(2, 2)])
    tpk = Wc.t[np.argmax(amp)]
    sel = (Wc.t > Wc.t[0] + 400) & (Wc.t < tpk - 300) & np.isfinite(om_fl)
    R = om_fl / om_orb - 1.0
    return dict(t=Wc.t, R=R, sel=sel, tpk=tpk, om_orb=om_orb, om_fl=om_fl,
                edot=edot, jzdot=jzdot)


def summarise(r, n_smooth=None):
    """Orbit-averaged residue and the amplitude of its radial oscillation."""
    t, R, sel = r["t"][r["sel"]], r["R"][r["sel"]], None
    if len(t) < 100:
        return None
    # smooth over ~1 orbit, then discard a full window at each end: a boxcar
    # this wide leaves large edge artefacts, and on the shorter runs the window
    # is a sizeable fraction of the record.
    from scipy.ndimage import uniform_filter1d
    dt = np.median(np.diff(t))
    w = int(max(11, min(len(t) // 8, (2 * np.pi / np.median(r["om_orb"])) / dt)))
    if w % 2 == 0:
        w += 1
    Rs = uniform_filter1d(R, w, mode="nearest")
    good = slice(w, -w)
    return dict(
        R_secular_med=float(np.median(np.abs(Rs[good]))),
        R_secular_max=float(np.max(np.abs(Rs[good]))),
        R_osc_rms=float(np.std((R - Rs)[good])),
        n=int(len(t)),
    )


def main():
    cat = json.load(open(os.path.join(DATA, "catalog.json")))
    ecc = json.load(open(os.path.join(DATA, "eccentricity.json"))) \
        if os.path.exists(os.path.join(DATA, "eccentricity.json")) else {}
    best = {}
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
        if os.path.exists(sim_path(name, int(lev))):
            best[name] = max(best.get(name, 0), int(lev))

    out = {}
    print(f"{'name':22s}{'L':>3s}{'e_meas':>8s}{'|R|_sec med':>13s}"
          f"{'|R|_sec max':>13s}{'R_osc rms':>11s}")
    print("-" * 70)
    for name, lev in sorted(best.items()):
        try:
            r = residue(name, lev)
            s = summarise(r)
            if s is None:
                print(f"{name:22s}{lev:>3d}  [too short]")
                continue
            e = (ecc.get(name) or {}).get("ecc_median")
            s["ecc"] = e
            s["lev"] = lev
            out[name] = s
            print(f"{name:22s}{lev:>3d}{(f'{e:.4f}' if e is not None else '--'):>8s}"
                  f"{s['R_secular_med']:>13.3e}{s['R_secular_max']:>13.3e}"
                  f"{s['R_osc_rms']:>11.3e}")
        except Exception as ex:
            print(f"{name:22s}{lev:>3d}  [warn] {type(ex).__name__}: {ex}"[:110])

    json.dump(out, open(os.path.join(DATA, "first_law.json"), "w"), indent=1)
    print(f"\nwrote data/first_law.json ({len(out)} simulations)")


if __name__ == "__main__":
    main()
