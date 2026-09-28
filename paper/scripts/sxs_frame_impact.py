#!/usr/bin/env python3
"""How much does leaving vectors in the initial-data frame actually matter?

The SXS catalogue quotes spins and the remnant kick *at* the reference time but
*in* the initial-data frame.  A user who feeds those numbers to a waveform
model expecting the LAL convention -- orbital angular momentum along +z,
separation along +x at t_ref -- is using components from the wrong frame.  This
script measures how large that error is across the public catalogue.

For each simulation we build the rotation R of Eq. (frame) from the reference
metadata,

    L_hat  from  reference_orbital_frequency   (Omega is parallel to L)
    n_hat  from  reference_position1 - reference_position2
    rows of R = (n_perp, L x n_perp, L)

and compare each vector v, as tabulated, against R v:

    dtheta      = angle between chi_ref (as tabulated) and R chi_ref
    theta_kick  = the same for the remnant velocity

Note this is *not* the same as comparing the initial and reference spins: that
difference is physical precession between t=0 and t_ref, which is real and not
an error.  The quantity here is purely the frame convention.

Writes data/sxs_frame_impact.json.  Metadata only -- no waveform downloads.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def vec3(x):
    try:
        v = np.asarray(x, dtype=float)
        return v if v.shape == (3,) and np.all(np.isfinite(v)) else None
    except (TypeError, ValueError):
        return None


def rotation(Lhat, nhat):
    """Rows (n_perp, L x n_perp, L): the unique proper rotation of Eq. (frame)."""
    n_perp = nhat - np.dot(nhat, Lhat) * Lhat
    nn = np.linalg.norm(n_perp)
    if nn < 1e-8:
        return None
    n_perp = n_perp / nn
    return np.vstack([n_perp, np.cross(Lhat, n_perp), Lhat])


def angle(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return None
    return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))


def main():
    import sxs
    sims = sxs.load("simulations")

    out = []
    for k in sims:
        if not k.startswith("SXS:BBH:"):
            continue
        m = sims[k]
        om = vec3(m.get("reference_orbital_frequency"))
        p1 = vec3(m.get("reference_position1"))
        p2 = vec3(m.get("reference_position2"))
        if om is None or p1 is None or p2 is None:
            continue
        nL, nn = np.linalg.norm(om), np.linalg.norm(p1 - p2)
        if nL < 1e-12 or nn < 1e-12:
            continue
        R = rotation(om / nL, (p1 - p2) / nn)
        if R is None:
            continue

        rec = {"sim": k}
        for tag, key in (("chiA", "reference_dimensionless_spin1"),
                         ("chiB", "reference_dimensionless_spin2")):
            v = vec3(m.get(key))
            if v is not None and np.linalg.norm(v) > 1e-3:
                rec[tag] = angle(v, R @ v)
        v = vec3(m.get("remnant_velocity"))
        if v is not None:
            rec["kick"] = angle(v, R @ v)
        if len(rec) > 1:
            out.append(rec)

    json.dump(out, open(os.path.join(DATA, "sxs_frame_impact.json"), "w"))
    print(f"wrote data/sxs_frame_impact.json ({len(out)} simulations)")

    for tag, lab in (("chiA", "spin 1"), ("chiB", "spin 2"),
                     ("kick", "remnant kick")):
        d = np.array([r[tag] for r in out if r.get(tag) is not None])
        if not len(d):
            continue
        print(f"  {lab:14s} n={len(d):5d}  median {np.degrees(np.median(d)):6.1f} deg"
              f"   >10 deg: {(d > np.radians(10)).mean():5.1%}"
              f"   >90 deg: {(d > np.pi/2).mean():5.1%}")


if __name__ == "__main__":
    main()
