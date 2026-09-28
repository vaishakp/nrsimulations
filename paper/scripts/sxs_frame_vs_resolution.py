#!/usr/bin/env python3
"""RMS(dh_ref) / RMS(h_res) over the SXS catalogue.

Panel (c) of the frame-impact figure: is the initial-data-frame convention a
larger error than the numerical error of the waveform it is applied to?

    dh_ref = h  -  R[h]      modes as published, minus the same modes with the
                             decomposition basis rotated into the t_ref frame
    h_res  = h(Lev_hi) - h(Lev_lo)

and we histogram RMS(dh_ref) / RMS(h_res).  Above one, the frame convention
costs more than the resolution.

The rotation R is the one of Eq. (frame), built from the reference metadata
exactly as in sxs_frame_impact.py, and applied to the mode decomposition with
the Wigner matrices of the corresponding quaternion -- so no `transformed`
products are needed, only the published strain at two levels.

This downloads two strain files per simulation (a few MB each), so it takes a
sample rather than the whole catalogue; pass a size as the first argument.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DEFAULT_N = 150


def vec3(x):
    try:
        v = np.asarray(x, dtype=float)
        return v if v.shape == (3,) and np.all(np.isfinite(v)) else None
    except (TypeError, ValueError):
        return None


def frame_quaternion(m):
    """Quaternion of the rotation taking (L, n_perp) to (z, x).

    Returned as a `quaternionic.array`, which is what WaveformModes.rotate
    expects; `numpy-quaternion` objects are rejected with a TypeError.
    """
    import quaternionic
    om = vec3(m.get("reference_orbital_frequency"))
    p1 = vec3(m.get("reference_position1"))
    p2 = vec3(m.get("reference_position2"))
    if om is None or p1 is None or p2 is None:
        return None
    nL = np.linalg.norm(om)
    d = p1 - p2
    nd = np.linalg.norm(d)
    if nL < 1e-12 or nd < 1e-12:
        return None
    L = om / nL
    n = d / nd
    n_perp = n - np.dot(n, L) * L
    if np.linalg.norm(n_perp) < 1e-8:
        return None
    n_perp /= np.linalg.norm(n_perp)
    # rows of R are (n_perp, L x n_perp, L)
    R = np.vstack([n_perp, np.cross(L, n_perp), L])
    return quaternionic.array.from_rotation_matrix(R)


def load(sim, lev):
    import sxs
    last = None
    for ver in ("v3.0", "v2.0", ""):
        try:
            return sxs.load(f"{sim}{ver}/Lev{lev}").h
        except Exception as ex:
            last = ex
    raise last


def rms_on(grid, w, ref_t):
    d = np.asarray(w.data)
    out = 0.0
    for i in range(d.shape[1]):
        c = np.interp(grid, ref_t, d[:, i].real) \
            + 1j * np.interp(grid, ref_t, d[:, i].imag)
        out += np.mean(np.abs(c) ** 2)
    return out


def ratio_for(sim, levs, m):
    q = frame_quaternion(m)
    if q is None:
        return None
    hi, lo = levs[-1], levs[-2]
    wh = load(sim, hi)
    wl = load(sim, lo)

    # restrict to ell=2 and a common inspiral window
    t0 = max(wh.t[0], wl.t[0]) + 100.0
    t1 = min(wh.t[-1], wl.t[-1]) - 100.0
    if not (t1 > t0 + 500):
        return None
    grid = np.linspace(t0, t1, 3000)

    rot = wh.rotate(q)
    num = den = 0.0
    for (l, mm) in [(2, x) for x in range(-2, 3)]:
        ih, il, ir = wh.index(l, mm), wl.index(l, mm), rot.index(l, mm)
        a = np.interp(grid, wh.t, wh.data[:, ih].real) \
            + 1j * np.interp(grid, wh.t, wh.data[:, ih].imag)
        r = np.interp(grid, rot.t, rot.data[:, ir].real) \
            + 1j * np.interp(grid, rot.t, rot.data[:, ir].imag)
        b = np.interp(grid, wl.t, wl.data[:, il].real) \
            + 1j * np.interp(grid, wl.t, wl.data[:, il].imag)
        num += np.mean(np.abs(a - r) ** 2)
        den += np.mean(np.abs(a - b) ** 2)
    if den <= 0 or num <= 0:
        return None
    return float(np.sqrt(num / den))


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    import sxs
    sims = sxs.load("simulations")

    cand = []
    for k in sims:
        if not k.startswith("SXS:BBH:"):
            continue
        m = sims[k]
        levs = sorted(m.get("lev_numbers") or [])
        if len(levs) >= 2 and frame_quaternion(m) is not None:
            cand.append((k, levs, m))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(cand))[:n_want]
    print(f"{len(cand)} candidates; sampling {len(idx)}", flush=True)

    out = []
    for j, i in enumerate(idx, 1):
        k, levs, m = cand[i]
        try:
            r = ratio_for(k, levs, m)
        except Exception as ex:
            print(f"  [{j:4d}] {k:18s} [warn] {type(ex).__name__}"[:90],
                  flush=True)
            continue
        if r is None:
            continue
        out.append({"sim": k, "levs": levs[-2:], "ratio": r})
        if j % 10 == 0 or j < 5:
            a = np.array([x["ratio"] for x in out])
            print(f"  [{j:4d}] {k:18s} ratio {r:9.3g}   "
                  f"running median {np.median(a):.2f}  n={len(a)}", flush=True)

    json.dump(out, open(os.path.join(DATA, "sxs_frame_vs_resolution.json"),
                        "w"), indent=1)
    a = np.array([x["ratio"] for x in out])
    print(f"\nn={len(a)}  median {np.median(a):.2f}  "
          f"above 1: {(a > 1).mean():.1%}  range {a.min():.3g}-{a.max():.3g}")
    print(f"wrote data/sxs_frame_vs_resolution.json")


if __name__ == "__main__":
    main()
