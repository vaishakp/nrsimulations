#!/usr/bin/env python3
"""Measure the per-level phase convergence factor f across the SXS catalogue.

The resolution forecast of Sec. "What next-generation detectors would demand"
needs the factor by which the (2,2) phase error falls per refinement level.
Our own catalogue cannot supply it: no simulation here has two *consecutive*
level pairs of waveform data, so the paper brackets f using the constraint
convergence of EccPrecDiff001 as a proxy.  That proxy is the weakest link in
the estimate.

SXS has thousands of simulations with three or more levels *and* published
waveforms at each, so f can be measured from actual phase differences.  For
each sampled simulation we take the three highest consecutive levels
(lo, mid, hi) and form

    f = max|dphi(lo, mid)| / max|dphi(mid, hi)| ,

with each phase difference computed exactly as for our own runs -- aligned over
an early window in (dt, dphi) and maximised over the inspiral -- so the numbers
are directly comparable with Table III.

The sample deliberately spans orbit count, because the question is not just
what f is but whether it degrades for long waveforms: our record run is 80
orbits, far longer than a typical SXS run, and if f falls with length then
borrowing an SXS value would flatter the forecast.

Downloads strain data into ~/.cache/sxs (a few MB per level).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_waveform_figures import amp_phase, align_and_diff

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER_BIN = 3
ORBIT_BINS = [(5, 15), (15, 25), (25, 40), (40, 70), (70, 200)]


def num(x):
    try:
        if isinstance(x, str):
            x = x.strip().lstrip("<>~")
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def mode22(sim, lev):
    """(t, h22) for one simulation at one level, from the SXS strain data.

    The load spec is "<id><version>/Lev<n>"; the version must be given
    explicitly for simulations that have been superseded, and the default
    (v2.0) is not always the one that carries the Levs we want, so try the
    current release first and fall back.
    """
    import sxs
    last = None
    for ver in ("v3.0", "v2.0", ""):
        try:
            h = sxs.load(f"{sim}{ver}/Lev{lev}").h
            return np.asarray(h.t), np.asarray(h.data[:, h.index(2, 2)])
        except Exception as ex:      # unavailable version/Lev combination
            last = ex
    raise last


def dphi_pair(sim, lo, hi):
    tA, hA = mode22(sim, hi)
    tB, hB = mode22(sim, lo)
    r = align_and_diff(tA, hA, tB, hB)
    if r is None:
        return None
    amp, _ = amp_phase(tA, hA)
    tpk = tA[np.argmax(amp)]
    pre = r["t"] < tpk - 50.0
    if pre.sum() < 50:
        return None
    return float(np.max(np.abs(r["dphi"][pre])))


def main():
    import sxs
    sims = sxs.load("simulations")

    cand = []
    for k in sims:
        if not k.startswith("SXS:BBH:"):
            continue
        m = sims[k]
        n = num(m.get("number_of_orbits"))
        levs = sorted(m.get("lev_numbers") or [])
        if n is None or n < 5 or len(levs) < 3:
            continue
        cand.append((k, n, levs[-3:]))

    # spread the sample over orbit count rather than taking the longest runs
    chosen = []
    for lo, hi in ORBIT_BINS:
        inbin = sorted([c for c in cand if lo <= c[1] < hi], key=lambda c: -c[1])
        chosen += inbin[:N_PER_BIN]
    print(f"{len(cand)} candidates; sampling {len(chosen)}\n")

    out = []
    print(f"{'simulation':18s}{'orbits':>8s}{'levs':>12s}"
          f"{'dphi(lo,mid)':>14s}{'dphi(mid,hi)':>14s}{'f':>8s}")
    print("-" * 74)
    for k, n, levs in chosen:
        lo, mid, hi = levs
        try:
            d1 = dphi_pair(k, lo, mid)
            d2 = dphi_pair(k, mid, hi)
        except Exception as ex:
            print(f"{k:18s}{n:8.1f}   [warn] {type(ex).__name__}: {ex}"[:100])
            continue
        if not d1 or not d2 or d2 <= 0:
            print(f"{k:18s}{n:8.1f}   [skip] unusable phase difference")
            continue
        f = d1 / d2
        out.append(dict(sim=k, orbits=n, levs=levs, dphi_lo_mid=d1,
                        dphi_mid_hi=d2, f=f))
        print(f"{k:18s}{n:8.1f}{str(levs):>12s}{d1:14.4g}{d2:14.4g}{f:8.2f}")

    if not out:
        print("\nno usable pairs")
        return
    fs = np.array([r["f"] for r in out])
    ns = np.array([r["orbits"] for r in out])
    print(f"\nf over {len(fs)} simulations: median {np.median(fs):.2f}, "
          f"range {fs.min():.2f}-{fs.max():.2f}")
    if len(fs) > 3:
        c = np.corrcoef(np.log(ns), np.log(fs))[0, 1]
        print(f"corr(ln orbits, ln f) = {c:+.2f}  "
              f"-> {'f degrades with length' if c < -0.3 else 'no strong length trend'}")
        long_ = fs[ns > 40]
        if len(long_):
            print(f"runs over 40 orbits (n={len(long_)}): median f = "
                  f"{np.median(long_):.2f}")
    json.dump(out, open(os.path.join(DATA, "sxs_convergence.json"), "w"),
              indent=1)
    print(f"\nwrote data/sxs_convergence.json ({len(out)} simulations)")


if __name__ == "__main__":
    main()
