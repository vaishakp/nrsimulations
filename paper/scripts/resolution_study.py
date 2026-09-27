#!/usr/bin/env python3
"""Is another resolution level worth running?

Measures the observed convergence factor from the one simulation that has three
resolutions (EccContPrecDiff001: Lev2, Lev3, Lev4), then asks what a further
level would buy against what it would cost, and against the phase accuracy that
next-generation detectors actually demand.

The accuracy target used is the standard indistinguishability criterion: two
waveforms are indistinguishable at signal-to-noise ratio rho when their
mismatch satisfies  MM < D / (2 rho^2)  with D the number of intrinsic
parameters; for a phase-dominated difference this corresponds roughly to
  delta_phi  <~  sqrt(D) / rho .
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import DATA
from make_waveform_figures import load_mode, align_and_diff, sim_path

HERE = os.path.dirname(os.path.abspath(__file__))


def phase_diff(name, lo, hi):
    a, b = sim_path(name, hi), sim_path(name, lo)
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    tA, hA = load_mode(a)
    tB, hB = load_mode(b)
    r = align_and_diff(tA, hA, tB, hB)
    if r is None:
        return None
    pre = r["t"] < r["tpk"] - 200
    return float(np.max(np.abs(r["dphi"][pre])))


def main():
    print("=" * 72)
    print("OBSERVED CONVERGENCE (the only 3-level simulation)")
    print("=" * 72)
    name = "EccContPrecDiff001"
    d23 = phase_diff(name, 2, 3)
    d34 = phase_diff(name, 3, 4)
    if d23 and d34:
        factor = d23 / d34
        print(f"  {name}:")
        print(f"    max |dphi| over inspiral, Lev2 vs Lev3 : {d23:8.3f} rad")
        print(f"    max |dphi| over inspiral, Lev3 vs Lev4 : {d34:8.3f} rad")
        print(f"    convergence factor per level           : {factor:8.2f}x")
    else:
        print("  [unavailable]")
        return

    # constraint-based cross-check
    con = json.load(open(os.path.join(DATA, "constraints.json")))
    rr = [v["ratio_lo_hi"] for v in con.values() if "ratio_lo_hi" in v]
    lens = [max(v[k]["t_end"] for k in v if k.isdigit())
            for v in con.values() if "ratio_lo_hi" in v]
    longr = [r for r, L in zip(rr, lens) if L > 10000]
    print(f"\n  cross-check: constraint improvement per level, long runs, "
          f"median {np.median(longr):.2f}x  (n={len(longr)})")

    print("\n" + "=" * 72)
    print("WHAT ANOTHER LEVEL WOULD BUY  --  EccPrecDiff002 (the record run)")
    print("=" * 72)
    conv = json.load(open(os.path.join(DATA, "convergence.json")))
    rec = next((c for c in conv if c["name"] == "EccPrecDiff002"), None)
    if rec is None:
        print("  [no convergence record]")
        return
    cur = rec["dphi_max_inspiral"]
    print(f"  current Lev2-Lev3 phase difference      : {cur:8.2f} rad")
    for n in (1, 2, 3):
        print(f"    after {n} more level(s) (x{factor:.1f} each)  : "
              f"{cur / factor ** n:8.3f} rad")

    print("\n" + "=" * 72)
    print("WHAT THE DETECTORS DEMAND")
    print("=" * 72)
    det = json.load(open(os.path.join(DATA, "detectability.json")))
    dd = det.get("EccPrecDiff002", {})
    D = 8.0   # intrinsic parameters for an eccentric, precessing binary
    for lab, rho in [("A+", dd.get("snr", {}).get("A+")),
                     ("ET", dd.get("snr", {}).get("ET")),
                     ("CE", dd.get("snr", {}).get("CE"))]:
        if not rho:
            continue
        tol = np.sqrt(D) / rho
        need = np.log(cur / tol) / np.log(factor)
        print(f"  {lab:3s}  SNR {rho:6.0f}   tolerable dphi ~ {tol:8.5f} rad"
              f"   -> needs {need:5.1f} more levels")

    print("\n" + "=" * 72)
    print("WHAT ANOTHER LEVEL WOULD COST")
    print("=" * 72)
    tim = json.load(open(os.path.join(DATA, "timings.json")))
    t = tim.get("per_sim", tim).get("EccPrecDiff002") if isinstance(
        tim.get("per_sim", tim), dict) else None
    print(f"  timings record: {json.dumps(t)[:300] if t else '[not found]'}")


if __name__ == "__main__":
    main()
