#!/usr/bin/env python3
"""Count orbits and GW cycles directly from the (2,2) mode of each waveform.

The catalogue tables on the project website report "Ncycles", which is the number
of *gravitational-wave* cycles of the (2,2) mode, phi_22 / 2pi.  Since the
dominant quadrupole has phi_22 = 2 phi_orb, the number of *orbits* is half that.
This script measures both directly so the paper quotes them unambiguously.
"""
import glob
import json
import os
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
WF = os.path.join(HERE, "..", "waveforms")


def load22(path):
    with h5py.File(path, "r") as f:
        k = "Y_l2_m2.dat" if "Y_l2_m2.dat" in f else "Y_l2_m2"
        d = f[k][()]
    return d[:, 0], d[:, 1] + 1j * d[:, 2]


out = {}
for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
    p = os.path.join(d, "extrapolated", "rhOverM_Extrapolated_N2_CoM.h5")
    if not os.path.exists(p):
        continue
    name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
    t, h = load22(p)
    a = np.abs(h)
    ph = np.unwrap(np.angle(h))
    jpk = int(np.argmax(a))
    # start after the junk-radiation transient
    j0 = int(np.searchsorted(t, t[0] + 300.0))
    dphi22 = abs(ph[jpk] - ph[j0])
    e = out.setdefault(name, {})
    e[f"Lev{lev}"] = {
        "gw_cycles": dphi22 / (2 * np.pi),
        "orbits": dphi22 / (4 * np.pi),
        "phi22_rad": dphi22,
        "t_start": float(t[j0]),
        "t_peak": float(t[jpk]),
        "t_span_M": float(t[jpk] - t[j0]),
    }

# best (highest level) per simulation
summary = {}
for name, levs in out.items():
    best = sorted(levs)[-1]
    summary[name] = dict(levs[best], lev=best)

json.dump({"per_level": out, "summary": summary},
          open(os.path.join(DATA, "orbit_counts.json"), "w"), indent=1)

cat = json.load(open(os.path.join(DATA, "catalog.json")))
print(f"{'simulation':<22}{'lev':>4}{'orbits':>9}{'GWcyc':>9}{'t_span/M':>10}"
      f"{'table Ncyc':>12}{'ratio':>7}")
print("-" * 73)
for n in sorted(summary):
    s = summary[n]
    tb = cat.get(n, {}).get("ncycles_max")
    rat = (tb / s["gw_cycles"]) if tb else float("nan")
    print(f"{n:<22}{s['lev']:>4}{s['orbits']:>9.1f}{s['gw_cycles']:>9.1f}"
          f"{s['t_span_M']:>10.0f}{(tb if tb else 0):>12.1f}{rat:>7.2f}")

top = sorted(summary.items(), key=lambda kv: -kv[1]["orbits"])[:5]
print("\nlongest by orbit count:")
for n, s in top:
    e = cat.get(n, {})
    print(f"  {n:<20} {s['orbits']:6.1f} orbits ({s['gw_cycles']:.1f} GW cycles), "
          f"q={e.get('q')}, e0={e.get('ecc')}, chi_p={e.get('chi_p')}, "
          f"prec={e.get('precessing')}")
print(f"\nwrote {os.path.join(DATA, 'orbit_counts.json')}")
