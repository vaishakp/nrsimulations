#!/usr/bin/env python3
"""Co-precessing-frame approximation: NR against aligned-spin models.

For each precessing simulation we compare the NR co-precessing-frame (2,2) mode
against an aligned-spin waveform with the same (q, chi_1z, chi_2z), generated

  * quasi-circular  -- NRHybSur3dq8 and SEOBNRv5HM
  * eccentric       -- SEOBNRv5EHM at the *measured* eccentricity

The difference between the two tells us how much of the residual is the
co-precessing approximation itself and how much is simply that the comparison
waveform was not eccentric.  The eccentric model's relative anomaly is not
known a priori, so it is scanned and the best match kept -- it plays the same
role as the time and phase offsets.
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import (WF, DATA, sim_path, load_modes, to_waveform,
                          coprecessing, amp_phase)
from twist_test import (ref_metadata, start_frequency, aligned_spin_model,
                        eccentric_model, align_22, resample_modes,
                        flat_mismatch)

N_ANOMALY = 8


def nr_coprecessing(name, lev, dt_target=0.5):
    t, modes = load_modes(sim_path(name, lev), ell_max=4)
    dec = max(1, int(round(dt_target / np.median(np.diff(t)))))
    W = to_waveform(t, modes, ell_max=4, decimate=dec)
    Wc, R = coprecessing(W)
    tpk = W.t[np.argmax(np.abs(W.data[:, W.index(2, 2)]))]
    Wc.t = Wc.t - tpk
    W.t = W.t - tpk
    return W, Wc, R


def compare(tC, hC, tM, hM_modes, label):
    """Align a model mode set to the NR co-precessing (2,2) and score it."""
    al = align_22(tC, hC, tM, hM_modes[(2, 2)])
    if al is None:
        return None
    hm = resample_modes(hM_modes, tM, tC, al)
    sel = (tC > tC[0] + 300) & (tC < -100) & (np.abs(hm[(2, 2)]) > 0)
    if sel.sum() < 200:
        return None
    aC, pC = amp_phase(hC)
    aS, pS = amp_phase(hm[(2, 2)])
    dphi = pC - pS
    dphi = dphi - np.median(dphi[sel][:max(1, sel.sum() // 10)])
    dA = (aC - aS) / np.maximum(aS, 1e-30)
    return dict(
        label=label,
        dphi_med=float(np.median(np.abs(dphi[sel]))),
        dphi_max=float(np.max(np.abs(dphi[sel]))),
        dA_med=float(np.median(np.abs(dA[sel]))),
        mismatch=float(flat_mismatch(tC, hC, hm[(2, 2)], sel)),
        dt=float(al["dt"]), n=int(sel.sum()),
    )


def run_one(name, lev, cat, ecc):
    md = ref_metadata(name, lev)
    if md is None:
        return None
    q = cat[name]["q"]
    c1z, c2z = md["chiA_ref"][2], md["chiB_ref"][2]
    W, Wc, R = nr_coprecessing(name, lev)
    hC = Wc.data[:, Wc.index(2, 2)]
    j = np.searchsorted(Wc.t, Wc.t[0] + 300)
    f0 = start_frequency(Wc.t[j:], hC[j:])
    e_meas = (ecc.get(name) or {}).get("ecc_median")
    e_spread = (ecc.get(name) or {}).get("ecc_spread")

    out = dict(name=name, lev=lev, q=q, chi1z=c1z, chi2z=c2z, f22_start=f0,
               ecc=e_meas, ecc_spread=e_spread, results={})

    # quasi-circular arms
    if q <= 8.0 and max(abs(c1z), abs(c2z)) <= 0.8:
        try:
            tM, hM = aligned_spin_model(q, c1z, c2z, f_low=f0 * 0.95)
            r = compare(Wc.t, hC, tM, hM, "NRHybSur3dq8 (e=0)")
            if r:
                out["results"]["nrhybsur_qc"] = r
        except Exception as e:
            out["results"]["nrhybsur_qc"] = {"error": str(e)[:150]}
    try:
        tM, hM = eccentric_model(q, c1z, c2z, 0.0, f0 * 0.95,
                                 approximant="SEOBNRv5HM")
        r = compare(Wc.t, hC, tM, hM, "SEOBNRv5HM (e=0)")
        if r:
            out["results"]["seob_qc"] = r
    except Exception as e:
        out["results"]["seob_qc"] = {"error": str(e)[:150]}

    # eccentric arm, scanning the relative anomaly
    if e_meas is not None and e_meas > 0.01:
        best = None
        for ell in np.linspace(0.0, 2 * np.pi, N_ANOMALY, endpoint=False):
            try:
                tM, hM = eccentric_model(q, c1z, c2z, e_meas, f0 * 0.95,
                                         rel_anomaly=float(ell))
                r = compare(Wc.t, hC, tM, hM, "SEOBNRv5EHM")
                if r and (best is None or r["mismatch"] < best["mismatch"]):
                    r["rel_anomaly"] = float(ell)
                    best = r
            except Exception as e:
                out.setdefault("ecc_errors", []).append(str(e)[:120])
        if best:
            out["results"]["seob_ecc"] = best
    return out


def main():
    cat = json.load(open(os.path.join(DATA, "catalog.json")))
    ecc = json.load(open(os.path.join(DATA, "eccentricity.json")))
    best = {}
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
        if os.path.exists(sim_path(name, int(lev))):
            best[name] = max(best.get(name, 0), int(lev))

    targets = [(n, l) for n, l in sorted(best.items())
               if cat.get(n, {}).get("precessing")]
    print(f"{len(targets)} precessing simulations\n")

    out = {}
    for name, lev in targets:
        try:
            r = run_one(name, lev, cat, ecc)
        except Exception as e:
            print(f"{name:22s} FAILED {type(e).__name__}: {e}"[:130])
            continue
        if r is None:
            continue
        out[name] = r
        e = r["ecc"]
        print(f"{name:22s} L{lev} q={r['q']:<4} e={(f'{e:.3f}' if e else '--'):>6s}")
        for k, v in r["results"].items():
            if "error" in v:
                print(f"    {k:14s} ERROR {v['error'][:80]}")
            else:
                print(f"    {v['label']:22s} max|dphi|={v['dphi_max']:8.3f} rad"
                      f"  |dA|={v['dA_med']:7.4f}  mismatch={v['mismatch']:.3e}")
        print()

    json.dump(out, open(os.path.join(DATA, "twist_study.json"), "w"), indent=1)
    print(f"wrote data/twist_study.json ({len(out)} simulations)")


if __name__ == "__main__":
    main()
