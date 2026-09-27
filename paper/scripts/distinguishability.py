#!/usr/bin/env python3
"""Are the resolution levels distinguishable to a detector?

The phase-difference figures of Sec. VI say how much two resolutions differ.
What matters observationally is whether that difference is *detectable*, which
is the Lindblom-Owen-Brown criterion: two waveforms are indistinguishable when

    <dh|dh>  <  D ,        dh = h_1 - h_2 ,

with D the number of intrinsic parameters (Baird et al., arXiv:1211.0546, use
the chi^2_D threshold; D itself is the common convention).  In terms of the
mismatch MM between normalised waveforms and the signal-to-noise ratio rho,

    MM  <  D / (2 rho^2)   <=>   rho  <  rho_crit = sqrt( D / (2 MM) ) .

`rho_crit` is the useful number: above that signal-to-noise ratio the
catalogue's own numerical error becomes measurable, so it is the point beyond
which these waveforms cannot be used as ground truth without qualification.

The mismatch is noise-weighted and maximised over relative time and phase
shifts, at the total mass for which the waveform just spans the detector band
(so the whole NR waveform is used and no hybridisation is implied).
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import DATA, WF
from detectability import psd_array, MTSUN_SI, MRSUN_SI, PC_SI, F_LOW
from make_waveform_figures import load_mode, sim_path

D_PARAMS = 8.0     # eccentric + precessing: m1,m2,6 spin... plus e, l0
DETS = ["CE", "ET", "A+"]


def taper(y, frac=0.02):
    n = len(y)
    k = max(1, int(frac * n))
    w = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    w[:k] = ramp
    w[-k:] = ramp[::-1]
    return y * w


def mismatch(t1, h1, t2, h2, M, det, f_low=F_LOW):
    """Noise-weighted mismatch, maximised over time and phase shift."""
    M_sec = M * MTSUN_SI
    dt = min(np.median(np.diff(t1)), np.median(np.diff(t2))) * M_sec
    t0 = max(t1[0], t2[0]) * M_sec
    t1e = min(t1[-1], t2[-1]) * M_sec
    grid = np.arange(t0, t1e, dt)
    if len(grid) < 1024:
        return None
    a = np.interp(grid, t1 * M_sec, h1.real) + 1j * np.interp(
        grid, t1 * M_sec, h1.imag)
    b = np.interp(grid, t2 * M_sec, h2.real) + 1j * np.interp(
        grid, t2 * M_sec, h2.imag)
    a, b = taper(a), taper(b)

    n = 1 << int(np.ceil(np.log2(len(grid) * 2)))
    A = np.fft.rfft(a.real, n) * dt
    B = np.fft.rfft(b.real, n) * dt
    f = np.fft.rfftfreq(n, dt)
    S = psd_array(det, f)
    band = (f >= f_low) & np.isfinite(S) & (f > 0)
    if band.sum() < 64:
        return None

    w = np.zeros_like(S)
    w[band] = 4.0 / S[band]
    na = np.sqrt(np.sum(w * np.abs(A) ** 2) * (f[1] - f[0]))
    nb = np.sqrt(np.sum(w * np.abs(B) ** 2) * (f[1] - f[0]))
    if na <= 0 or nb <= 0:
        return None
    # maximise over time shift by inverse-transforming the weighted correlation,
    # and over phase by taking the modulus
    integrand = np.zeros(len(f), dtype=complex)
    integrand[band] = (A[band] * np.conj(B[band]) * w[band])
    corr = np.fft.irfft(integrand, n) * len(f) * (f[1] - f[0])
    return float(1.0 - np.max(np.abs(corr)) / (na * nb))


def main():
    det_info = json.load(open(os.path.join(DATA, "detectability.json")))
    cat = json.load(open(os.path.join(DATA, "catalog.json")))

    pairs = {}
    for d in sorted(glob.glob(os.path.join(WF, "*_waveforms_Lev*"))):
        name, lev = os.path.basename(d).rsplit("_waveforms_Lev", 1)
        if os.path.exists(sim_path(name, int(lev))):
            pairs.setdefault(name, []).append(int(lev))
    pairs = {k: sorted(v) for k, v in pairs.items() if len(v) >= 2}

    out = {}
    print(f"{'sim':22s}{'levels':>8s}{'M[Msun]':>9s}" +
          "".join(f"{('MM_'+x):>11s}" for x in DETS) +
          "".join(f"{('rho*_'+x):>10s}" for x in DETS))
    print("-" * 100)
    for name, levs in sorted(pairs.items()):
        lo, hi = levs[-2], levs[-1]
        info = det_info.get(name)
        if info is None:
            continue
        M = info["M_max_Msun"]
        try:
            t1, h1 = load_mode(sim_path(name, hi))
            t2, h2 = load_mode(sim_path(name, lo))
        except Exception:
            continue
        row, rho = {}, {}
        for dd in DETS:
            mm = mismatch(t1, h1, t2, h2, M, dd)
            if mm is None or mm <= 0:
                continue
            row[dd] = mm
            rho[dd] = float(np.sqrt(D_PARAMS / (2.0 * mm)))
        if not row:
            continue
        out[name] = dict(lo=lo, hi=hi, M=M, mismatch=row, rho_crit=rho,
                         snr=info["snr"], precessing=info["precessing"])
        print(f"{name:22s}{f'{lo}-{hi}':>8s}{M:>9.0f}" +
              "".join(f"{row.get(x, float('nan')):>11.2e}" for x in DETS) +
              "".join(f"{rho.get(x, float('nan')):>10.0f}" for x in DETS))

    json.dump(out, open(os.path.join(DATA, "distinguishability.json"), "w"),
              indent=1)
    print(f"\nwrote data/distinguishability.json ({len(out)} simulations)")

    # summary: is the actual SNR above or below rho_crit?
    print("\n" + "=" * 72)
    print("IS THE NUMERICAL ERROR DETECTABLE AT THE ACTUAL SNR?  (1 Gpc, face-on)")
    print("=" * 72)
    for dd in DETS:
        rc = np.array([v["rho_crit"][dd] for v in out.values() if dd in v["rho_crit"]])
        sn = np.array([v["snr"][dd] for v in out.values() if dd in v["rho_crit"]])
        bad = sn > rc
        print(f"  {dd:3s}: median rho_crit = {np.median(rc):7.0f} | "
              f"median actual SNR = {np.median(sn):7.0f} | "
              f"distinguishable for {bad.sum()}/{len(rc)} simulations")


if __name__ == "__main__":
    main()
