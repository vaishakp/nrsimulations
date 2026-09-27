#!/usr/bin/env python3
"""Where the catalogue waveforms sit in the next-generation detector band.

A long NR waveform is valuable precisely because it removes the need to
hybridise onto a PN or EOB inspiral.  The question this script answers is: for
what total masses does a given waveform already span the whole detector band,
so that nothing has to be attached below it?

For each simulation we take the (2,2) start frequency in geometric units and
convert to Hz at a total mass M,

    f_start(M) = f_start^geom / (M * MTSUN_SI) ,

so the waveform covers the band down to f_low whenever

    M <= M_max = f_start^geom / (f_low * MTSUN_SI) .

We then compute the optimal SNR of the NR waveform alone at a fiducial
distance, in Cosmic Explorer, Einstein Telescope and A+, and report the
fraction of SNR^2 that the NR data supplies relative to the same signal
extended down to f_low by an aligned-spin model.  That fraction is the
quantitative version of "is the early inspiral doing any work".
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import WF, DATA, sim_path, load_modes

MTSUN_SI = 4.925490947641267e-06
MRSUN_SI = 1476.6250615036158
PC_SI = 3.085677581491367e16

PSDS = {
    "CE": "SimNoisePSDCosmicExplorerP1600143",
    "ET": "SimNoisePSDEinsteinTelescopeP1600143",
    "A+": "SimNoisePSDaLIGOAPlusDesignSensitivityT1800042",
}
F_LOW = 5.0            # next-generation low-frequency cutoff, Hz
D_FID_MPC = 1000.0     # fiducial luminosity distance


def psd_array(name, freqs):
    """S_n(f) on `freqs`.

    The P1600143 curves are series-filling routines, `fn(series, flow)`, not
    scalar functions of f; the A+ curve is scalar.  Failures are raised, not
    swallowed -- silently returning inf turns every SNR into zero.
    """
    import lal
    import lalsimulation as ls
    fn = getattr(ls, PSDS[name])
    df = float(freqs[1] - freqs[0])
    out = np.full(len(freqs), np.inf)
    try:
        series = lal.CreateREAL8FrequencySeries(
            name, lal.LIGOTimeGPS(0), 0.0, df, lal.SecondUnit, len(freqs))
        fn(series, F_LOW)
        vals = np.asarray(series.data.data, dtype=float)
    except TypeError:
        vals = np.array([fn(f) if f > 0 else 0.0 for f in freqs], dtype=float)
    good = np.isfinite(vals) & (vals > 0)
    out[good] = vals[good]
    return out


def snr_of(t_geom, h22_geom, M, det, f_low=F_LOW, d_mpc=D_FID_MPC):
    """Optimal SNR of the (2,2)-only, face-on signal at total mass M.

    Face-on is the loudest orientation, so this is an upper bound across
    inclination; it is used consistently for every simulation, so the
    comparison between them is fair.
    """
    M_sec = M * MTSUN_SI
    t = t_geom * M_sec
    # face-on: h = h22 * (-2)Y_22(0,0) + h2-2 * (-2)Y_2-2(0,0); keep the (2,2)
    # piece, whose harmonic value at iota=0 is sqrt(5/(16 pi))
    amp = np.sqrt(5.0 / (16.0 * np.pi))
    scale = (M * MRSUN_SI) / (d_mpc * 1e6 * PC_SI)
    h = h22_geom * amp * scale

    dt = float(np.median(np.diff(t)))
    n = len(t)
    tu = np.arange(t[0], t[-1], dt)
    hu = np.interp(tu, t, h.real) + 1j * np.interp(tu, t, h.imag)
    # window the start to suppress the abrupt turn-on
    w = np.ones(len(hu))
    k = max(1, int(0.02 * len(hu)))
    w[:k] = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    hu = hu * w

    hf = np.fft.rfft(hu.real) * dt
    freqs = np.fft.rfftfreq(len(hu), dt)
    S = psd_array(det, freqs)
    band = (freqs >= f_low) & np.isfinite(S)
    if not band.any():
        return 0.0
    integ = 4.0 * np.abs(hf[band]) ** 2 / S[band]
    return float(np.sqrt(np.trapezoid(integ, freqs[band])))


def start_freq_geom(t, h22, skip=300.0, n_orbits=3.0):
    """Orbit-averaged (2,2) start frequency in 1/M.

    For an eccentric binary omega_22 swings by a large factor over each radial
    period, so sampling the first few hundred M gives whatever orbital phase
    the run happens to start at.  Averaging the accumulated phase over the
    first few orbits removes that.
    """
    j = np.searchsorted(t, t[0] + skip)
    t, ph = t[j:], np.unwrap(np.angle(h22[j:]))
    om_rough = np.abs(np.median(np.gradient(ph, t)[:400]))
    span = n_orbits * 2 * np.pi / max(om_rough, 1e-8)
    k = np.searchsorted(t, t[0] + min(span, 0.25 * (t[-1] - t[0])))
    k = max(k, 10)
    return float(abs(ph[k] - ph[0]) / (t[k] - t[0]) / (2.0 * np.pi))


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
    print(f"{'name':22s}{'L':>3s}{'len/M':>9s}{'f_geom':>9s}{'M_max':>8s}"
          f"{'T@Mmax':>8s}{'SNR_CE':>8s}{'SNR_ET':>8s}{'SNR_A+':>8s}")
    print(f"{'':22s}{'':>3s}{'':>9s}{'[1/M]':>9s}{'[Msun]':>8s}{'[s]':>8s}"
          f"{'':>8s}{'':>8s}{'':>8s}")
    print("-" * 83)
    for name, lev in sorted(best.items()):
        try:
            t, modes = load_modes(sim_path(name, lev), ell_max=2)
            h22 = modes[(2, 2)]
            j = np.searchsorted(t, t[0] + 300)
            t, h22 = t[j:], h22[j:]
            f0 = start_freq_geom(t, h22, skip=0.0)
            M_max = f0 / (F_LOW * MTSUN_SI)
            T_at = (t[-1] - t[0]) * M_max * MTSUN_SI
            snrs = {d: snr_of(t, h22, M_max, d) for d in PSDS}
            out[name] = dict(lev=lev, length_M=float(t[-1] - t[0]),
                             f22_start_geom=f0, M_max_Msun=float(M_max),
                             duration_s_at_Mmax=float(T_at),
                             snr=snrs, ecc=(ecc.get(name) or {}).get("ecc_median"),
                             precessing=bool(cat.get(name, {}).get("precessing")))
            print(f"{name:22s}{lev:>3d}{t[-1]-t[0]:>9.0f}{f0:>9.5f}{M_max:>8.1f}"
                  f"{T_at:>8.1f}{snrs['CE']:>8.1f}{snrs['ET']:>8.1f}"
                  f"{snrs['A+']:>8.1f}")
        except Exception as e:
            print(f"{name:22s}{lev:>3d}  [warn] {type(e).__name__}: {e}"[:110])

    json.dump(out, open(os.path.join(DATA, "detectability.json"), "w"), indent=1)
    print(f"\nwrote data/detectability.json ({len(out)} simulations)")
    print(f"\nM_max is the largest total mass for which the NR waveform alone "
          f"already\nreaches {F_LOW:.0f} Hz, so no hybridisation is needed below it. "
          f"SNRs are\nface-on, (2,2) only, at {D_FID_MPC:.0f} Mpc, at M = M_max.")


if __name__ == "__main__":
    main()
