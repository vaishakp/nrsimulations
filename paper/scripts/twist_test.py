#!/usr/bin/env python3
"""Test of the co-precessing-frame ("twisting up") approximation.

The approximation asserts that a precessing waveform can be built by rotating a
*non-precessing* waveform of the same masses and aligned spin components through
the precession Euler angles.  We test it directly against NR:

  1. rotate the precessing NR waveform into its quadrupole-aligned frame,
     recording the rotor R(t);
  2. generate an aligned-spin model waveform with the same (q, chi_1z, chi_2z);
  3. compare it to the NR co-precessing modes (amplitude, phase), and
  4. twist it up with R(t) and compare the result to the full inertial-frame NR
     waveform.

Because every simulation in this catalogue is eccentric, the residual of step 3
mixes two distinct errors: the failure of the co-precessing approximation, and
the absence of eccentricity in the aligned-spin model.  Running the comparison
with both a quasi-circular and an eccentric aligned-spin model separates them.
"""
import os
import sys
import json
import numpy as np
import h5py
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coprecessing import (WF, DATA, sim_path, load_modes, to_waveform,
                          coprecessing, euler_angles, amp_phase)

SUR_PATH = os.path.expanduser("~/Downloads/NRHybSur3dq8.h5")
_SUR = None


def surrogate():
    global _SUR
    if _SUR is None:
        import gwsurrogate
        _SUR = gwsurrogate.LoadSurrogate(SUR_PATH)
    return _SUR


def ref_metadata(name, lev):
    p = os.path.join(WF, f"{name}_waveforms_Lev{lev}", "transformed",
                     "reference_metadata.json")
    return json.load(open(p)) if os.path.exists(p) else None


def start_frequency(t, h22, navg=400):
    """Initial (2,2) frequency f = omega_22 / 2pi in 1/M, averaged over the
    first `navg` samples after the junk-radiation window."""
    ph = np.unwrap(np.angle(h22))
    om = np.gradient(ph, t)
    return float(np.abs(np.median(om[:navg])) / (2.0 * np.pi))


def aligned_spin_model(q, chi1z, chi2z, f_low, dt=0.5, ellMax=4):
    """NRHybSur3dq8 modes, merger at t=0, in dimensionless units.

    `f_low` must be set from the NR start frequency: the surrogate is
    PN-hybridised and `f_low=0` asks for the full hybrid, which at NR sampling
    rates is tens of millions of samples and exhausts memory.
    """
    t, h, _ = surrogate()(
        q=q, chiA0=[0.0, 0.0, chi1z], chiB0=[0.0, 0.0, chi2z],
        dt=dt, f_low=f_low, ellMax=ellMax, units="dimensionless")
    return t, h


ECC_PY = os.path.expanduser("~/soft/anaconda/envs/pyseobnr/bin/python")
ECC_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "data", "model_cache")


def eccentric_model(q, chi1z, chi2z, ecc, f22_start, rel_anomaly=0.0,
                    dt=0.5, ell_max=4, approximant="SEOBNRv5EHM"):
    """SEOBNRv5EHM modes in dimensionless units, via the pyseobnr conda env.

    Returns (t/M with the (2,2) peak at 0, {(l,m): r h_lm / M}).  Results are
    cached on disk: each call is a subprocess launch plus an EOB integration.
    """
    import hashlib
    import subprocess

    os.makedirs(ECC_CACHE, exist_ok=True)
    key = hashlib.md5(repr((q, chi1z, chi2z, ecc, f22_start, rel_anomaly, dt,
                            ell_max, approximant)).encode()).hexdigest()[:16]
    path = os.path.join(ECC_CACHE, f"{approximant}_{key}.h5")

    if not os.path.exists(path):
        if not os.path.exists(ECC_PY):
            raise RuntimeError(f"pyseobnr interpreter not found at {ECC_PY}")
        # `--flag=value`, never `--flag value`: argparse's negative-number
        # detection does not recognise scientific notation, so a spin of
        # -4.7e-05 is parsed as an unknown option and the call dies in usage.
        cmd = [ECC_PY, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "gen_eccentric_model.py"),
               f"--out={path}", f"--q={q!r}", f"--chi1z={chi1z!r}",
               f"--chi2z={chi2z!r}", f"--ecc={ecc!r}",
               f"--rel-anomaly={rel_anomaly!r}",
               f"--f22-start={f22_start!r}", f"--dt={dt!r}",
               f"--ell-max={ell_max}", f"--approximant={approximant}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(path):
            raise RuntimeError(
                f"{approximant} generation failed: {r.stderr.strip()[-400:]}")

    with h5py.File(path, "r") as f:
        t = f["t"][()]
        modes = {}
        for k in f["modes"]:
            d = f["modes"][k]
            modes[(int(d.attrs["l"]), int(d.attrs["m"]))] = d[()][:, 0] + 1j * d[()][:, 1]
    return t, modes


def align_22(tA, hA, tB, hB, window=(0.05, 0.30)):
    """Solve for (dt, dphi) putting B onto A using the (2,2) phase.

    Returns a callable that resamples any mode of B onto A's time grid with the
    alignment applied, plus the offsets.  The phase offset is applied as
    exp(i m dphi) so that the whole mode set stays consistent.
    """
    ampA, phA = amp_phase(hA)
    ampB, phB = amp_phase(hB)
    tpkA, tpkB = tA[np.argmax(ampA)], tB[np.argmax(ampB)]
    dt0 = tpkA - tpkB
    t0 = max(tA[0], tB[0] + dt0) + 300.0
    t1 = tpkA - 100.0
    if not (t1 > t0):
        return None
    grid = np.linspace(t0, t1, 30000)
    w0, w1 = t0 + window[0] * (t1 - t0), t0 + window[1] * (t1 - t0)
    win = (grid >= w0) & (grid <= w1)
    fA = np.interp(grid, tA, phA)

    def resid(dt):
        fB = np.interp(grid, tB + dt0 + dt, phB)
        return float(np.var((fA - fB)[win]))

    r = minimize_scalar(resid, bounds=(-400.0, 400.0), method="bounded",
                        options={"xatol": 1e-4})
    dt = dt0 + r.x
    fB = np.interp(grid, tB + dt, phB)
    dphi = float(np.mean((fA - fB)[win])) / 2.0   # per unit m
    return dict(dt=dt, dphi=dphi, t0=t0, t1=t1)


def resample_modes(h, tB, t_target, al, ell_max=4):
    """Resample model modes onto t_target with the (dt, dphi) alignment."""
    out = {}
    for (l, m), y in h.items():
        if l > ell_max:
            continue
        re = np.interp(t_target, tB + al["dt"], y.real, left=0.0, right=0.0)
        im = np.interp(t_target, tB + al["dt"], y.imag, left=0.0, right=0.0)
        out[(l, m)] = (re + 1j * im) * np.exp(1j * m * al["dphi"])
    return out


def flat_mismatch(t, a, b, mask=None):
    """Time-domain mismatch with a flat noise weighting, maximised over an
    overall phase (not over time -- the waveforms are already time-aligned)."""
    if mask is not None:
        t, a, b = t[mask], a[mask], b[mask]
    ip = np.trapezoid(a * np.conj(b), t)
    na = np.sqrt(np.trapezoid(np.abs(a) ** 2, t).real)
    nb = np.sqrt(np.trapezoid(np.abs(b) ** 2, t).real)
    return 1.0 - np.abs(ip) / (na * nb)
