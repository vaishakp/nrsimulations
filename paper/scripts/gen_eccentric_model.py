#!/usr/bin/env python3
"""Generate an eccentric aligned-spin model waveform in dimensionless units.

`pyseobnr` (SEOBNRv5EHM) cannot be built against this machine's SWIG 4.5.1 --
its `pygsl-lite` dependency emits Python-2 C API calls -- so it lives in a
separate conda environment:

    ~/soft/anaconda/envs/pyseobnr/bin/python

This script is meant to be run *by that interpreter*, writing modes to an HDF5
file that the main environment reads back.  `eccentric_model()` in
`twist_test.py` is the caller.

SEOBNRv5EHM works in physical units, so we fix a fiducial total mass and undo
the scaling:  t/M = t_sec / (M M_sun^sec),  r h/M = h D / (M M_sun^metre).
The fiducial mass cancels; it only sets the numerical range.
"""
import argparse
import json
import numpy as np
import h5py

MTSUN_SI = 4.925490947641267e-06     # GMsun/c^3 in seconds
MRSUN_SI = 1476.6250615036158        # GMsun/c^2 in metres
PC_SI = 3.085677581491367e16

M_FID = 50.0          # fiducial total mass, solar masses
D_FID_MPC = 500.0     # fiducial distance


def generate(q, chi1z, chi2z, ecc, f22_start_geom, rel_anomaly=0.0,
             dt_geom=0.5, ell_max=4, approximant="SEOBNRv5EHM"):
    """Return (t/M, {(l,m): r h_lm / M}) with the peak of |h22| at t = 0."""
    from pyseobnr.generate_waveform import GenerateWaveform

    M_sec = M_FID * MTSUN_SI
    m1 = M_FID * q / (1.0 + q)
    m2 = M_FID / (1.0 + q)

    params = dict(
        mass1=m1, mass2=m2,
        spin1x=0.0, spin1y=0.0, spin1z=chi1z,
        spin2x=0.0, spin2y=0.0, spin2z=chi2z,
        f22_start=f22_start_geom / M_sec,
        deltaT=dt_geom * M_sec,
        distance=D_FID_MPC,
        inclination=0.0,
        approximant=approximant,
    )
    if ecc > 0:
        params["eccentricity"] = ecc
        params["rel_anomaly"] = rel_anomaly

    wf = GenerateWaveform(params)
    t_sec, hlm = wf.generate_td_modes()

    # physical strain -> r h / M
    scale = (D_FID_MPC * 1e6 * PC_SI) / (M_FID * MRSUN_SI)
    t = np.asarray(t_sec) / M_sec
    out = {lm: np.asarray(h) * scale for lm, h in hlm.items()
           if lm[0] <= ell_max}
    if (2, 2) in out:
        t = t - t[np.argmax(np.abs(out[(2, 2)]))]
    return t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--chi1z", type=float, default=0.0)
    ap.add_argument("--chi2z", type=float, default=0.0)
    ap.add_argument("--ecc", type=float, default=0.0)
    ap.add_argument("--rel-anomaly", type=float, default=0.0)
    ap.add_argument("--f22-start", type=float, required=True,
                    help="(2,2) start frequency in 1/M (geometric)")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--ell-max", type=int, default=4)
    ap.add_argument("--approximant", default="SEOBNRv5EHM")
    a = ap.parse_args()

    t, modes = generate(a.q, a.chi1z, a.chi2z, a.ecc, a.f22_start,
                        a.rel_anomaly, a.dt, a.ell_max, a.approximant)
    with h5py.File(a.out, "w") as f:
        f.create_dataset("t", data=t)
        g = f.create_group("modes")
        for (l, m), h in modes.items():
            d = g.create_dataset(f"Y_l{l}_m{m}", data=np.column_stack(
                [h.real, h.imag]))
            d.attrs["l"], d.attrs["m"] = l, m
        f.attrs["params"] = json.dumps(vars(a))
    print(f"wrote {a.out}: {len(t)} samples, t in [{t[0]:.0f}, {t[-1]:.0f}] M, "
          f"{len(modes)} modes")


if __name__ == "__main__":
    main()
