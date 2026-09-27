#!/usr/bin/env python3
"""Shared machinery for the co-precessing-frame study.

Loads a transformed NR waveform into a `scri.WaveformModes`, rotates it to the
quadrupole-aligned (co-precessing) frame, and provides the diagnostics used to
ask how closely the co-precessing-frame waveform resembles a non-precessing one.
"""
import os
import numpy as np
import h5py
import quaternion
import scri

HERE = os.path.dirname(os.path.abspath(__file__))
WF = os.path.join(HERE, "..", "waveforms")
DATA = os.path.join(HERE, "..", "data")


def sim_path(name, lev, kind="transformed", n=2):
    d = os.path.join(WF, f"{name}_waveforms_Lev{lev}")
    if kind == "extrapolated":
        return os.path.join(d, "extrapolated", f"rhOverM_Extrapolated_N{n}_CoM.h5")
    return os.path.join(d, "transformed",
                        f"rhOverM_extrapolated_CoM_transformed_N{n}.h5")


def load_modes(path, ell_max=8):
    """Read a SpEC-style mode file into (t, dict[(l,m)] -> complex array)."""
    out, t = {}, None
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if not k.startswith("Y_l"):
                continue
            b = k[:-4] if k.endswith(".dat") else k
            lp, mp = b.split("_")[1:3]
            l, m = int(lp[1:]), int(mp[1:])
            if l > ell_max:
                continue
            d = f[k][()]
            if t is None:
                t = d[:, 0].copy()
            out[(l, m)] = d[:, 1] + 1j * d[:, 2]
    return t, out


def to_waveform(t, modes, ell_min=2, ell_max=8, decimate=1):
    """Pack a mode dict into a scri.WaveformModes in the inertial frame."""
    lms = [(l, m) for l in range(ell_min, ell_max + 1)
           for m in range(-l, l + 1)]
    missing = [lm for lm in lms if lm not in modes]
    if missing:
        raise KeyError(f"missing modes: {missing[:5]}")
    d = np.column_stack([modes[lm] for lm in lms])
    if decimate > 1:
        t, d = t[::decimate].copy(), d[::decimate].copy()
    return scri.WaveformModes(
        t=np.ascontiguousarray(t),
        data=np.ascontiguousarray(d),
        ell_min=ell_min, ell_max=ell_max,
        frameType=scri.Inertial, dataType=scri.h,
        r_is_scaled_out=True, m_is_scaled_out=True,
    )


def coprecessing(W, transition=True):
    """Return (co-precessing copy, rotor array R(t)).

    The frame is the one in which the dominant eigenvector of the
    <L^(a) L^b)> matrix points along +z, with the remaining rotational freedom
    fixed by minimal rotation (Boyle 2013).  h^CP = D(R)^-1 h.
    """
    Wc = W.copy()
    kw = {}
    if transition:
        tpk = W.t[np.argmax(np.linalg.norm(W.data, axis=1))]
        kw["transition_times"] = (tpk, tpk + 50.0)
    Wc.to_coprecessing_frame(**kw)
    return Wc, Wc.frame.copy()


def euler_angles(R):
    """(alpha, beta, gamma) of the rotor array, unwrapped in alpha and gamma."""
    a, b, g = quaternion.as_euler_angles(R).T
    return np.unwrap(a), b, np.unwrap(g)


def _idx(W, l, m):
    return W.index(l, m)


def mode_power(W, l):
    i0 = W.index(l, -l)
    return np.sum(np.abs(W.data[:, i0:i0 + 2 * l + 1]) ** 2, axis=1)


def quadrupole_concentration(W):
    """Fraction of the ell=2 power carried by the m = +-2 modes."""
    p22 = (np.abs(W.data[:, W.index(2, 2)]) ** 2
           + np.abs(W.data[:, W.index(2, -2)]) ** 2)
    return p22 / mode_power(W, 2)


def mirror_asymmetry(W, ell_max=4):
    """Relative violation of h_{l,-m} = (-1)^l conj(h_{lm}).

    Exact for a non-precessing (equatorially symmetric) binary in its natural
    frame; in the co-precessing frame of a precessing binary it is a direct,
    model-independent measure of how far the waveform is from a non-precessing
    one.
    """
    num = np.zeros(W.n_times)
    den = np.zeros(W.n_times)
    for l in range(2, min(ell_max, W.ell_max) + 1):
        for m in range(-l, l + 1):
            a = W.data[:, W.index(l, m)]
            b = W.data[:, W.index(l, -m)]
            num += np.abs(b - (-1) ** l * np.conj(a)) ** 2
            den += np.abs(a) ** 2
    return np.sqrt(num / np.maximum(den, 1e-300))


def amp_phase(h):
    return np.abs(h), np.unwrap(np.angle(h))
