#!/usr/bin/env python3
"""Harvest the normalised GH constraint record from the SpEC run tree.

Runs ON THE CLUSTER (sonic), where the run tree lives:

    scp paper/scripts/harvest_constraints.py vaishak.p@sonic:/tmp/
    ssh vaishak.p@sonic 'cd /tmp && python3 harvest_constraints.py > harvest.json'

The system python there is 3.6.8 with the standard library only, so this file
must not import numpy or anything else external.

Reads, per evolution segment,

    Run/ConstraintNorms/NormalizedGhCe_Norms.dat
    columns: [1] t   [2] VolLp(NormalizedGhCe)   [3] L2   [4] Linf

and joins the segments of one (simulation, eccentricity-reduction dir,
resolution) into a single time series, dropping checkpoint-restart overlap.

TWO SEGMENT LAYOUTS EXIST and both must be walked.  The inspiral segments sit
directly under the evolution directory,

    Ev/Lev3_AA/Run/ConstraintNorms/...

but the ringdown segments are nested one level deeper, inside a per-level
ringdown directory:

    Ev/Lev3_Ringdown/Lev3_AA/Run/ConstraintNorms/...

An earlier version of this harvest matched only the first layout and therefore
returned no post-merger data at all, which was misread as the ringdown not
having been evolved.  It had been: every ringdown segment in the catalogue
carries a constraint record.
"""
from __future__ import print_function

import json
import os
import re
import sys

ROOT = "/mnt/pfs/vaishak.p/sims/SpEC/gcc/bfi"
DAT = os.path.join("Run", "ConstraintNorms", "NormalizedGhCe_Norms.dat")
SEG = re.compile(r"^Lev(\d+)_([A-Z]{2})$")
RD = re.compile(r"^Lev(\d+)_Ringdown$")
TARGET_POINTS = 2000


def read_dat(path):
    """[(t, L2, Linf)] from a SpEC .dat file, skipping '#' headers."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                p = line.split()
                if len(p) < 4:
                    continue
                try:
                    out.append((float(p[0]), float(p[2]), float(p[3])))
                except ValueError:
                    continue
    except IOError:
        return []
    return out


def segments(ev):
    """[(lev, seg_name, dat_path, is_ringdown)] for one Ev directory.

    Walks both the inspiral layout and the nested ringdown layout.  `.bak`
    directories are stray copies and are excluded.
    """
    found = []
    try:
        entries = sorted(os.listdir(ev))
    except OSError:
        return found
    for e in entries:
        if e.endswith(".bak"):
            continue
        d = os.path.join(ev, e)
        if not os.path.isdir(d):
            continue
        m = SEG.match(e)
        if m:
            p = os.path.join(d, DAT)
            if os.path.exists(p):
                found.append((int(m.group(1)), e, p, False))
            continue
        m = RD.match(e)
        if m:
            lev = int(m.group(1))
            try:
                inner = sorted(os.listdir(d))
            except OSError:
                continue
            for s in inner:
                if s.endswith(".bak"):
                    continue
                sm = SEG.match(s)
                if not sm:
                    continue
                p = os.path.join(d, s, DAT)
                if os.path.exists(p):
                    found.append((lev, e + "/" + s, p, True))
    return found


def join(rows_per_seg):
    """Concatenate segments in time order, dropping restart overlap.

    Segments are ordered by their first timestamp; a restart re-runs from the
    last checkpoint, so any row at or before the last time already kept is a
    duplicate of evolution we have.
    """
    rows_per_seg.sort(key=lambda r: r[0][0] if r[0] else 0.0)
    t_last = None
    joined = []
    for rows, is_rd in rows_per_seg:
        for t, l2, li in rows:
            if t_last is not None and t <= t_last:
                continue
            joined.append((t, l2, li, 1 if is_rd else 0))
            t_last = t
    return joined


def summarise(joined):
    if not joined:
        return None
    l2 = sorted(r[1] for r in joined)
    li = sorted(r[2] for r in joined)

    def med(v):
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    rd = [r for r in joined if r[3]]
    rec = {
        "t_start": joined[0][0],
        "t_end": joined[-1][0],
        "n_rows": len(joined),
        "L2_median": med(l2),
        "L2_max": l2[-1],
        "L2_final": joined[-1][1],
        "Linf_median": med(li),
        "Linf_max": li[-1],
        "Linf_final": joined[-1][2],
        "n_ringdown_rows": len(rd),
        "t_ringdown_start": rd[0][0] if rd else None,
        "t_inspiral_end": max([r[0] for r in joined if not r[3]] or [None]),
    }
    if rd:
        r2 = sorted(r[1] for r in rd)
        rec["L2_median_ringdown"] = med(r2)
        rec["L2_max_ringdown"] = r2[-1]
    return rec


def main():
    out = {}
    for series in sorted(os.listdir(ROOT)):
        sdir = os.path.join(ROOT, series)
        if not os.path.isdir(sdir):
            continue
        for sim in sorted(os.listdir(sdir)):
            simdir = os.path.join(sdir, sim)
            if not os.path.isdir(simdir):
                continue
            for ecc in sorted(os.listdir(simdir)):
                ev = os.path.join(simdir, ecc, "Ev")
                if not os.path.isdir(ev):
                    continue
                by_lev = {}
                for lev, name, path, is_rd in segments(ev):
                    by_lev.setdefault(lev, []).append(
                        (read_dat(path), is_rd))
                for lev, packs in sorted(by_lev.items()):
                    joined = join(packs)
                    rec = summarise(joined)
                    if rec is None:
                        continue
                    step = max(1, len(joined) // TARGET_POINTS)
                    key = "%s|%s|Lev%d" % (sim, ecc, lev)
                    out[key] = {
                        "n_segments": len(packs),
                        "n_segments_ringdown": sum(1 for _, rd in packs if rd),
                        "norm": rec,
                        "norm_series": [[r[0], r[1], r[2]]
                                        for r in joined[::step]],
                        "ringdown_series": [[r[0], r[1], r[2]]
                                            for r in joined[::step] if r[3]],
                    }
                    sys.stderr.write(
                        "%-40s n=%6d  t=[%.1f, %.1f]  rd_rows=%d\n"
                        % (key, rec["n_rows"], rec["t_start"], rec["t_end"],
                           rec["n_ringdown_rows"]))
    json.dump(out, sys.stdout)
    sys.stderr.write("\n%d records\n" % len(out))


if __name__ == "__main__":
    main()
