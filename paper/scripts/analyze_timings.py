#!/usr/bin/env python3
"""Reduce the raw sonic harvest into per-simulation wall-clock / CPU-hour tables."""
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

with open(os.path.join(DATA, "sims_harvest.json")) as f:
    H = json.load(f)

# TimeInfo.dat columns (SpEC): [1]=t  [2]=DayOfYear  [3]=Nprocs  [4]=T[hours]
#                              [5]=CPU-h  [6..9]=rates
I_T, I_DOY, I_NPROC, I_WALL, I_CPUH = 0, 1, 2, 3, 4

rows = []
for series, sims in H.items():
    for sim, ent in sims.items():
        for ecc, edat in ent["ecc_dirs"].items():
            for lev, segs in edat["segments"].items():
                t_end = 0.0
                wall = 0.0
                cpuh = 0.0
                nproc = None
                mtimes = []
                nseg = 0
                for s in segs:
                    ti = s.get("timeinfo")
                    if not ti:
                        continue
                    nseg += 1
                    last = ti["last"]
                    # SpEC's TimeInfo.dat T[hours] and CPU-h are cumulative over the
                    # whole run, not per segment -- take the running maximum.
                    t_end = max(t_end, last[I_T])
                    wall = max(wall, last[I_WALL])
                    cpuh = max(cpuh, last[I_CPUH])
                    nproc = int(last[I_NPROC])
                    if s.get("run_mtime_max"):
                        mtimes.append(s["run_mtime_max"])
                    if s.get("run_mtime_min"):
                        mtimes.append(s["run_mtime_min"])
                if nseg == 0:
                    continue
                rows.append(
                    dict(
                        series=series,
                        sim=sim,
                        ecc=ecc,
                        lev=int(lev),
                        nseg=nseg,
                        t_final_M=t_end,
                        wall_h=wall,
                        cpu_h=cpuh,
                        nproc=nproc,
                        speed_M_per_h=(t_end / wall) if wall > 0 else None,
                        start=min(mtimes) if mtimes else None,
                        end=max(mtimes) if mtimes else None,
                    )
                )

rows.sort(key=lambda r: (r["series"], r["sim"], r["lev"]))


def fmt(ts):
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


print(f"{'simulation':<22}{'Lev':>4}{'nseg':>5}{'t_f/M':>9}{'wall_h':>9}"
      f"{'CPU-h':>10}{'np':>4}{'M/h':>7}  {'start':<11}{'end':<11}{'days':>6}")
print("-" * 106)
for r in rows:
    days = ""
    if r["start"] and r["end"]:
        days = f"{(r['end'] - r['start']) / 86400:.1f}"
    print(f"{r['sim']:<22}{r['lev']:>4}{r['nseg']:>5}{r['t_final_M']:>9.0f}"
          f"{r['wall_h']:>9.1f}{r['cpu_h']:>10.0f}{r['nproc'] or 0:>4}"
          f"{(r['speed_M_per_h'] or 0):>7.0f}  {fmt(r['start']):<11}{fmt(r['end']):<11}{days:>6}")

tot_cpu = sum(r["cpu_h"] for r in rows)
tot_wall = sum(r["wall_h"] for r in rows)
alldates = [r["start"] for r in rows if r["start"]] + [r["end"] for r in rows if r["end"]]
print("-" * 106)
print(f"TOTAL: {len(rows)} (sim,Lev) runs   CPU-h = {tot_cpu:,.0f}   "
      f"summed wall-h = {tot_wall:,.0f}")
if alldates:
    print(f"Campaign window: {fmt(min(alldates))}  ->  {fmt(max(alldates))}")

with open(os.path.join(DATA, "timings.json"), "w") as f:
    json.dump(rows, f, indent=1)
print(f"\nwrote {os.path.join(DATA, 'timings.json')}")
