#!/usr/bin/env python3
"""Parse the markdown parameter tables in EccentricAlignedPrecessing.md into a CSV."""
import ast
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "..", "..", "EccentricAlignedPrecessing.md")
OUT = os.path.join(HERE, "..", "data", "catalog.csv")

text = open(MD).read()


def split_tables(txt):
    """Yield (header_cells, [row_cells...]) for every markdown pipe table."""
    lines = txt.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[:\- |]+\|\s*$", lines[i + 1]):
            hdr = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            yield hdr, rows
            i = j
        else:
            i += 1


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def vec(s):
    try:
        v = ast.literal_eval(s)
        return [float(x) for x in v]
    except Exception:
        return None


def ncyc(s):
    """'{2: 48.4, 3: 48.37}' -> {2: 48.4, 3: 48.37}"""
    try:
        d = ast.literal_eval(s)
        return {int(k): float(v) for k, v in d.items() if v is not None}
    except Exception:
        return {}


def status(s):
    try:
        d = ast.literal_eval(s)
        return {int(k): v for k, v in d.items()}
    except Exception:
        return {}


sims = {}
for hdr, rows in split_tables(text):
    cols = {h: k for k, h in enumerate(hdr)}
    # the simulation name lives in the (unnamed) first column
    if hdr[0] != "":
        continue

    def g(r, *names):
        for n in names:
            if n in cols and cols[n] < len(r):
                return r[cols[n]]
        return None

    for r in rows:
        name = r[0]
        if not name:
            continue
        e = sims.setdefault(name, {"name": name})
        for key, fn, names in [
            ("q", num, ["MassRatio"]),
            ("D0", num, ["D0"]),
            ("Omega0", num, ["Omega0", "$Omega_{0}$"]),
            ("adot0", num, ["adot0"]),
            ("ecc", num, ["Eccentricity"]),
            ("a_semi", num, ["SemiMajorAxis"]),
            ("anomaly", num, ["AnomalyAngle"]),
            ("chi_eff", num, ["ChiEff", "$Chi_{eff}$"]),
            ("chi_p", num, ["ChiPrec", "$Chi_{p}$"]),
            ("t_ref", num, ["ReqRefTime", "RequestedReferenceTime"]),
            ("omega_ref", num, ["Omega_ref"]),
            ("chiA", vec, ["ChiA"]),
            ("chiB", vec, ["ChiB"]),
            ("chiA_ref", vec, ["ChiA_ref"]),
            ("chiB_ref", vec, ["ChiB_ref"]),
            ("MA_ref", num, ["MA_ref"]),
            ("MB_ref", num, ["MB_ref"]),
        ]:
            raw = g(r, *names)
            if raw in (None, "", "nan"):
                continue
            v = fn(raw)
            if v is not None:
                e[key] = v
        nc = ncyc(g(r, "Ncycles") or "")
        if nc:
            e.setdefault("ncycles", {}).update(nc)
        nc22 = ncyc(g(r, "Ncycles (2,2)") or "")
        if nc22:
            e.setdefault("ncycles22", {}).update(nc22)
        st = status(g(r, "Status") or "")
        if st:
            e.setdefault("status", {}).update(st)

# series tag
for n, e in sims.items():
    if n.startswith("ICTSEccParallel"):
        e["series"] = "ICTSEccParallel"
    elif n.startswith("EccContPrecDiff"):
        e["series"] = "EccContPrecDiff"
    elif n.startswith("EccPrecDiff") or n.startswith("eccprec"):
        e["series"] = "EccPrecDiff"
    else:
        e["series"] = "other"
    # precessing if either in-plane spin component is non-negligible
    ip = 0.0
    for k in ("chiA", "chiB"):
        v = e.get(k)
        if v:
            ip = max(ip, (v[0] ** 2 + v[1] ** 2) ** 0.5)
    e["inplane_spin"] = ip
    e["precessing"] = ip > 1e-3
    nc = e.get("ncycles", {})
    e["ncycles_max"] = max(nc.values()) if nc else None
    e["levels"] = sorted(e.get("status", {}).keys())
    e["n_completed"] = sum(1 for v in e.get("status", {}).values() if v == "Completed")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fields = ["name", "series", "q", "D0", "Omega0", "adot0", "ecc", "a_semi", "anomaly",
          "chi_eff", "chi_p", "inplane_spin", "precessing", "t_ref", "omega_ref",
          "ncycles_max", "n_completed", "chiA", "chiB", "chiA_ref", "chiB_ref"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for n in sorted(sims):
        w.writerow(sims[n])

with open(OUT.replace(".csv", ".json"), "w") as f:
    json.dump(sims, f, indent=1)

print(f"parsed {len(sims)} simulations -> {OUT}")
nprec = sum(1 for e in sims.values() if e["precessing"])
necc = sum(1 for e in sims.values() if (e.get("ecc") or 0) > 1e-3)
nboth = sum(1 for e in sims.values() if e["precessing"] and (e.get("ecc") or 0) > 1e-3)
print(f"  precessing: {nprec}   eccentric: {necc}   both: {nboth}")
longest = sorted((e for e in sims.values() if e.get("ncycles_max")),
                 key=lambda e: -e["ncycles_max"])[:6]
for e in longest:
    print(f"  {e['name']:<20} Ncyc={e['ncycles_max']:8.1f} q={e.get('q')} "
          f"ecc={e.get('ecc')} chi_p={e.get('chi_p')} prec={e['precessing']}")
