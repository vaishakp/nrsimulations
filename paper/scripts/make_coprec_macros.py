#!/usr/bin/env python3
"""Emit macros_coprec.tex from the co-precessing / first-law / band studies."""
import os
import sys
import json
import numpy as np

from paper_dir import PAPER

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def load(f):
    p = os.path.join(DATA, f)
    return json.load(open(p)) if os.path.exists(p) else {}


def fmt(x, n=2):
    return f"{x:.{n}f}"


def sci(x, n=1):
    """LaTeX scientific notation, e.g. 6.7\\times10^{-3}."""
    if x == 0:
        return "0"
    ex = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** ex
    return rf"{mant:.{n}f}\times10^{{{ex}}}"


def loglog(x, y):
    """Power-law fit y ~ x^a.  Returns (a, sigma_a, r, scatter_factor).

    A bare `polyfit` slope is not a quotable result: over these sample sizes
    the standard error is large enough that the second decimal is noise.  Every
    exponent in this paper goes through here so that it can be quoted with its
    uncertainty, and so that the residual scatter -- which is what tells you
    whether a single power law describes the sample at all -- is available
    alongside it.
    """
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    n = len(lx)
    A = np.vstack([lx, np.ones(n)]).T
    beta, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ beta
    s2 = resid @ resid / (n - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return beta[0], se[0], np.corrcoef(lx, ly)[0, 1], float(np.exp(np.sqrt(s2)))


def main():
    surv = load("coprec_survey.json")
    ecc = load("eccentricity.json")
    fl = load("first_law.json")
    det = load("detectability.json")
    tw = load("twist_study.json")
    cat = load("catalog.json")
    M = {}

    # ---------------------------------------------------- co-precessing frame
    pr = [r for r in surv if r["prec"]]
    npr = [r for r in surv if not r["prec"]]
    mu_p = np.array([r["mu_cp"] for r in pr])
    mu_n = np.array([r["mu_cp"] for r in npr])
    b = np.array([r["beta_med"] for r in pr])
    M["NPrecCP"] = len(pr)
    M["NAlignCP"] = len(npr)
    M["MirrorAlignMin"] = sci(mu_n.min())
    M["MirrorAlignMax"] = sci(mu_n.max())
    M["MirrorPrecMin"] = fmt(100 * mu_p.min(), 1)
    M["MirrorPrecMax"] = fmt(100 * mu_p.max(), 1)
    M["MirrorPrecMed"] = fmt(100 * np.median(mu_p), 1)
    # A log--log slope here is 0.93 +- 0.19, i.e. indistinguishable from
    # linear, so we quote the violation as a range and a rate per degree
    # rather than as a fitted exponent.
    bexp, bexp_err, _, _ = loglog(b, mu_p)
    M["MirrorBetaExp"] = fmt(bexp, 2)
    M["MirrorBetaExpErr"] = fmt(bexp_err, 2)
    M["MirrorPerDeg"] = fmt(np.mean(100 * mu_p / b), 2)
    M["BetaMin"] = fmt(b.min(), 1)
    M["BetaMax"] = fmt(b.max(), 1)
    M["BetaMinRound"] = fmt(b.min(), 0)
    M["BetaMaxRound"] = fmt(b.max(), 0)
    ci = np.array([r["conc_in"] for r in pr])
    cc = np.array([r["conc_cp"] for r in pr])
    M["ConcInMin"] = fmt(ci.min(), 2)
    M["ConcCPMin"] = fmt(cc.min(), 3)
    M["ConcGainMed"] = fmt(np.median((1 - ci) / (1 - cc)), 0)

    # degeneracy between beta and e, which is why we cannot isolate an
    # eccentricity dependence
    pe = [(r["beta_med"], (ecc.get(r["name"]) or {}).get("ecc_median"))
          for r in pr]
    pe = [(x, y) for x, y in pe
          if y is not None and ((ecc.get("x") or {}).get("ecc_spread") or 0) < 1]
    pe = [(x, y) for x, y in pe if y is not None]
    if len(pe) > 3:
        bb = np.array([p[0] for p in pe])
        ee = np.array([p[1] for p in pe])
        M["BetaEccCorr"] = fmt(np.corrcoef(np.log(bb), ee)[0, 1], 2)
        M["NBetaEcc"] = len(pe)

    # ------------------------------------------------------------ eccentricity
    meas = {k: v for k, v in ecc.items() if v.get("ecc_median") is not None}
    rel = {k: v for k, v in meas.items() if (v.get("ecc_spread") or 1) < 0.02}
    M["NEccMeasured"] = len(meas)
    M["NEccReliable"] = len(rel)
    M["EccSpreadMed"] = sci(np.median([v["ecc_spread"] for v in rel.values()]))
    disag = [k for k, v in rel.items()
             if v.get("ecc_table") is not None
             and abs(v["ecc_median"] - v["ecc_table"])
             > max(0.02, 0.25 * v["ecc_table"])]
    M["NEccDisagree"] = len(disag)
    for tag, key in [("Long", "EccPrecDiff002"), ("LongA", "EccPrecDiff001")]:
        if key in ecc and ecc[key].get("ecc_median") is not None:
            M[f"Ecc{tag}Meas"] = fmt(ecc[key]["ecc_median"], 3)
            M[f"Ecc{tag}Table"] = fmt(ecc[key]["ecc_table"], 2)
            M[f"Ecc{tag}Spread"] = fmt(ecc[key]["ecc_spread"], 3)
    M["EccMeasMax"] = fmt(max(v["ecc_median"] for v in rel.values()), 3)

    # -------------------------------------------------------------- first law
    ok = {k: v for k, v in fl.items()
          if v.get("ecc") is not None and v["R_secular_med"] < 1.0}
    P = [(v["ecc"], v["R_secular_med"]) for k, v in ok.items()
         if cat.get(k, {}).get("precessing")]
    N = [(v["ecc"], v["R_secular_med"]) for k, v in ok.items()
         if not cat.get(k, {}).get("precessing")]
    allx = np.array([p[0] for p in P + N])
    ally = np.array([p[1] for p in P + N])
    M["NFirstLaw"] = len(ok)

    # The residue is a violation of a quasi-circular identity, so it is quoted
    # as a percentage and as a bound.  A pooled power-law exponent is NOT
    # quoted: the aligned-spin and precessing runs are offset from one another
    # by a factor of ~3 (FLRatio below) and occupy different parts of the e
    # range, so a single line through both returns an intermediate slope
    # belonging to neither population -- and one driven almost entirely by the
    # three runs above e = 0.3, two of which (EccPrecDiff001/002) are the same
    # physical configuration.  The exponents are reported per population.
    M["FLBest"] = sci(ally.min())
    M["FLBestPct"] = f"{100 * ally.min():.3f}"
    M["FLMedPct"] = fmt(100 * np.median(ally), 2)
    M["FLWorstPct"] = fmt(100 * ally.max(), 1)
    low = ally[allx < 0.2]
    M["FLLowEccCut"] = "0.2"
    M["FLNLowEcc"] = int((allx < 0.2).sum())
    M["FLLowEccBoundPct"] = fmt(100 * low.max(), 2)
    M["FLEccWorst"] = fmt(allx[ally.argmax()], 2)

    xN = np.array([p[0] for p in N]); yN = np.array([p[1] for p in N])
    xP = np.array([p[0] for p in P]); yP = np.array([p[1] for p in P])
    M["NFLAlign"] = len(N)
    M["NFLPrec"] = len(P)
    for tag, xx, yy in (("Align", xN, yN), ("Prec", xP, yP)):
        a, ae, r, sc = loglog(xx, yy)
        M[f"FLExp{tag}"] = fmt(a, 2)
        M[f"FLExp{tag}Err"] = fmt(ae, 2)
        M[f"FLCorr{tag}"] = fmt(r, 2)
        M[f"FLScatter{tag}"] = fmt(sc, 2)
    # the aligned-spin runs follow a clean quadratic; quote the coefficient
    k2 = yN / xN ** 2
    M["FLCoeffAlign"] = fmt(k2.mean(), 3)
    M["FLCoeffAlignErr"] = fmt(k2.std(ddof=1), 3)
    M["FLCoeffAlignPct"] = fmt(100 * k2.mean(), 1)
    M["FLCoeffAlignScatterPct"] = fmt(100 * k2.std(ddof=1) / k2.mean(), 0)

    lowP = [y for x, y in P if x < 0.2]
    lowN = [y for x, y in N if x < 0.2]
    M["FLPrecMed"] = sci(np.median(lowP))
    M["FLAlignMed"] = sci(np.median(lowN))
    M["FLPrecMedPct"] = fmt(100 * np.median(lowP), 2)
    M["FLAlignMedPct"] = fmt(100 * np.median(lowN), 2)
    M["FLRatio"] = fmt(np.median(lowP) / np.median(lowN), 1)

    # The outlier is selected on the *peak* residue, not the orbit-averaged
    # one: its median is unremarkable, so it stays in the sample above.
    bad = sorted(k for k, v in fl.items() if v["R_secular_max"] >= 1.0)
    M["FLOutlier"] = bad[0].replace("_", r"\_") if bad else "none"
    if bad:
        M["FLOutlierPeak"] = sci(fl[bad[0]]["R_secular_max"])
        M["FLOutlierMedPct"] = fmt(100 * fl[bad[0]]["R_secular_med"], 2)

    # ------------------------------------------- first law: frame comparison
    frames = load("first_law_frames.json")
    if frames:
        beta_of = {r["name"]: r["beta_med"] for r in surv}
        fa = [(k, v) for k, v in frames.items() if not v["precessing"]]
        fp = [(k, v) for k, v in frames.items() if v["precessing"]]
        ra = np.array([v["ratio"] for _, v in fa])
        rp = np.array([v["ratio"] for _, v in fp])
        M["NFLFrameAlign"] = len(fa)
        M["NFLFramePrec"] = len(fp)
        # aligned-spin runs have no precession, so the two frames must agree:
        # this is the null test for the whole construction.
        M["FLFrameNull"] = sci(np.abs(ra - 1).max())
        M["FLFrameRatioMin"] = fmt(rp.min(), 2)
        M["FLFrameRatioMed"] = fmt(np.median(rp), 1)
        M["FLFrameRatioMax"] = fmt(rp.max(), 0)
        pe = [(k, v) for k, v in fp if v.get("ecc")]
        if len(pe) > 3:
            ee = np.array([v["ecc"] for _, v in pe])
            # the same fit as FLExpPrec, but on the inertial-frame residue:
            # the eccentricity dependence does not survive the frame change.
            a, ae, r, _ = loglog(ee, np.array([v["R_in"] for _, v in pe]))
            M["FLExpPrecInert"] = fmt(a, 2)
            M["FLExpPrecInertErr"] = fmt(ae, 2)
            M["FLCorrPrecInert"] = fmt(r, 2)
        bp = np.array([beta_of[k] for k, _ in fp if k in beta_of])
        rb = np.array([v["ratio"] for k, v in fp if k in beta_of])
        if len(bp) > 3:
            M["FLFrameBetaCorr"] = fmt(
                np.corrcoef(np.log(bp), np.log(rb))[0, 1], 2)

    # --------------------------------------------------------- twisting study
    qc, ec = [], []
    for n, r in tw.items():
        e = r.get("ecc")
        a = r["results"].get("nrhybsur_qc") or r["results"].get("seob_qc")
        c = r["results"].get("seob_ecc")
        if e is None or a is None or "mismatch" not in a:
            continue
        qc.append((n, e, a["mismatch"]))
        if c and "mismatch" in c:
            ec.append((n, e, c["mismatch"], a["mismatch"]))
    loe = [m for n, e, m in qc if e < 0.03]
    if loe:
        M["TwistLowEccMM"] = sci(min(loe))
        M["TwistLowEccMMMax"] = sci(max(loe))
    mid = [(a / m) for n, e, m, a in ec if 0.03 < e < 0.25]
    if mid:
        M["TwistEccGainMed"] = fmt(np.median(mid), 0)
        M["TwistEccGainMax"] = fmt(max(mid), 0)
    hie = [m for n, e, m, a in ec if e > 0.3]
    if hie:
        M["TwistHighEccMM"] = fmt(min(hie), 2)
    M["NTwist"] = len(qc)

    # ------------------------------------------------------------ band / SNR
    if det:
        long_ = det.get("EccPrecDiff002")
        if long_:
            M["BandLongMmax"] = fmt(long_["M_max_Msun"], 0)
            M["BandLongDur"] = fmt(long_["duration_s_at_Mmax"], 1)
            M["BandLongSNRCE"] = fmt(long_["snr"]["CE"], 0)
            M["BandLongSNRET"] = fmt(long_["snr"]["ET"], 0)
            M["BandLongSNRAP"] = fmt(long_["snr"]["A+"], 0)
        mm = np.array([v["M_max_Msun"] for v in det.values()])
        M["BandMmaxMin"] = fmt(mm.min(), 0)
        M["BandMmaxMax"] = fmt(mm.max(), 0)

    # ------------------------------------------------------------ constraints
    con = load("constraints.json")
    if con:
        finest = {}
        for sim, e in con.items():
            ks = [int(k) for k in e if k.isdigit()]
            if ks:
                finest[sim] = e[str(max(ks))]
        l2 = np.array([v["L2_median"] for v in finest.values()])
        M["NConstraint"] = len(finest)
        M["ConMedian"] = sci(np.median(l2))
        M["ConMin"] = sci(l2.min())
        M["ConMax"] = sci(l2.max())
        rr, lens = [], []
        for sim, e in con.items():
            if "ratio_lo_hi" not in e:
                continue
            rr.append(e["ratio_lo_hi"])
            lens.append(finest[sim]["t_end"])
        rr, lens = np.array(rr), np.array(lens)
        M["NConRatio"] = len(rr)
        M["ConRatioMed"] = fmt(np.median(rr), 2)
        M["ConRatioLong"] = fmt(np.median(rr[lens > 10000]), 2)
        M["ConRatioShort"] = fmt(np.median(rr[lens < 5000]), 2)
        M["NConLong"] = int((lens > 10000).sum())
        M["NConShort"] = int((lens < 5000).sum())
        from scipy.stats import spearmanr
        M["ConLenCorr"] = fmt(spearmanr(lens, rr)[0], 2)
        lo = np.array([finest[s]["L2_median"] for s in con
                       if finest[s]["t_end"] > 10000])
        hi = np.array([finest[s]["L2_median"] for s in con
                       if finest[s]["t_end"] < 5000])
        M["ConLongMed"] = sci(np.median(lo))
        M["ConShortMed"] = sci(np.median(hi))

    cm = load("constraint_merger.json")
    if cm:
        M["ConJunkFactor"] = fmt(cm["junk"]["median"], 0)
        M["ConMergeLTwo"] = fmt(cm["500M"]["L2_median"], 2)
        M["ConMergeLinf"] = fmt(cm["500M"]["Linf_median"], 2)
        M["ConMergeLTwoMax"] = fmt(cm["500M"]["L2_max"], 1)
        M["ConMergeLinfMax"] = fmt(cm["500M"]["Linf_max"], 1)
        M["ConMergeLTwoThirty"] = fmt(cm["30M"]["L2_median"], 2)
        M["NConMerge"] = cm["500M"]["n"]
        M["ConGapMed"] = fmt(cm["gap"]["median"], 0)
        M["NConGapClose"] = cm["gap"]["n_close"]
        M["NConGapTot"] = cm["gap"]["n"]

    out = os.path.join(PAPER, "macros_coprec.tex")
    with open(out, "w") as f:
        f.write("% autogenerated by scripts/make_coprec_macros.py"
                " -- do not edit\n")
        for k, v in M.items():
            f.write(rf"\newcommand{{\{k}}}{{{v}}}" + "\n")
    print(f"wrote macros_coprec.tex ({len(M)} macros)")
    for k, v in M.items():
        print(f"  \\{k:20s} = {v}")


if __name__ == "__main__":
    main()
