"""Recompute the numbers quoted in the manuscript from the Stage-2 result files and compare with the values written in
the text (see report/05_numbers_register.md).   python report/check_numbers.py   (repo root; exit 1 on any mismatch)
Result folders: arp_rh/results_v2 (ROADEF, dynamic), arp_rh/results_us2014_v2, arp_rh/results_cg_v2."""
import glob
import json
import os
import statistics as st
import sys

R = os.path.dirname(os.path.abspath(__file__)) + "/.."
A = R + "/arp_rh"
V2, US, CG = A + "/results_v2/", A + "/results_us2014_v2/", A + "/results_cg_v2/"
J = lambda p: json.load(open(p))
fails = []


def chk(name, got, want, tol=0.051):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) and not isinstance(want, bool) else got == want
    if not ok:
        fails.append((name, got, want))
    print(("ok   " if ok else "FAIL ") + f"{name}: {got if not isinstance(got, float) else round(got, 3)} (text: {want})")


# ------------------------------------------------------------------------------------------------ ROADEF one day
A1 = ["A01", "A02", "A03", "A04", "A06", "A07", "A08", "A09", "XA01", "XA03"]
T1 = {"FH": "fh-fine", "OCC": "occ", "STG": "rh-60-60-stage-none-static-fine", "STG30": "rh-30-30-stage-none-static-fine",
      "RHM": "rh-60-60-exec-none-static-fine", "M120": "rh-60-120-exec-none-static-fine",
      "M180": "rh-60-180-exec-none-static-fine", "M360": "rh-60-360-exec-none-static-fine",
      "T60": "rh-60-60-exec-boundary[kms]-static-fine", "T120": "rh-60-120-exec-boundary[kms]-static-fine",
      "T180": "rh-60-180-exec-boundary[kms]-static-fine", "T360": "rh-60-360-exec-boundary[kms]-static-fine",
      "DAY": "rh-60-day-exec-boundary[kms]-static-fine"}
D1 = {k: {n: J(V2 + f"{n}__{t}.json") for n in A1} for k, t in T1.items()}
g1 = lambda k, n: 100 * (D1[k][n]["eval"]["total"] / D1["FH"][n]["eval"]["total"] - 1)
m1 = lambda k: st.mean(g1(k, n) for n in A1)
fhc = [D1["FH"][n]["certificate"] for n in A1]
chk("A1 FH all LP Optimal", all(c["lp"]["status"] == "Optimal" for c in fhc), True)
chk("A1 FH max gap %", max(100 * c["eps"] for c in fhc), 0.07, 0.006)
chk("A1 FH exact count", sum(c["claim"] == "exact" for c in fhc), 4)
chk("A1 FH mean time", st.mean(D1["FH"][n]["runtime"] for n in A1), 126, 0.6)
chk("A1 FH max time", max(D1["FH"][n]["runtime"] for n in A1), 692, 0.6)
chk("A1 cols min k", min(D1["FH"][n]["log"][0]["cols"] for n in A1) / 1000, 52, 0.6)
chk("A1 cols max k", max(D1["FH"][n]["log"][0]["cols"] for n in A1) / 1000, 138, 0.6)
chk("A1 OCC min %", min(g1("OCC", n) for n in A1), 36, 0.6); chk("A1 OCC max %", max(g1("OCC", n) for n in A1), 238, 0.6)
for k, mu, mx in (("STG", 26.1, 60.5), ("STG30", 27.2, None), ("RHM", 22.8, None), ("M120", 8.5, None), ("M180", 3.3, None),
                  ("M360", 1.0, None), ("T60", 1.9, 3.1), ("T120", 0.8, None), ("T180", 0.5, None), ("T360", 0.1, None)):
    chk(f"A1 {k} mean", m1(k), mu)
    if mx is not None:
        chk(f"A1 {k} max", max(g1(k, n) for n in A1), mx)
chk("A1 DAY within FH tolerance", max(g1("DAY", n) for n in A1) <= 0.07, True)
chk("A1 T60 mean time", st.mean(D1["T60"][n]["runtime"] for n in A1), 6, 0.6)
chk("A1 T60 longest stage <= 2 s", max(D1["T60"][n]["max_stage_time"] for n in A1) <= 2, True)
chk("A1 M360/RHM time ratio", sum(D1["M360"][n]["runtime"] for n in A1) / sum(D1["RHM"][n]["runtime"] for n in A1), 18, 0.6)
chk("A1 T180/FH time ratio (one sixth)", sum(D1["T180"][n]["runtime"] for n in A1) / sum(D1["FH"][n]["runtime"] for n in A1), 1 / 6, 0.02)
br = lambda k, c: st.mean(D1[k][n]["eval"][c] for n in A1) / 1e3
chk("A1 STG broken kEUR", br("STG", "pax_cancel_cost"), 236, 0.6); chk("A1 FH broken kEUR", br("FH", "pax_cancel_cost"), 158, 0.6)
chk("A1 STG pos kEUR", br("STG", "pos_cost"), 76, 0.6); chk("A1 FH pos kEUR", br("FH", "pos_cost"), 9, 0.6)
chk("A1 T60 pos kEUR", br("T60", "pos_cost"), 12, 0.6); chk("A1 STG spill kEUR", br("STG", "spill_cost"), 26, 0.6)
chk("A1 STG misplaced", sum(D1["STG"][n]["eval"]["pos_missing"] for n in A1), 166)
chk("A1 FH misplaced", sum(D1["FH"][n]["eval"]["pos_missing"] for n in A1), 7)
chk("A1 STG cancellations", sum(D1["STG"][n]["eval"]["n_cancel"] for n in A1), 8)
# ablation (five instances)
AB = ["A01", "A02", "A03", "A04", "XA01"]
ab = lambda b, lk: st.mean(100 * (J(V2 + f"{n}__rh-60-{lk}-exec-{'none' if b == 'none' else f'boundary[{b}]'}-static-fine.json")["eval"]["total"]
                                  / D1["FH"][n]["eval"]["total"] - 1) for n in AB)
chk("abl none 60", ab("none", 60), 37.7); chk("abl k 60", ab("k", 60), 1.6); chk("abl none 180", ab("none", 180), 3.9)
chk("abl k 180", ab("k", 180), 0.4); chk("abl n 60", ab("n", 60), 16.6); chk("abl nkms 60", ab("nkms", 60), 3.2)
chk("abl kms 60", ab("kms", 60), 1.7); chk("abl c 60", ab("c", 60), 41.3); chk("abl ckms 60", ab("ckms", 60), 6.5)
chk("abl kv 60", ab("kv", 60), 1.5); chk("abl kv 180", ab("kv", 180), 0.5)
# regimes
for rg, lo, hi in (("eu261", None, None), ("eu2025", None, None)):
    pass
REG = ["A01", "A02", "A03", "A04", "A06", "A07", "A08", "A09"]
for rg in ("roadef", "eu261", "eu2025"):
    nm = lambda n: n if rg == "roadef" else f"{n}-{rg}"
    fh = {n: J(V2 + f"{nm(n)}__fh-fine.json") for n in REG}
    chk(f"{rg} no cancellation", sum(fh[n]["eval"]["n_cancel"] for n in REG), 0)
ta = lambda rg: st.mean(J(V2 + (f"{n}__" if rg == "roadef" else f"{n}-{rg}__") + "fh-fine.json")["eval"]["total_delay"] for n in REG)
chk("regime delay change < 4%", max(abs(ta(r) / ta("roadef") - 1) for r in ("eu261", "eu2025")) < 0.04, True)

# ------------------------------------------------------------------------------------------------ ROADEF two days
A2 = ["A05", "A10", "XA02", "XA04"]
def bk(n):
    import sys as _s
    _s.path.insert(0, A)
    from arp import reporting as rp
    recs = [J(f) for f in glob.glob(V2 + f"{n}__*.json")]
    return rp.reference(rp.group(recs)[n])["eval"]["total"]
BK = {n: bk(n) for n in A2}
g2 = lambda t: st.mean(100 * (J(V2 + f"{n}__{t}.json")["eval"]["total"] / BK[n] - 1) for n in A2)
chk("A2 T(60/120)", g2("rh-60-120-exec-boundary[kms]-static-fine"), 2.0); chk("A2 T(30/120)", g2("rh-30-120-exec-boundary[kms]-static-fine"), 2.8)
chk("A2 T(60/60)", g2("rh-60-60-exec-boundary[kms]-static-fine"), 7.6); chk("A2 RHM", g2("rh-60-60-exec-none-static-fine"), 11.4)
chk("A2 STG", g2("rh-60-60-stage-none-static-fine"), 48.2); chk("A2 RHM 360", g2("rh-60-360-exec-none-static-fine"), 21.5)
chk("A2 FH = OCC", all(abs(J(V2 + f"{n}__fh-fine.json")["eval"]["total"] - J(V2 + f"{n}__occ.json")["eval"]["total"]) < 1e-6 for n in A2), True)

# ------------------------------------------------------------------------------------------------ dynamic
by = {}
for f in glob.glob(V2 + "*-s?-c0.3-l60__*.json"):
    r = J(f)
    by.setdefault(r["name"], {})[r["policy"]] = r
gd = {}
for n, d in by.items():
    ref = d["fh-pi-fine"]["eval"]["total"]
    for p, r in d.items():
        if r["eval"]["feasible"]:
            gd.setdefault(p, []).append(100 * (r["eval"]["total"] / ref - 1))
chk("dyn scenarios", len(by), 40)
chk("dyn PI max gap %", max(100 * d["fh-pi-fine"]["certificate"]["eps"] for d in by.values()), 0.53, 0.006)
chk("dyn PI exact count", sum(d["fh-pi-fine"]["certificate"]["claim"] == "exact" for d in by.values()), 10)
for p, mu, med in (("rh-60-inf-exec-none-progressive-fine", 2.8, 1.5), ("rh-60-180-exec-boundary[kms]-progressive-fine", 3.0, 2.2),
                   ("rh-60-60-exec-boundary[kms]-progressive-fine", 5.0, 3.8), ("rh-60-60-exec-none-progressive-fine", 22.0, None),
                   ("fh-fine", 4.1, None), ("occ", 146, None), ("rh-60-day-exec-boundary[kms]-progressive-fine", None, 1.6)):
    if mu is not None:
        chk(f"dyn mean {p}", st.mean(gd[p]), mu, 0.51 if p == "occ" else 0.051)
    if med is not None:
        chk(f"dyn median {p}", st.median(gd[p]), med)
chk("dyn RH-inf max", max(gd["rh-60-inf-exec-none-progressive-fine"]), 18.6); chk("dyn day max", max(gd["rh-60-day-exec-boundary[kms]-progressive-fine"]), 53.2)
chk("dyn OCC feasible scenarios", len(gd["occ"]), 32)
rt = lambda p: st.mean(d[p]["runtime"] for d in by.values())
chk("dyn RH-inf time", rt("rh-60-inf-exec-none-progressive-fine"), 876, 0.6); chk("dyn T180 time", rt("rh-60-180-exec-boundary[kms]-progressive-fine"), 55, 0.6)
chk("dyn T60 time", rt("rh-60-60-exec-boundary[kms]-progressive-fine"), 11, 0.6)
day = [d["rh-60-day-exec-boundary[kms]-progressive-fine"]["certificate"]["stages"] for d in by.values()]
chk("dyn day runs with a non-optimal stage", sum(s["mip_status"] != ["Optimal"] for s in day), 10)
chk("dyn day runs with fallback", sum(s["fallbacks"] > 0 for s in day), 4)
chk("dyn all FH/FH-PI/RH plans feasible", all(r["eval"]["feasible"] for d in by.values() for p, r in d.items() if p != "occ"), True)

# ------------------------------------------------------------------------------------------------ US2014-R
INS = ["US03", "US08_EMB120", "US02_CRJ700", "US08_CRJ700", "US02_A320", "US08_A320", "US02_B737", "US02_CRJ200"]
G = lambda n: "coarse" if n in ("US08_A320", "US02_B737", "US02_CRJ200") else "fine"
TAG = {"OCC": "occ", "FH": "fh-{g}", "STG": "rh-60-60-stage-none-static-{g}", "RHM": "rh-60-60-exec-none-static-{g}",
       "T60": "rh-60-60-exec-boundary[kms]-static-{g}", "T180": "rh-60-180-exec-boundary[kms]-static-{g}",
       "T360": "rh-60-360-exec-boundary[kms]-static-{g}", "DAY": "rh-60-day-exec-boundary[kms]-static-{g}"}
D = {n: {k: J(US + f"{n}__{t.format(g=G(n))}.json") for k, t in TAG.items()} for n in INS}
ev = lambda n, k: D[n][k]["eval"]
gap = lambda n, k: 100 * (ev(n, k)["total"] / ev(n, "FH")["total"] - 1)
rng = lambda k: (min(gap(n, k) for n in INS), max(gap(n, k) for n in INS))
mean = lambda k: st.mean(gap(n, k) for n in INS)
mv = lambda k: sum(v for n in INS for kk, v in ev(n, k)["violations"].items() if kk.startswith("maint"))
chk("US FH max gap %", max(100 * D[n]["FH"]["certificate"]["eps"] for n in INS), 0.24, 0.006)
chk("US FH exact count", sum(D[n]["FH"]["certificate"]["claim"] == "exact" for n in INS), 4)
chk("US FH min time", min(D[n]["FH"]["runtime"] for n in INS), 53, 0.6); chk("US FH max time", max(D[n]["FH"]["runtime"] for n in INS), 842, 0.6)
chk("US FH violations", sum(ev(n, "FH")["n_violations"] for n in INS), 0)
chk("US02_B737 FH gap %", 100 * D["US02_B737"]["FH"]["certificate"]["eps"], 0.16, 0.006)
chk("OCC gap min", rng("OCC")[0], 46, 0.6); chk("OCC gap max", rng("OCC")[1], 184, 0.6)
ratios = [ev(n, "OCC")["n_cancel"] / ev(n, "FH")["n_cancel"] for n in INS]
chk("OCC/FH canc min", min(ratios), 3.5, 0.06); chk("OCC/FH canc max", max(ratios), 26, 0.6)
chk("OCC infeasible", sorted(n for n in INS if not ev(n, "OCC")["feasible"]), ["US02_A320", "US02_B737"])
for k, lo, hi, mu in (("STG", 21.8, 101.1, 68.9), ("RHM", 21.1, 66.7, 51.8), ("T60", 12.3, 54.6, 29.1), ("T180", 4.8, 27.4, 15.0),
                      ("T360", 1.1, 7.6, 3.6), ("DAY", 0.0, 7.0, 2.0)):
    chk(f"US {k} min", rng(k)[0], lo); chk(f"US {k} max", rng(k)[1], hi); chk(f"US {k} mean", mean(k), mu)
chk("T60 beats STG and RHM on all", all(gap(n, "T60") < min(gap(n, "STG"), gap(n, "RHM")) for n in INS), True)
chk("T180 better than T60 on all", all(gap(n, "T180") < gap(n, "T60") for n in INS), True)
for n, a, b in (("US02_A320", 47.3, 22.0), ("US08_A320", 65.9, 40.7), ("US02_B737", 66.7, 54.6), ("US02_CRJ200", 49.8, 33.1)):
    chk(f"{n} RHM", gap(n, "RHM"), a); chk(f"{n} T60", gap(n, "T60"), b)
chk("maint viol T60", mv("T60"), 7); chk("maint viol STG", mv("STG"), 40); chk("maint viol RHM", mv("RHM"), 40)
chk("T60 fewer viol than both on all", all(ev(n, "T60")["n_violations"] < min(ev(n, "STG")["n_violations"], ev(n, "RHM")["n_violations"]) for n in INS), True)
chk("T360 viol", mv("T360"), 3); chk("DAY viol", mv("DAY"), 0)
chk("no hard violation STG/RHM/RH-T", all(ev(n, k)["feasible"] for n in INS for k in ("STG", "RHM", "T60", "T180", "T360", "DAY")), True)
chk("US max stage s", max(D[n][k]["max_stage_time"] for n in INS for k in ("T60", "T180", "T360", "DAY")), 671, 0.6)
chk("US fallbacks", sum(D[n][k]["fallbacks"] for n in INS for k in ("T60", "T180", "T360", "DAY", "STG", "RHM")), 0)
for k, lo, hi in (("T60", 11, 57), ("T180", 28, 275), ("T360", 72, 689), ("DAY", 185, 10935)):
    chk(f"{k} runtime min", min(D[n][k]["runtime"] for n in INS), lo, 0.6); chk(f"{k} runtime max", max(D[n][k]["runtime"] for n in INS), hi, 0.6)
NQ = {k: J(US + f"US02_A320_NQ__{t.format(g='fine')}.json") for k, t in TAG.items() if k not in ("T360", "DAY")}
ng = lambda k: 100 * (NQ[k]["eval"]["total"] / NQ["FH"]["eval"]["total"] - 1)
chk("NQ T60", ng("T60"), 0.7); chk("NQ T180", ng("T180"), 0.9); chk("NQ STG", ng("STG"), 29.7); chk("NQ RHM", ng("RHM"), 26.6)
chk("NQ OCC %", ng("OCC"), 5, 0.06); chk("NQ FH cost M", NQ["FH"]["eval"]["total"] / 1e6, 4.44, 0.006)
chk("A320 FH cost M", ev("US02_A320", "FH")["total"] / 1e6, 8.26, 0.006); chk("NQ FH cancellations", NQ["FH"]["eval"]["n_cancel"], 0)
chk("A320 FH cancellations", ev("US02_A320", "FH")["n_cancel"], 24); chk("A320 OCC %", gap("US02_A320", "OCC"), 179, 0.6)
chk("NQ STG viol", NQ["STG"]["eval"]["n_violations"], 10); chk("NQ RHM viol", NQ["RHM"]["eval"]["n_violations"], 10)
chk("A320 T60", gap("US02_A320", "T60"), 22.0); chk("A320 T180", gap("US02_A320", "T180"), 21.7)
chk("US03 T60", gap("US03", "T60"), 24.1); chk("US03 T180", gap("US03", "T180"), 8.3)

# ------------------------------------------------------------------------------------------------ column generation
a1, a2 = J(CG + "A01__cg-fine.json"), J(CG + "A02__cg-fine.json")
chk("CG A01 LP == flow LP", abs(a1["cg"]["lp_obj"] - a1["fh_lp_bound"]) < 1e-4, True)
chk("CG A02 LP == flow LP", abs(a2["cg"]["lp_obj"] - a2["fh_lp_bound"]) < 1e-4, True)
chk("CG A01 plan = FH", abs(a1["eval"]["total"] - D1["FH"]["A01"]["eval"]["total"]) < 1e-4, True)
chk("CG A02 plan = FH", abs(a2["eval"]["total"] - D1["FH"]["A02"]["eval"]["total"]) < 1e-4, True)
cu = {n: J(glob.glob(US + f"{n}__cg-*.json")[0]) for n in INS}
fhlp = {n: cu[n]["fh_lp_bound"] or D[n]["FH"]["bound"] for n in INS}
r = [cu[n]["lp_bound"] / fhlp[n] for n in INS]
chk("US CG RMP/flow LP min", min(r), 1.46, 0.006); chk("US CG RMP/flow LP max", max(r), 2.68, 0.006)


# ------------------------------------------------------------------------------------------------ B / XB / X (coarse grid)
sys.path.insert(0, A)
from arp import reporting as rp  # noqa: E402
BB = [f"B{i:02d}" for i in range(1, 11)] + [f"XB{i:02d}" for i in range(1, 5)]
GB = {n: rp.group([J(f) for f in glob.glob(V2 + f"{n}__*.json")])[n] for n in BB}
refB = {n: rp.reference(GB[n]) for n in BB}


def gB(lab):
    g = [100 * (GB[n][lab]["eval"]["total"] / refB[n]["eval"]["total"] - 1) for n in BB if lab in GB[n] and rp.feasible(GB[n][lab])]
    return st.mean(g)


chk("B STG ratio", 1 + gB("STG-60") / 100, 4.1); chk("B RHM ratio", 1 + gB("RH-M(60/60)") / 100, 2.9)
chk("B T60 ratio", 1 + gB("RH-T(60/60)") / 100, 2.6); chk("B T180 %", gB("RH-T(60/180)"), 58, 0.6); chk("B T360 %", gB("RH-T(60/360)"), 12, 0.6)
fhbest = sorted(n for n in BB if refB[n]["lab"] == "FH")
chk("B FH best known", fhbest, ["B01", "B03", "B04", "B08", "XB01", "XB02", "XB03"])
best_rh = lambda n: min(r["eval"]["total"] for k, r in GB[n].items() if k.startswith("RH") and rp.feasible(r))
adv = [100 * (1 - GB[n]["FH"]["eval"]["total"] / best_rh(n)) for n in fhbest]
chk("B FH vs best RH min %", min(adv), 6, 0.6); chk("B FH vs best RH max %", max(adv), 33, 0.6)
chk("B FH no maint viol", sorted(n for n in fhbest if not GB[n]["FH"]["eval"]["violations"]), ["B01", "B03", "B08", "XB03"])


def pg(t):
    r = J(V2 + t)
    c = r["certificate"]
    price = r["eval"]["total"] - c["problem"]["constant_offset_C"]
    return 100 * (price - c["bound"]) / price


chk("B08 gap", pg("B08__fh-coarse.json"), 9.3); chk("XB01 gap", pg("XB01__fh-coarse.json"), 4.5)
chk("XB02 gap", pg("XB02__fh-coarse.json"), 0.07, 0.006); chk("XB03 gap", pg("XB03__fh-coarse.json"), 5.0)
chk("B04 gap", pg("B04__fh-coarse-2h.json"), 3.1); chk("B01 4h gap", pg("B01__fh-coarse-4h.json"), 2.4); chk("B03 4h gap", pg("B03__fh-coarse-4h.json"), 2.5)
chk("B02 4h LP time h", J(V2 + "B02__fh-coarse-4h.json")["certificate"]["lp"]["time"] / 3600, 1.9, 0.06)
cr = ["B05", "B10", "XB04"]
red = [100 * (1 - refB[n]["eval"]["total"] / GB[n]["OCC"]["eval"]["total"]) for n in cr]
chk("B cap-red ref below OCC min %", min(red), 47, 0.6); chk("B cap-red ref below OCC max %", max(red), 59, 0.6)
mt = lambda n, lab: sum(v for k, v in GB[n][lab]["eval"]["violations"].items() if k.startswith("maint"))
sel = ["B01", "B02", "B03", "B04", "B06", "B07", "B08", "B09", "XB01", "XB03"]
chk("B T180 maint min", min(mt(n, "RH-T(60/180)") for n in sel), 2); chk("B T180 maint max", max(mt(n, "RH-T(60/180)") for n in sel), 5)
chk("B STG maint min", min(mt(n, "STG-60") for n in sel), 8); chk("B STG maint max", max(mt(n, "STG-60") for n in sel), 14)
day = [GB[n]["RH-T(60/day)"]["eval"]["total"] / 1e6 for n in BB if "RH-T(60/day)" in GB[n]]
chk("B day min MEUR", min(day), 96, 0.6); chk("B day max MEUR", max(day), 128, 0.6)
chk("B01 abl K 360 MEUR", GB["B01"]["RH-[k](60/360)"]["eval"]["total"] / 1e6, 9.5); chk("B01 KMS 360 MEUR", GB["B01"]["RH-T(60/360)"]["eval"]["total"] / 1e6, 9.1)
XX = ["X01", "X02", "X03", "X04"]
xo = {n: J(V2 + f"{n}__occ.json")["eval"] for n in XX}
x60 = {n: J(V2 + f"{n}__rh-60-60-exec-boundary[kms]-static-coarse.json") for n in XX}
xv = [100 * (1 - x60[n]["eval"]["total"] / xo[n]["total"]) for n in XX]
chk("X vs OCC min %", min(xv), 4, 0.6); chk("X vs OCC max %", max(xv), 17, 0.6)
chk("X runtime min", min(x60[n]["runtime"] for n in XX) / 60, 3, 0.6); chk("X runtime max", max(x60[n]["runtime"] for n in XX) / 60, 6, 0.6)
chk("X max stage <= 30 s", max(x60[n]["max_stage_time"] for n in XX) <= 30.5, True)
chk("X canc", (min(x60[n]["eval"]["n_cancel"] for n in XX), max(x60[n]["eval"]["n_cancel"] for n in XX)), (0, 12))
chk("X maint <= 1", max(sum(v for k, v in x60[n]["eval"]["violations"].items() if k.startswith("maint")) for n in XX) <= 1, True)
# ------------------------------------------------------------------------------------------------ CG B01 and US
b1, bi = J(CG + "B01__cg-coarse.json"), J(CG + "B01__cg-coarse-ipm.json")
lpB = J(V2 + "B01__fh-coarse-4h.json")["bound"]
chk("B01 flow LP", lpB, 7883673, 1); chk("B01 simplex ratio", b1["cg"]["lp_obj"] / lpB, 2.30, 0.006)
chk("B01 simplex iterations", b1["cg"]["iterations"], 11); chk("B01 LP share %", 100 * b1["cg"]["lp_time"] / b1["cg"]["time"], 90, 0.6)
chk("B01 IPM above flow LP %", 100 * (bi["cg"]["lp_obj"] / lpB - 1), 7.7); chk("B01 IPM iterations", bi["cg"]["iterations"], 28)
chk("B01 master rows", b1["master_rows"], 4859)
vcg = [100 * (cu[n]["eval"]["total"] / ev(n, "OCC")["total"] - 1) for n in INS]
chk("US CG vs OCC min", min(vcg), -5.7); chk("US CG vs OCC max", max(vcg), 13.6); chk("US CG = OCC count", sum(abs(x) < 0.005 for x in vcg), 4)
chk("US CG violations", sum(cu[n]["eval"]["n_violations"] for n in INS), 0)
chk("A01 CG iterations", a1["cg"]["iterations"], 65); chk("A02 CG iterations", a2["cg"]["iterations"], 42)
chk("A01 CG total s", a1["runtime"], 194, 0.6); chk("A02 CG total s", a2["runtime"], 56, 0.6)
chk("US02_B737 FH LP s", D["US02_B737"]["FH"]["certificate"]["lp"]["time"], 712, 0.6)
chk("A01 FH LP s", D1["FH"]["A01"]["certificate"]["lp"]["time"], 26, 0.6)

print("\nFAILED:" if fails else "\nall checks passed", fails if fails else "")
sys.exit(1 if fails else 0)
