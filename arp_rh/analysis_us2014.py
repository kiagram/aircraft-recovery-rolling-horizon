"""US2014 campaign -> paper/tables/tab_us2014.tex, paper/figs/fig_us2014_gap.{pdf,png}.
   python analysis_us2014.py [results_dir] [out_dir]      (reads results_us2014/*.json and CG json written by run_cg.py)
Reference for all gaps is the best known cost (min evaluate() total over the methods of that instance); the FH column
also gives the solver's own gap to its bound.  All methods of an instance use the same delay grid (fine <= 1,000 flights).
R-5: references and "best" are chosen feasibility first (arp.reporting); runs on another grid than the instance's are
ignored; infeasible plans are marked with a dagger."""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_us2014")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper")
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
NQ = "US02_A320_NQ"
# order of the extra methods in tab_us2014_methods.tex
METHODS = [("STG-60", "STG-60"), ("RH-M 60", "RH-M 60/60"), ("RH-T 60", "RH-T 60/60"), ("RH-T 180", "RH-T 60/180"),
           ("RH-T 360", "RH-T 60/360"), ("RH-T day", "RH-T 60/day")]
INST = ["US03", "US08_EMB120", "US02_CRJ700", "US08_CRJ700", "US02_A320", "US08_A320", "US02_B737", "US02_CRJ200"]
HARD = "US02_B737"
sys.path.insert(0, HERE)
from arp import reporting as rp   # noqa: E402


def load_all():
    by = {}
    for f in glob.glob(os.path.join(RES, "*__*.json")):      # per-run results only (summary_*/structure_* files are outputs)
        r = json.load(open(f))
        r.pop("plan", None)
        if "cg" in r and "policy" not in r:
            lab, name = "CG", r["instance"]
        else:
            name, p = r["name"], r["policy"]
            m = re.match(r"rh-60-(60|180|360|day)-exec-boundary\[kms\]-static", p)
            lab = ("OCC" if p == "occ" else "FH" if p.startswith("fh") else
                   "STG-60" if p.startswith("rh-60-60-stage-none-static") else
                   "RH-M 60" if p.startswith("rh-60-60-exec-none-static") else
                   f"RH-T {m.group(1)}" if m else None)
        if lab:
            grid = r.get("grid") or ("coarse" if "coarse" in r.get("policy", "") else "fine")
            if lab != "OCC" and grid != rp.grid_of(r.get("instance", name)):
                print("ignored (grid differs from the instance grid, D6):", os.path.basename(f))
                continue
            old = by.setdefault(name, {}).get(lab)
            if old is None or rp.key(r) < rp.key(old):
                by[name][lab] = r
    return by


def cell(r, tot):
    """gap (%) to `tot`, violations in parentheses, dagger if infeasible"""
    return f"{100 * (r['eval']['total'] / tot - 1):.1f}" + (f" ({r['eval']['n_violations']})" if r["eval"]["n_violations"] else "") + rp.mark(r)


def fh_ref(d, n):
    fh = d["FH"]
    if not rp.feasible(fh):
        print("WARNING: FH reference infeasible on", n, rp.hard_violations(fh))
    return fh["eval"]["total"]


def structure_stats(n, root):
    """Quota structure of the open flights at their scheduled times (no disruption delays applied): share of used airport-hour
    cells whose planned movements reach/exceed the quota, and share of departure cells whose NEXT hour has zero departure quota."""
    from collections import Counter
    from arp.instance import load
    from arp.model import capacity
    I = load(n, root)
    op = [f for f in I.flights if f.dep >= I.start and not f.is_transp]
    dep, arr = Counter(), Counter()
    for f in op:
        dep[(f.orig, f.dep // 60)] += 1
        arr[(f.dest, f.arr // 60)] += 1
    tot = tight = over = nxt0 = 0
    for (a, h), c in dep.items():
        cd = capacity(I, I.alt_cap, a, h * 60)[0]
        tot += 1; tight += c >= cd; over += c > cd
        nxt0 += capacity(I, I.alt_cap, a, (h + 1) * 60)[0] == 0
    nd = tot
    for (a, h), c in arr.items():
        ca = capacity(I, I.alt_cap, a, h * 60)[1]
        tot += 1; tight += c >= ca; over += c > ca
    tails = len([a for a in I.aircraft.values() if not a.is_transp])
    return dict(n=n, open=len(op), tails=tails, per_tail=len(op) / tails, span_h=(I.end - I.start) / 60, airports=len(I.airports),
                tight=tight / tot, over=over / tot, next0=nxt0 / nd)


FLEET = {"US03": "EMB-120, wiki sc. 3", "US08_EMB120": "EMB-120, Aug.", "US02_CRJ700": "CRJ-700 (OO)", "US08_CRJ700": "CRJ-700 (OO), Aug.",
         "US02_A320": "A320 fam. (UA)", "US08_A320": "A320 fam. (UA), Aug.", "US02_B737": "B737 fam. (UA)", "US02_CRJ200": "CRJ-200 (OO)",
         NQ: "A320 fam. (UA), network quotas"}


def inst_table():
    """Characteristics of the eight scale-down instances (+ NQ twin) from us2014/instances/<name>/build_stats.json."""
    root = os.path.join(HERE, "..", "us2014", "instances")
    with open(os.path.join(OUT, "tables", "tab_us_inst.tex"), "w") as f:
        f.write("% generated by arp_rh/analysis_us2014.py (inst_table) from us2014/instances/*/build_stats.json\n")
        f.write("\\begin{tabular}{llrrrrrrrrrrr}\\toprule\n")
        f.write("Inst. & Fleet & Fl. & Tails & Airp. & Itin. & Pax (k) & Delays & Canc. & Outages & Red. & Maint. & Drop (\\%)\\\\\\midrule\n")
        for n in INST + [NQ]:
            st = json.load(open(os.path.join(root, n, "build_stats.json")))
            f.write(" & ".join([n.replace("_", "\\_"), FLEET[n], fmt(st["flights"]), fmt(st["tails"]), fmt(st["airports"]), fmt(st["itineraries"]),
                                fmt(st["pax"] / 1000, 1), fmt(st["disr_delays"]), fmt(st["disr_cancels"]), fmt(st["aircraft_outages"]),
                                fmt(st["airport_reductions"]), fmt(st["maintenance"]), fmt(100 * st["repair_drop_rate"], 1)]) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")


def results_table(by):
    """Body table: gap (%) to FH, violations in parentheses; FH time and gap to its own bound."""
    cols = [m for m in METHODS]
    with open(os.path.join(OUT, "tables", "tab_us_results.tex"), "w") as f:
        f.write("% generated by arp_rh/analysis_us2014.py (results_table); gap (%) to the FH plan of the same grid, maintenance violations in parentheses\n")
        f.write("\\begin{tabular}{llrrr" + "r" * len(cols) + "}\\toprule\n")
        f.write("Inst. & Grid & FH t (s) & FH gap (\\%) & OCC & " + " & ".join(l for _, l in cols) + "\\\\\\midrule\n")
        for n in INST:
            d = by[n]
            fh = d["FH"]
            tot = fh_ref(d, n)
            cells = [n.replace("_", "\\_"), fh["grid"], fmt(fh["runtime"]), fmt(abs(100 * (fh.get("gap") or 0)), 2),
                     cell(d["OCC"], tot)]
            for k, _ in cols:
                r = d.get(k)
                cells.append("--" if r is None else cell(r, tot))
            f.write(" & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")


def nq_table(by):
    """Quota-basis sensitivity: US02_A320 (own-flight quotas), its network-quota twin, and US03 (network quotas, different fleet)."""
    st = {r["n"]: r for r in json.load(open(os.path.join(RES, "structure_us2014.json")))}
    rows = ["US02_A320", NQ, "US03"]
    with open(os.path.join(OUT, "tables", "tab_us_nq.tex"), "w") as f:
        f.write("% generated by arp_rh/analysis_us2014.py (nq_table); gap (%) to FH; maintenance violations in parentheses\n")
        f.write("\\begin{tabular}{lrrrrrrrrr}\\toprule\n")
        f.write("Inst. & FH cost & FH canc. & Tight cells & Next hr 0 & STG-60 & RH-M 60/60 & RH-T 60/60 & RH-T 60/180 & RH-T 60/360\\\\\\midrule\n")
        for n in rows:
            d = by[n]
            tot = fh_ref(d, n)
            cells = [n.replace("_", "\\_"), fmt(tot), fmt(d["FH"]["eval"]["n_cancel"]), fmt(100 * st[n]["tight"]) + "\\%", fmt(100 * st[n]["next0"]) + "\\%"]
            for k in ("STG-60", "RH-M 60", "RH-T 60", "RH-T 180", "RH-T 360"):
                r = d.get(k)
                cells.append("--" if r is None else cell(r, tot))
            f.write(" & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")


STRUCT = ["A01", "B01"] + INST + [NQ]


def structure_table():
    root = os.path.join(HERE, "..", "roadef2009")
    rows = []
    for n in STRUCT:
        try:
            rows.append(structure_stats(n, root))
        except Exception as e:      # instance not built locally
            print("structure: skip", n, e)
    json.dump(rows, open(os.path.join(RES, "structure_us2014.json"), "w"), indent=1)
    with open(os.path.join(OUT, "tables", "tab_us_structure.tex"), "w") as f:
        f.write("% generated by arp_rh/analysis_us2014.py (structure_stats): quota structure of the open flights at scheduled times\n")
        f.write("\\begin{tabular}{lrrrrrrrr}\\toprule\n")
        f.write("Inst. & Open fl. & Tails & Fl./tail & Span (h) & Airports & Tight cells & Over quota & Next hour 0\\\\\\midrule\n")
        for r in rows:
            f.write(" & ".join([r["n"].replace("_", "\\_"), fmt(r["open"]), fmt(r["tails"]), fmt(r["per_tail"], 1), fmt(r["span_h"]),
                                fmt(r["airports"]), fmt(100 * r["tight"]) + "\\%", fmt(100 * r["over"]) + "\\%",
                                fmt(100 * r["next0"]) + "\\%"]) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")


def methods_table(by):
    """Gap (%) of every policy to the FH solution (same grid), cancellations, violations; plus the NQ sensitivity twin."""
    cols = [m for m in METHODS]
    out, summ = [], []
    for n in INST + [NQ]:
        d = by.get(n, {})
        if "FH" not in d:
            continue
        fh = fh_ref(d, n)
        row = dict(n=n, fh=fh, fh_feasible=rp.feasible(d["FH"]), fh_gap=(d["FH"].get("gap") or 0) * 100, fh_t=d["FH"]["runtime"], fh_canc=d["FH"]["eval"]["n_cancel"],
                   fh_viol=d["FH"]["eval"]["n_violations"], occ=d["OCC"]["eval"]["total"] / fh * 100 - 100 if "OCC" in d else None)
        for k, _ in cols:
            if k in d:
                e = d[k]["eval"]
                row[k] = dict(gap=100 * (e["total"] / fh - 1), feasible=rp.feasible(d[k]), viol=e["n_violations"], canc=e["n_cancel"], rt=d[k]["runtime"], cost=e["total"])
        out.append(row)
    summ = out
    json.dump(summ, open(os.path.join(RES, "summary_us2014_methods.json"), "w"), indent=1, default=str)


def fmt(x, nd=0):
    return "--" if x is None else f"{x:,.{nd}f}"


def main():
    sys.path.insert(0, HERE)
    from arp.instance import load
    by = load_all()
    rows, pts = [], {}
    for n in INST:
        d = by.get(n, {})
        if not d:
            continue
        I = load(n, os.path.join(HERE, "..", "roadef2009"))
        fl, tails = len(I.flights), len([a for a in I.aircraft.values() if not a.is_transp])
        tot = {k: r["eval"]["total"] for k, r in d.items()}
        bkr = rp.best(r for k, r in d.items() if k in ("OCC", "FH", "RH-T 60", "RH-T 180", "CG"))   # reference of tab_us2014: the original five methods, feasible first
        bk = bkr["eval"]["total"]
        print(rp.describe(n, dict(bkr, lab=next(k for k, r in d.items() if r is bkr))))
        g = lambda k: 100 * (tot[k] / bk - 1) if k in tot else None
        fh, cg = d.get("FH"), d.get("CG")
        viol = {k: r["eval"]["n_violations"] for k, r in d.items()}
        flag = "/".join(str(viol.get(k, "-")) for k in ["OCC", "FH", "RH-T 60", "RH-T 180", "CG"])
        grid = (fh or cg or {}).get("grid", "fine")
        rows.append(dict(n=n, fl=fl, tails=tails, grid=grid, occ=tot.get("OCC"), fh=tot.get("FH"),
                         fhb=fh and fh.get("bound"), fhg=fh and fh.get("gap") is not None and 100 * fh["gap"], fht=fh and fh["runtime"],
                         r60=tot.get("RH-T 60"), g60=g("RH-T 60"), r180=tot.get("RH-T 180"), g180=g("RH-T 180"),
                         cglb=cg and cg.get("lb_valid") if cg and cg.get("lb_valid", 0) > 0 else None,  # non-positive Lagrangian value = no usable bound
                         cglp=cg and cg.get("lp_bound"), cgub=cg and cg.get("ub"), cgt=cg and cg["runtime"],
                         cgv=tot.get("CG"), viol=flag, bk=bk))
        pts[n] = (fl, {k: g(k) for k in ["OCC", "FH", "RH-T 60", "RH-T 180", "CG"]})
    with open(os.path.join(OUT, "tables", "tab_us2014.tex"), "w") as f:
        f.write("% generated by arp_rh/analysis_us2014.py; gaps (%) vs best known cost; violations = OCC/FH/RH60/RH180/CG\n")
        f.write("\\begin{tabular}{lrrlrrrrrrrrrrrrl}\\toprule\n")
        f.write("Inst. & Fl. & Tails & Grid & OCC & FH & FH bound & FH t (s) & RH-T 60/60 & gap & RH-T 60/180 & gap & CG LB & CG UB & CG t (s) & CG gap & Viol.\\\\\\midrule\n")
        for r in rows:
            cgg = None if not r["cgub"] or not r["cglb"] else 100 * (r["cgub"] / r["cglb"] - 1)
            f.write(" & ".join([r["n"].replace("_", "\\_") + ("$^\\ast$" if r["n"] == HARD else ""), fmt(r["fl"]), fmt(r["tails"]), r["grid"],
                                fmt(r["occ"]), fmt(r["fh"]), fmt(r["fhb"]), fmt(r["fht"]), fmt(r["r60"]), fmt(r["g60"], 1),
                                fmt(r["r180"]), fmt(r["g180"], 1), fmt(r["cglb"]), fmt(r["cgub"]), fmt(r["cgt"]), fmt(cgg, 1),
                                r["viol"]]) + "\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    methods_table(by)
    structure_table()
    inst_table()
    results_table(by)
    nq_table(by)
    # figure: gap to FH of every rolling-horizon policy, instances ordered by size
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    sty = {"STG-60": ("#52514e", "s", "STG-60"), "RH-M 60": ("#8a6bbe", "x", "RH-M 60/60"), "RH-T 60": ("#eb6834", "^", "RH-T 60/60"),
           "RH-T 180": ("#1baf7a", "v", "RH-T 60/180"), "RH-T 360": ("#2a78d6", "o", "RH-T 60/360"), "RH-T day": ("#eda100", "D", "RH-T 60/day")}
    order = sorted(INST, key=lambda n: next(r["fl"] for r in rows if r["n"] == n))
    for k, (c, m, lab) in sty.items():
        ys = [100 * (by[n][k]["eval"]["total"] / by[n]["FH"]["eval"]["total"] - 1) if k in by[n] else None for n in order]
        ax.plot(range(len(order)), [max(y, 0.1) if y is not None else None for y in ys], color=c, marker=m, lw=1.2, ms=5, label=lab)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([n.replace("US0", "US").replace("_", "\n") + f"\n{next(r['fl'] for r in rows if r['n'] == n)}" for n in order], fontsize=7)
    ax.set_ylabel("Gap to FH (%)"); ax.set_yscale("log"); ax.set_ylim(0.1, 150)
    ax.legend(frameon=False, fontsize=7, ncol=3); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figs", f"fig_us2014_gap.{ext}"), dpi=200)
    json.dump(rows, open(os.path.join(RES, "summary_us2014.json"), "w"), indent=1, default=str)
    for r in rows:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
