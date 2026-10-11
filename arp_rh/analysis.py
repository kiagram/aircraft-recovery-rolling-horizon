"""Aggregate results/*.json into paper tables (LaTeX) and figures.   python analysis.py [results_dir] [out_dir]"""
import collections
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("..", "paper")
os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "ink": "#0b0b0b", "ink2": "#52514e",
     "grid": "#e4e3df"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})

A1 = ["A01", "A02", "A03", "A04", "A06", "A07", "A08", "A09", "XA01", "XA03"]
A2 = ["A05", "A10", "XA02", "XA04"]
BB = [f"B{i:02d}" for i in range(1, 11)] + [f"XB{i:02d}" for i in range(1, 5)]
XX = ["X01", "X02", "X03", "X04"]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arp import reporting as rp   # noqa: E402  (R-5: feasibility-first selection, one grid per instance)

R = []
for f in glob.glob(os.path.join(RES, "*.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if not isinstance(r, dict) or "eval" not in r or "policy" not in r:
        continue
    r.pop("plan", None)
    R.append(r)
by = collections.defaultdict(dict, rp.group(R))       # name -> label -> best record (feasible first, then cost)
REFS = {}                                             # name -> reference record (printed and saved for audit)


def refrec(name, prefer=None):
    """Reference of an instance: the `prefer` method (e.g. the FH run) if it exists and is feasible, else the best
    feasible plan of any method on the instance's grid; an infeasible reference only if no feasible plan exists."""
    d = by.get(name, {})
    r = d.get(prefer) if prefer else None
    if r is None or not rp.feasible(r):
        r = rp.reference(d)
    REFS[name] = r
    return r


def ref(name, prefer=None):
    r = refrec(name, prefer)
    return None if r is None else r["eval"]["total"]


def gap(name, lab, reference=None):
    d = by[name]
    if lab not in d:
        return None
    rv = reference if reference is not None else ref(name)
    if rv is None:
        return None
    return 100.0 * (d[lab]["eval"]["total"] / rv - 1.0)


def gaps_of(pol, insts, prefer=None):
    """Gaps (%) of method `pol` on `insts`: feasible plans only (count of infeasible plans returned separately)."""
    g, rt, ms, n_inf = [], [], [], 0
    for n in insts:
        if n not in by or pol not in by[n]:
            continue
        r = by[n][pol]
        if not rp.feasible(r):
            n_inf += 1
            continue
        rv = ref(n, prefer)
        if rv is None:
            continue
        g.append(100 * (r["eval"]["total"] / rv - 1))
        rt.append(r["runtime"])
        ms.append(r.get("max_stage_time") or r["runtime"])
    return g, rt, ms, n_inf


def fmt(x, nd=1):
    if x is None or not np.isfinite(x):
        return "--"
    if round(x, nd) == 0:
        x = 0.0     # no "-0.0"
    return f"{x:,.{nd}f}"


summary = {}

# ---------------------------------------------------------------- Table: instances
def instance_table():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arp.instance import load
    rows = []
    for n in A1 + A2 + BB + XX:
        try:
            I = load(n, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roadef2009"))
        except Exception:
            continue
        nd = sum(1 for d in I.flight_disr.values() if d > 0)
        nc = sum(1 for d in I.flight_disr.values() if d < 0)
        rows.append((n, len(I.modifiable), len([a for a in I.aircraft.values() if not a.is_transp]), len(I.airports),
                     len(I.itineraries), (I.end - I.start) / 60, nd, nc, sum(len(v) for v in I.ac_unavail.values()),
                     sum(len(v) for v in I.alt_cap.values()), sum(1 for a in I.aircraft.values() if a.maint)))
    with open(os.path.join(OUT, "tables", "tab_instances.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrrrrrrrr}\\toprule\n")
        fh.write("Inst. & Open fl. & Aircraft & Airports & Itin. & Window (h) & Delays & Canc. & Outages & Cap. red. & Maint.\\\\\\midrule\n")
        for r in rows:
            fh.write(" & ".join([r[0]] + [f"{v:,}" if isinstance(v, int) else f"{v:.0f}" for v in r[1:]]) + "\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")


# ---------------------------------------------------------------- Table: FH performance
def fh_table():
    lines = []
    for n in A1 + A2 + BB:
        d = by.get(n, {})
        if "FH" not in d:
            continue
        r = d["FH"]
        e = r["eval"]
        lg = r["log"][0] if r["log"] else {}
        occ = d.get("OCC", {}).get("eval", {}).get("total")
        gp = r.get("gap")
        lines.append((n, lg.get("cols"), lg.get("rows"), r.get("bound"), r.get("obj"), None if gp is None else 100 * gp,
                      r["runtime"], e["total"], occ, e["n_cancel"], e["total_delay"], rp.mark(r)))
    with open(os.path.join(OUT, "tables", "tab_fh.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrrrrrr}\\toprule\n")
        fh.write("Inst. & Columns & Rows & LP bound & MIP obj. & Gap (\\%) & Time (s) & Cost FH & Cost OCC\\\\\\midrule\n")
        for l in lines:
            fh.write(f"{l[0]} & {l[1]:,} & {l[2]:,} & {fmt(l[3], 0)} & {fmt(l[4], 0)} & {fmt(l[5], 2)} & {l[6]:,.0f} & "
                     f"{l[7]:,.0f} & {fmt(l[8], 0)}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["fh"] = lines
    fh_body_table(lines)


def fh_body_table(lines):
    """Condensed FH table for the body: groups of instances (full per-instance table: tab_fh.tex, appendix)."""
    L = {l[0]: l for l in lines}
    # Stage-2 runs: B01-B03 4 h, B04 2 h, the other B/XB 1 h (D6 coarse grid)
    groups = [("A, one day", A1), ("A, two days", A2), ("B01--B03 (4\\,h), B04 (2\\,h)", ["B01", "B02", "B03", "B04"]),
              ("B08, XB01--XB03", ["B08", "XB01", "XB02", "XB03"]),
              ("B05--B07, B09, B10, XB04", ["B05", "B06", "B07", "B09", "B10", "XB04"])]
    rows = []
    with open(os.path.join(OUT, "tables", "tab_fh_body.tex"), "w") as fh:
        fh.write("% generated by arp_rh/analysis.py (fh_body_table); per-instance values: tab_fh.tex\n")
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        fh.write("Group & $n$ & Columns (k) & Gap to LP bound (\\%) & Time (s) & FH / OCC cost\\\\\\midrule\n")
        for name, ins in groups:
            ls = [L[i] for i in ins if i in L]
            if not ls:
                continue
            cols = [l[1] for l in ls if l[1]]
            gaps = [l[5] for l in ls if l[5] is not None and np.isfinite(l[5])]
            ts = [l[6] for l in ls]
            rat = [l[7] / l[8] for l in ls if l[8]]
            gtxt = "--" if not gaps else (f"{np.mean(gaps):.2f} (max {max(gaps):.2f})" if len(gaps) == len(ls) else
                                          f"{np.mean(gaps):.1f} (max {max(gaps):.1f}); {len(ls) - len(gaps)} without")
            fh.write(f"{name} & {len(ls)} & {f"{min(cols) / 1000:.0f}--{max(cols) / 1000:.0f}" if round(min(cols) / 1000) != round(max(cols) / 1000) else f"{min(cols) / 1000:.0f}"} & {gtxt} & "
                     f"{np.mean(ts):,.0f} (max {max(ts):,.0f}) & "
                     + (f"{min(rat):.2f}--{max(rat):.2f}" if rat else "--") + "\\\\\n")
            rows.append((name, len(ls), gaps, ts, rat))
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["fh_body"] = rows


# ---------------------------------------------------------------- Table: price of myopia
POLS_A = ["OCC", "STG-60", "STG-30", "RH-M(60/60)", "RH-M(60/120)", "RH-M(60/180)", "RH-M(60/360)",
          "RH-T(30/30)", "RH-T(60/60)", "RH-T(30/120)", "RH-T(60/120)", "RH-T(60/180)", "RH-T(60/360)", "RH-T(60/day)", "FH"]


def myopia_table():
    groups = [("A-set, 1 day (certified FH)", A1, "FH"), ("A-set, 2 days", A2, None), ("B/XB sets", BB, None)]
    rows = []
    stats = {}
    for pol in POLS_A:
        row = [pol]
        for gname, insts, refl in groups:
            g, rt, ms, n_inf = gaps_of(pol, insts, prefer=refl)
            stats[(pol, gname)] = (np.mean(g) if g else None, np.max(g) if g else None, np.mean(rt) if rt else None,
                                  np.mean(ms) if ms else None, len(g), n_inf)
            row += [stats[(pol, gname)]]
        rows.append(row)
    with open(os.path.join(OUT, "tables", "tab_myopia.tex"), "w") as fh:
        fh.write("% generated by arp_rh/analysis.py; means over feasible plans; $^{\\dagger k}$ = k infeasible plans excluded\n")
        fh.write("\\begin{tabular}{l" + "rrr" * len(groups) + "}\\toprule\n")
        fh.write(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{g[0]}}}" for g in groups) + "\\\\\n")
        fh.write("Policy" + " & Gap (\\%) & Max (\\%) & Time (s)" * len(groups) + "\\\\\\midrule\n")
        for row in rows:
            cells = [row[0]]
            for s in row[1:]:
                cells += [fmt(s[0]) + (f"$^{{\\dagger {s[5]}}}$" if s[5] else ""), fmt(s[1]), fmt(s[2], 0)]
            fh.write(" & ".join(cells) + "\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["myopia"] = {f"{k[0]}|{k[1]}": v for k, v in stats.items()}
    return stats


# ---------------------------------------------------------------- Figure: gap vs look-ahead
def fig_lookahead(stats):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    looks = [60, 120, 180, 360]
    Bsub = [n for n in BB if n in by and "RH-T(60/360)" in by[n] and "RH-M(60/360)" in by[n]]
    for ax, gname, insts in zip(axes, ["A-set, 1 day (certified FH)", "B/XB"], [A1, Bsub]):
        for fam, col, mk, lab, dy in [("RH-M", C["s2"], "s", "Myopic RH", 6), ("RH-T", C["s1"], "o", "RH + targets", -10)]:
            xs, ys = [], []
            for lk in looks:
                pol = f"{fam}(60/{lk})"
                g, _, _, n_inf = gaps_of(pol, insts, prefer="FH" if gname.startswith("A-set") else None)
                if len(g) == len(insts) and g:      # only complete, all-feasible series are plotted
                    xs.append(lk)
                    ys.append(np.mean(g))
            if xs:
                ax.plot(xs, ys, color=col, marker=mk, markersize=6, label=lab)
        g = gaps_of("STG-60", insts, prefer="FH" if gname.startswith("A-set") else None)[0]
        if g:
            ax.axhline(np.mean(g), color=C["ink2"], linewidth=1, linestyle="--")
            ax.text(62, np.mean(g), "thesis staging (60)", va="bottom", fontsize=8, color=C["ink2"])
        ax.set_xticks(looks)
        ax.set_xlabel(r"Look-ahead $\Lambda$ (min), step $\Delta$ = 60")
        ax.set_title(gname + ("" if gname.startswith("A-set") else f" ({len(insts)} instances, best-known ref.)"), fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, loc="center right")
    axes[0].set_ylabel("Mean cost gap to reference (%)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figs", f"fig_lookahead.{ext}"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: quality-time frontier (A1)
def fig_frontier(stats):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    g = "A-set, 1 day (certified FH)"
    pts = []
    for pol in POLS_A:
        s = stats.get((pol, g))
        if s and s[0] is not None and s[2] is not None:
            pts.append((pol, s[2], s[0]))
    for pol, t, v in pts:
        col = C["s1"] if pol.startswith("RH-T") else C["s2"] if pol.startswith("RH-M") else C["ink2"]
        mk = "o" if pol.startswith("RH-T") else "s" if pol.startswith("RH-M") else "D"
        t = max(t, 0.5)
        ax.scatter(t, max(v, 0.0), color=col, marker=mk, s=36, zorder=3, edgecolor="white", linewidth=1)
        off = {"STG-60": (-8, 6), "RH-M(60/60)": (6, -2), "FH": (-14, -12), "RH-T(60/day)": (4, 6), "OCC": (6, -4)}
        if pol in ("STG-60", "RH-M(60/60)", "RH-T(60/60)", "RH-T(60/180)", "RH-T(60/day)", "FH", "OCC", "RH-M(60/180)",
                   "RH-M(60/360)", "RH-T(30/120)"):
            ax.annotate(pol, (t, max(v, 0.0)), textcoords="offset points", xytext=off.get(pol, (5, 2)), fontsize=7,
                        color=C["ink"], ha="right" if off.get(pol, (5, 2))[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mean total run time (s, log scale; OCC < 1 s)")
    ax.set_ylabel("Mean gap to FH optimum (%)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figs", f"fig_frontier.{ext}"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: cost breakdown
def fig_breakdown():
    pols = ["OCC", "STG-60", "RH-M(60/60)", "RH-T(60/60)", "RH-T(60/180)", "FH"]
    comps = [("op_delta", "Operating cost change"), ("pax_cancel_cost", "Broken itineraries"),
             ("pax_delay_cost", "Passenger delay"), ("spill_cost", "Seat spill"), ("pos_cost", "Positioning")]
    cols = [C["s4"], C["s2"], C["s1"], C["s3"], "#4a3aa7"]
    M = np.zeros((len(pols), len(comps)))
    for i, pol in enumerate(pols):
        for n in A1:
            if n in by and pol in by[n]:
                e = by[n][pol]["eval"]
                for j, (c, _) in enumerate(comps):
                    M[i, j] += e[c] / 1e3
    M /= max(1, len(A1))
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    base_pos = np.zeros(len(pols))
    base_neg = np.zeros(len(pols))
    for j, (c, lab) in enumerate(comps):
        v = M[:, j]
        b = np.where(v >= 0, base_pos, base_neg)
        ax.barh(range(len(pols)), v, left=b, color=cols[j], label=lab, height=0.6, edgecolor="white", linewidth=1)
        base_pos += np.where(v >= 0, v, 0)
        base_neg += np.where(v < 0, v, 0)
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels(pols)
    ax.invert_yaxis()
    ax.set_xlabel("Mean recovery cost components per instance (kEUR), A-set one-day instances")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figs", f"fig_breakdown.{ext}"), dpi=200)
    plt.close(fig)
    summary["breakdown"] = {p: dict(zip([c for c, _ in comps], M[i].round(1).tolist())) for i, p in enumerate(pols)}


# ---------------------------------------------------------------- Ablation
def ablation_table():
    variants = [("RH-M", "none"), ("RH-[n]", "n"), ("RH-[c]", "c"), ("RH-[k]", "k"), ("RH-[km]", "km"),
                ("RH-[ks]", "ks"), ("RH-T", "kms"), ("RH-[kv]", "kv"), ("RH-[kvms]", "kvms"), ("RH-[nkms]", "nkms"),
                ("RH-[ckms]", "ckms")]
    insts = ["A01", "A02", "A03", "A04", "XA01"]
    out = []
    for vlab, b in variants:
        row = [b.upper() if b != "none" else "none"]
        for lk in (60, 180):
            lab = f"{vlab}(60/{lk})"
            g = gaps_of(lab, insts, prefer="FH")[0]
            row.append(np.mean(g) if len(g) == len(insts) else None)
        out.append(row)
    with open(os.path.join(OUT, "tables", "tab_ablation.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrr}\\toprule\n")
        fh.write("Targets & A-set 60/60 & A-set 60/180\\\\\\midrule\n")
        for r in out:
            fh.write(" & ".join([r[0]] + [fmt(v) for v in r[1:]]) + "\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["ablation"] = out


# ---------------------------------------------------------------- Dynamic
def dynamic_table():
    pols = ["OCC", "RH-M(60/60)", "RH-T(60/60)", "RH-T(60/180)", "RH-T(60/day)", "RH-inf", "FH", "FH-PI"]
    gaps = collections.defaultdict(list)
    rts = collections.defaultdict(list)
    n_inf = collections.Counter()
    for name, d in by.items():
        if "-s" not in name or "eu" in name:
            continue
        # reference: the perfect-information FH model with the same physical rules (R-6) if available
        r0 = d.get("FH-PI") or d.get("FH")
        if r0 is None:
            continue
        REFS[name] = r0
        if not rp.feasible(r0):
            n_inf["reference"] += 1
            continue
        refv = r0["eval"]["total"]
        for pol in pols:
            if pol in d:
                if not rp.feasible(d[pol]):
                    n_inf[pol] += 1
                    continue
                gaps[pol].append(100 * (d[pol]["eval"]["total"] / refv - 1))
                rts[pol].append(d[pol]["runtime"])
    with open(os.path.join(OUT, "tables", "tab_dynamic.tex"), "w") as fh:
        fh.write("% generated by arp_rh/analysis.py; feasible plans and feasible references only"
                 f"; excluded: {dict(n_inf)}\n")
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        fh.write("Policy & $n$ & Mean gap (\\%) & Median (\\%) & Max (\\%) & Time (s)\\\\\\midrule\n")
        for pol in pols:
            g = gaps.get(pol)
            if g:
                fh.write(f"{pol} & {len(g)} & {np.mean(g):.1f} & {np.median(g):.1f} & {np.max(g):.1f} & {np.mean(rts[pol]):.0f}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["dynamic_excluded_infeasible"] = dict(n_inf)
    summary["dynamic"] = {p: (len(g), float(np.mean(g)), float(np.median(g)), float(np.max(g))) for p, g in gaps.items()}
    if gaps:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        pl = [p for p in pols if p in gaps and p != ("FH-PI" if "FH-PI" in gaps else "FH")]
        data = [gaps[p] for p in pl]
        bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color=C["ink"]), flierprops=dict(marker="o", markersize=3))
        for patch, p in zip(bp["boxes"], pl):
            patch.set_facecolor(C["s1"] if p.startswith("RH-T") else C["s2"] if p.startswith("RH-M") else "#d9d8d4")
            patch.set_edgecolor("white")
        ax.set_yticks(range(1, len(pl) + 1))
        ax.set_yticklabels(pl)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Cost gap to the perfect-information reference FH-PI (%)" if "FH-PI" in gaps
                      else "Cost gap to clairvoyant full-horizon optimum (%)")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, "figs", f"fig_dynamic.{ext}"), dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------- EU261 regimes
def regime_table():
    rows = []
    for rg in ("roadef", "eu261", "eu2025"):
        acc = collections.defaultdict(list)
        for n in ["A01", "A02", "A03", "A04", "A06", "A07", "A08", "A09"]:
            name = n if rg == "roadef" else f"{n}-{rg}"
            d = by.get(name, {})
            if "FH" not in d:
                continue
            REFS[name] = d["FH"]
            if not rp.feasible(d["FH"]):
                print("regime table: infeasible FH reference skipped", name)
                continue
            fe = d["FH"]["eval"]
            acc["cost"].append(fe["total"])
            acc["canc"].append(fe["n_cancel"])
            acc["delay"].append(fe["total_delay"])
            acc["d180"].append(fe.get("n_delayed15", 0))
            for pol in ("RH-T(60/60)", "STG-60"):
                if pol in d and rp.feasible(d[pol]):
                    acc[pol].append(100 * (d[pol]["eval"]["total"] / fe["total"] - 1))
        if acc["cost"]:
            rows.append((rg, np.mean(acc["cost"]), np.mean(acc["canc"]), np.mean(acc["delay"]),
                         np.mean(acc["RH-T(60/60)"]) if acc["RH-T(60/60)"] else None,
                         np.mean(acc["STG-60"]) if acc["STG-60"] else None, len(acc["cost"])))
    with open(os.path.join(OUT, "tables", "tab_regime.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        fh.write("Regime & FH cost & Cancellations & Delay (min) & Gap RH-T(60/60) (\\%) & Gap STG-60 (\\%)\\\\\\midrule\n")
        for r in rows:
            fh.write(f"{r[0]} & {r[1]:,.0f} & {r[2]:.1f} & {r[3]:,.0f} & {fmt(r[4])} & {fmt(r[5])}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["regime"] = rows


if __name__ == "__main__":
    try:
        instance_table()
    except Exception as ex:
        print("instance table failed", ex)
    fh_table()
    st = myopia_table()
    fig_lookahead(st)
    fig_frontier(st)
    fig_breakdown()
    ablation_table()
    dynamic_table()
    regime_table()
    with open(os.path.join(OUT, "tables", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("records", len(R), "instances", len(by))


# ---------------------------------------------------------------- B/XB per-instance table
def b_table():
    cols = ["FH", "OCC", "STG-60", "RH-M(60/60)", "RH-T(60/60)", "RH-T(60/180)", "RH-M(60/360)", "RH-T(60/360)", "RH-T(60/day)"]
    lines = []
    for n in BB:
        d = by.get(n, {})
        if not d:
            continue
        cells = [n]
        b = rp.best(d[c] for c in cols if c in d)          # bold = best feasible plan (R-5)
        refrec(n)
        for c in cols:
            if c in d:
                e = d[c]["eval"]
                mv = sum(v for k, v in e["violations"].items() if k.startswith("maint"))
                s = f"{e['total'] / 1e6:.1f}" + (f"$^{{{mv}}}$" if mv else "") + rp.mark(d[c])
                cells.append(f"\\textbf{{{s}}}" if d[c] is b and rp.feasible(b) else s)
            else:
                cells.append("--")
        lines.append(cells)
    with open(os.path.join(OUT, "tables", "tab_B.tex"), "w") as fh:
        fh.write("% generated by arp_rh/analysis.py; bold = best feasible plan; superscript = maintenance violations "
                 "(priced, D2); $\\dagger$ = infeasible (hard violations); $\\S$ = total evaluated before R-4\n")
        fh.write("\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule\n")
        heads = [re.match(r"([A-Z-]+?)-?(?:\((.*)\)|(\d+))?$", c).groups() for c in cols]   # two-row header
        fh.write("Inst. & " + " & ".join(h[0] for h in heads) + "\\\\\n")
        fh.write(" & " + " & ".join(h[1] or h[2] or "" for h in heads) + "\\\\\\midrule\n")
        for l in lines:
            fh.write(" & ".join(l) + "\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    summary["B"] = lines


b_table()
for n in by:                    # every instance's best-known reference, also those not used in a table above
    if n not in REFS:
        refrec(n)
summary["references"] = {n: None if r is None else dict(label=r["lab"], policy=r["policy"], total=r["eval"]["total"],
                                                          feasible=rp.feasible(r), hard=rp.hard_violations(r),
                                                          legacy=rp.legacy(r))
                         for n, r in sorted(REFS.items())}
bad = {n: v for n, v in summary["references"].items() if v is None or not v["feasible"]}
print(f"references: {len(REFS)}; infeasible or missing: {len(bad)}")
for n, v in bad.items():
    print("  ", rp.describe(n, REFS[n]))
with open(os.path.join(OUT, "tables", "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1, default=str)
