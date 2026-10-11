"""Validation: loader round-trip, undisrupted-twin evaluate(), fixed-flight checks, SUMMARY.md / README table.

    python -m arp.us2014.validate [names...]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter

from .. import evaluate as ev
from ..instance import load_instance
from . import INST_DIR


def twin(I):
    J = copy.copy(I)
    J.flight_disr, J.alt_cap, J.ac_unavail = {}, {}, {}
    return J


def check(name: str, root: str = INST_DIR) -> dict:
    folder = os.path.join(root, name)
    st = json.load(open(os.path.join(folder, "build_stats.json")))
    I = load_instance(folder, name)
    out = dict(name=name)
    # --- (b) loader round-trip against the generating tables
    rt = dict(flights=len(I.flights) == st["flights"], aircraft=len(I.aircraft) == st["tails"],
              itineraries=len(I.itineraries) == st["itineraries"],
              pax=sum(i.count for i in I.itineraries) == st["pax"],
              maint=sum(1 for a in I.aircraft.values() if a.maint) == st["maintenance"],
              disr=len(I.flight_disr) == st["disr_delays"] + st["disr_cancels"],
              outages=sum(len(v) for v in I.ac_unavail.values()) == st["aircraft_outages"],
              reductions=sum(len(v) for v in I.alt_cap.values()) == st["airport_reductions"],
              airports=len(I.airports) >= st["airports"])
    out["roundtrip_ok"] = all(rt.values())
    out["roundtrip"] = rt
    # --- (c) undisrupted twin
    J = twin(I)
    plan = {f.idx: (f.dep, f.arr, f.aircraft) for f in J.flights}
    e = ev.evaluate(J, plan)
    out["twin_n_violations"] = e["n_violations"]
    out["twin_violations"] = e["violations"]
    out["twin_pos_missing"] = e["pos_missing"]
    out["twin_spilled"] = e["spilled"]
    out["twin_total_cost"] = round(e["total"], 1)
    # --- structural checks (wiki validations)
    bad_conn = sum(1 for i in I.itineraries for a, b in zip(i.legs, i.legs[1:])
                   if I.flights[a[0]].dest != I.flights[b[0]].orig or I.flights[b[0]].dep - I.flights[a[0]].arr < 30)
    out["itin_bad_connections"] = bad_conn
    out["itin_legs_hist"] = dict(Counter(len(i.legs) for i in I.itineraries))
    # --- disrupted instance: baseline plan with historical delays applied to *fixed* flights must be consistent
    fixed_bad = 0
    for f in I.flights:
        if f.dep < I.start:
            d = I.flight_disr.get(f.idx, 0)
            if d < 0:
                continue
            for ap, t0 in ((f.orig, f.dep + d), (f.dest, f.arr + d)):
                if t0 + 60 > I.start and ev.capacity(I, I.alt_cap, ap, t0 // 60 * 60)[0 if ap == f.orig else 1] < 1:
                    fixed_bad += 1
    out["fixed_flights_hitting_zero_capacity"] = fixed_bad
    out["itin_refs_to_dropped"] = itin_refs(folder)
    out["ok"] = out["itin_refs_to_dropped"] == 0 and out["roundtrip_ok"] and out["twin_n_violations"] == 0 and out["twin_pos_missing"] == 0 and bad_conn == 0
    out["stats"] = st
    return out


def itin_refs(folder: str) -> int:
    """Itineraries whose legs are missing from rotations.csv (e.g. flights dropped by the rotation repair): must be 0."""
    keys = {(r.split()[0], r.split()[1]) for r in open(os.path.join(folder, "rotations.csv")) if r.strip() and r[0] != "#"}
    bad = 0
    for ln in open(os.path.join(folder, "itineraries.csv")):
        t = ln.split()
        if not t or t[0] == "#":
            continue
        legs = [(t[k], t[k + 1]) for k in range(4, len(t), 3)]
        bad += any(k not in keys for k in legs)
    return bad


WITNESS = "witness_plan.json"


def save_witness(I, plan, folder: str):
    """Compact JSON plan keyed 'flight number|date' -> null (cancelled) or [dep, arr, tail id] (absolute minutes)."""
    out = {}
    for f in I.flights:
        if f.is_transp:
            continue
        v = plan[f.idx]
        out[f"{f.num}|{f.date:%d/%m/%y}"] = None if v == "C" else [int(v[0]), int(v[1]), v[2]]
    json.dump(out, open(os.path.join(folder, WITNESS), "w"), separators=(",", ":"), sort_keys=True)


def check_witness(name: str, root: str = INST_DIR):
    """Re-evaluate the stored witness plan on the disrupted instance (never trusts a stored number). None if absent."""
    path = os.path.join(root, name, WITNESS)
    if not os.path.exists(path):
        return None
    I = load_instance(os.path.join(root, name), name)
    w = json.load(open(path))
    plan = {}
    for f in I.flights:
        if f.is_transp:
            continue
        v = w.get(f"{f.num}|{f.date:%d/%m/%y}", "missing")
        plan[f.idx] = "C" if v is None else (None if v == "missing" else (v[0], v[1], v[2]))
    plan = {k: v for k, v in plan.items() if v is not None}
    e = ev.evaluate(I, plan)
    return dict(n_violations=e["n_violations"], pos_missing=e["pos_missing"], violations=e["violations"],
                total=round(e["total"], 1), n_cancel=e["n_cancel"])


def fh_gate(name: str, root: str = INST_DIR, time_limit: float = 600.0) -> dict:
    """Exact full-horizon run (HiGHS, one solver process, in-process) on the DISRUPTED instance.
    'solver-feasible': the solver plan has n_violations == 0.  'hard (witness-feasible)': it does not, but the stored
    witness plan re-evaluates (now) to n_violations == 0 (pos_missing is recorded, not required: end positions are soft and
    historical cancellations can make a shortfall unavoidable).  Otherwise 'FAIL'."""
    import time
    from ..model import Params
    from ..policies import full_horizon, occ_rule, plan_to_warm
    I = load_instance(os.path.join(root, name), name)
    o = occ_rule(I)
    p = Params(time_limit=time_limit, lp_restrict=True, rc_keep=50.0, verbose=False)
    t0 = time.time()
    r = full_horizon(I, p, warm=plan_to_warm(o.plan))
    wall = time.time() - t0
    e = ev.evaluate(I, r.plan)
    lg = r.log[-1] if r.log else {}
    out = dict(obj=lg.get("obj"), bound=lg.get("bound"), gap=lg.get("gap"), status=lg.get("status"),
               solve_s=lg.get("solve"), wall_s=round(wall, 1), n_violations=e["n_violations"], violations=e["violations"],
               total=round(e["total"], 1), n_cancel=e["n_cancel"], n_flights=lg.get("n_flights"))
    return classify(out, name, root)


def classify(out: dict, name: str, root: str = INST_DIR) -> dict:
    if out["n_violations"] == 0:
        out["gate"] = "solver-feasible"
    else:
        w = check_witness(name, root)
        out["witness"] = w
        out["gate"] = "hard (witness-feasible)" if w and w["n_violations"] == 0 else "FAIL"
    out["pass_"] = out["gate"] != "FAIL"
    return out


def main():
    names = sys.argv[1:] or sorted(d for d in os.listdir(INST_DIR) if os.path.isdir(os.path.join(INST_DIR, d)))
    res = [check(n) for n in names]
    for r in res:
        print(r["name"], "OK" if r["ok"] else "FAIL", {k: r[k] for k in ("roundtrip_ok", "twin_n_violations", "twin_pos_missing",
              "twin_spilled", "itin_bad_connections", "twin_violations")})
    json.dump(res, open(os.path.join(INST_DIR, "validation.json"), "w"), indent=1, default=str)
    sys.exit(0 if all(r["ok"] for r in res) else 1)


if __name__ == "__main__":
    main()
