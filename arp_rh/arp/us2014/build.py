"""One-command build of the US2014 (rebuilt) instances.

    python -m arp.us2014.build --all                 # download (if needed) + every instance + validation + README table
    python -m arp.us2014.build --instance US03       # a single instance
    python -m arp.us2014.build --check-determinism   # rebuild US03 twice, compare file hashes
Outputs: us2014/instances/<name>/ (ROADEF 2009 files + tails.csv + build_stats.json), CHECKSUMS.json, SUMMARY.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from . import CACHE_DIR, INST_DIR, LABEL, US_DIR
from . import itineraries as it_mod
from . import scenarios as sc
from .build_base import build_demand, build_otp, fleet
from .export_roadef import checksums, write
from .make_models import load_models

_CTX = {}


def ctx():
    if not _CTX:
        base = build_otp()
        ns, one, t100 = build_demand()
        seg = pd.read_parquet(os.path.join(CACHE_DIR, "db1b_segments.parquet"))
        _CTX.update(base=base, ns=ns, one=one, t100=t100, seg=seg, ac=sc.aircraft_table(), M=load_models())
    return _CTX


def _hour_floor(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("h")


def fix_capacity(fl, alt: pd.DataFrame, base_cap: dict, rec: pd.Timestamp, disr: pd.Series):
    """AirportCapacitySimulator substitute: raise reduced rates to the number of immovable (fixed) movements.
    Returns (alt rows DataFrame ap,ts,dep,arr, number of updates in the pass)."""
    red = {}
    for r in alt.itertuples():
        k = (r.ap, r.ts)
        if k not in red:
            d0, a0 = sc.cap_at(base_cap[r.ap], r.ts.hour)
            red[k] = [d0, a0]
        red[k][0 if r.kind == "dep" else 1] = r.rate
    f = fl[fl.dep < rec].copy()
    dl = f.fid.map(disr).fillna(0).astype(int)
    f = f[dl >= 0]
    dl = dl[dl >= 0]
    f["adep"] = f.dep + pd.to_timedelta(dl, unit="m")
    f["aarr"] = f.arr + pd.to_timedelta(dl, unit="m")
    fixed = Counter()
    for r in f.itertuples():
        for ap, ts, kind in ((r.orig, r.adep.floor("h"), 0), (r.dest, r.aarr.floor("h"), 1)):
            if ts + pd.Timedelta(hours=1) > rec:
                fixed[(ap, ts, kind)] += 1
    n_upd = 0
    for (ap, ts, kind), c in fixed.items():
        if (ap, ts) in red and red[(ap, ts)][kind] < c:
            red[(ap, ts)][kind] = c
            n_upd += 1
    return red, n_upd


def periods_from(red: dict):
    """Merge consecutive hours with identical (dep, arr) per airport into alt_airports rows."""
    rows = []
    by_ap = {}
    for (ap, ts), (d_, a_) in red.items():
        by_ap.setdefault(ap, []).append((ts, d_, a_))
    for ap, lst in sorted(by_ap.items()):
        lst.sort()
        cur = None
        for ts, d_, a_ in lst:
            if cur and cur[1] == ts and cur[3] == d_ and cur[4] == a_:
                cur[1] = ts + pd.Timedelta(hours=1)
            else:
                if cur:
                    rows.append(tuple(cur))
                cur = [ts, ts + pd.Timedelta(hours=1), None, d_, a_]
            cur[2] = ap
        if cur:
            rows.append(tuple(cur))
    return [(r[2], r[0], r[1], r[3], r[4]) for r in rows]


def build_instance(name: str, seed: int = 1, out_root: str = INST_DIR, type_rule: str = "wiki") -> dict:
    t0 = time.time()
    C = ctx()
    spec = sc.specs()[name]
    s0, rec, s1 = sc.window(spec.day0)
    rng = np.random.default_rng(sc.seed_of(spec.seed_name or name, seed))
    ac_all, M, base = C["ac"], C["M"], C["base"]
    st = dict(name=name, label=LABEL, window=[str(s0), str(rec), str(s1)], seed=seed, note=spec.note)

    w, st1 = sc.select_flights(base, ac_all, spec)
    st.update(st1)
    # cancellations before the recovery start are history (and would break the executed chain): removed, counted
    pre = (w.cancelled == 1) & (w.dep < rec)
    st["precancelled_removed"] = int(pre.sum())
    fl, st2 = sc.repair_rotations(w[~pre], M.turn.to_dict())
    st.update(st2)
    st["repair_rerun_dropped"] = sc.repair_rotations(fl, M.turn.to_dict())[1]["flights_dropped_by_repair"]   # wiki: "0 invalid sub-rotations"
    fl["cabF"], fl["cabB"], fl["cabE"] = (fl.tailnum.map(ac_all[c]) for c in ("cabF", "cabB", "cabE"))
    fl = sc.assign_flight_numbers(fl)
    ac = sc.aircraft_for(fl, ac_all, M)

    disr = sc.flight_disruptions(fl, rec)
    disr, st["propagated_fixed_delays"] = sc.propagate_fixed_delays(fl, disr, rec, M.turn.to_dict())

    # airport capacity: wiki = network-wide counts of the scenario dates; sub-fleet instances = own flights
    d0, d1 = s0.normalize(), s1.normalize()
    net = base[(base.dep >= s0 - pd.Timedelta(days=1)) & (base.dep <= s1 + pd.Timedelta(days=1))]
    cap_net, cnt_net = sc.capacity_table(net, d0, d1)
    cap = cap_net if spec.cap_basis == "network" else sc.capacity_table(fl, d0, d1)[0]
    aps = sorted(set(fl.orig) | set(fl.dest))
    base_cap = sc.capacity_periods(cap, aps)
    st["airports"] = len(aps)

    outages = sc.aircraft_outages(fl, rec, s1, pd.DataFrame(columns=["tailnum", "start", "end"]), disr)
    red_rows = sc.airport_reductions(net, cnt_net, cap, rec, s1, set(aps), spec.cap_basis)
    red, upd1 = fix_capacity(fl, red_rows, base_cap, rec, disr)
    red, upd2 = fix_capacity(fl, pd.DataFrame([(ap, ts, "dep", d_) for (ap, ts), (d_, a_) in red.items()] +
                                              [(ap, ts, "arr", a_) for (ap, ts), (d_, a_) in red.items()],
                                              columns=["ap", "ts", "kind", "rate"]), base_cap, rec, disr)
    alt_ap = periods_from(red)
    maint = sc.gen_maintenance(fl, ac, rec, s0, s1, rng, disr, set(red), set(outages.tailnum))
    st.update(capacity_fix_updates_pass1=upd1, capacity_fix_updates_rerun=upd2)

    positions = {}
    for tail, a in ac.iterrows():
        positions.setdefault(a.end_ap, Counter())[(a.model, tuple(int(x) for x in a.cfg))] += 1

    itins, sti = it_mod.generate(fl, C["ns"], C["one"], C["seg"], C["t100"], rng)
    st["itineraries_stats"] = sti

    folder = os.path.join(out_root, name)
    keep = {}
    if os.path.isdir(folder):          # a stored feasibility witness survives a rebuild (it is re-evaluated by the gate)
        wp = os.path.join(folder, "witness_plan.json")
        if os.path.exists(wp):
            keep[wp] = open(wp, "rb").read()
        shutil.rmtree(folder)
    sums = write(folder, rec_start=rec, rec_end=s1, fl=fl, ac=ac, maint=maint, positions=positions, base_cap=base_cap,
                 alt_ap=alt_ap, alt_fl=disr, outages=outages, itins=itins, type_rule=type_rule)
    for wp, data in keep.items():
        os.makedirs(os.path.dirname(wp), exist_ok=True)
        open(wp, "wb").write(data)
    sums = checksums(folder)
    fixed = fl[fl.dep < rec]
    st.update(flights=len(fl), tails=len(ac), pax=sti["pax"], itineraries=sti["itineraries"],
              maintenance=len(maint), fixed_flights=len(fixed), models=int(ac.model.nunique()),
              disr_delays=int((disr > 0).sum()), disr_cancels=int((disr < 0).sum()),
              disr_after_start=int(((disr.index.map(fl.set_index("fid").dep) >= rec)).sum()),
              aircraft_outages=len(outages), airport_reductions=len(alt_ap),
              airport_reduction_hours=len(red), seconds=round(time.time() - t0, 1))
    json.dump(st, open(os.path.join(folder, "build_stats.json"), "w"), indent=1, default=str)
    return dict(stats=st, sums=sums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--instance", nargs="*")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--type-rule", default="wiki", choices=["wiki", "distance"])
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--reuse-gate", action="store_true", help="reuse stored solver results for unchanged instances")
    ap.add_argument("--verify", action="store_true", help="compare instance files with CHECKSUMS.json")
    ap.add_argument("--out", default=INST_DIR)
    a = ap.parse_args()
    if a.check_determinism:
        r1 = build_instance("US03", a.seed, os.path.join(US_DIR, "cache", "det1"))["sums"]
        r2 = build_instance("US03", a.seed, os.path.join(US_DIR, "cache", "det2"))["sums"]
        print("deterministic" if r1 == r2 else "NOT deterministic", len(r1), "files")
        sys.exit(0 if r1 == r2 else 1)
    if a.verify:
        cs = json.load(open(os.path.join(a.out, "CHECKSUMS.json")))
        bad = [n for n, v in cs.items() if checksums(os.path.join(a.out, n)) != v]
        print("checksums OK" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    if a.all:
        from . import download
        if download.ensure() != 0:
            sys.exit("raw data missing / checksum mismatch: run python -m arp.us2014.download")
    names = [n for n, sp in sc.specs().items() if not sp.extra] if a.all else (a.instance or [])
    allsums = {}
    cs_path = os.path.join(a.out, "CHECKSUMS.json")
    if os.path.exists(cs_path) and not a.all:
        allsums = json.load(open(cs_path))
    for n in names:
        r = build_instance(n, a.seed, a.out, a.type_rule)
        allsums[n] = r["sums"]
        s = r["stats"]
        print(f"{n}: flights {s['flights']} tails {s['tails']} itins {s['itineraries']} pax {s['pax']} "
              f"drop {s['repair_drop_rate']:.3%} {s['seconds']}s", flush=True)
    if names:
        os.makedirs(a.out, exist_ok=True)
        json.dump(allsums, open(cs_path, "w", newline="\n"), indent=1, sort_keys=True)
    if a.all:
        from . import summary, validate
        res = [validate.check(n, a.out) for n in names]
        prev = {}
        vp = os.path.join(a.out, "validation.json")
        if a.reuse_gate and os.path.exists(vp):
            prev = {x["name"]: x for x in json.load(open(vp)) if x.get("fh")}
        for r in res:       # FH gate: every scale-down instance (<= 2,500 flights), one solver process at a time
            if r["stats"]["flights"] <= 2500:
                sha = hashlib.sha256(json.dumps(allsums.get(r["name"]), sort_keys=True).encode()).hexdigest()
                old = prev.get(r["name"])
                if old and old["fh"].get("sums_sha") == sha:     # same instance files: reuse the solver result, redo the witness check
                    r["fh"] = validate.classify({k: v for k, v in old["fh"].items() if k not in ("gate", "pass_", "witness")}, r["name"], a.out)
                else:
                    r["fh"] = validate.fh_gate(r["name"], a.out)
                    r["fh"]["sums_sha"] = sha
                r["fh"]["sums_sha"] = sha
                r["ok"] = r["ok"] and r["fh"]["pass_"]
                print("FH gate", r["name"], {k: r["fh"][k] for k in ("gate", "status", "obj", "bound", "gap", "wall_s", "n_violations")}, flush=True)
        json.dump(res, open(os.path.join(a.out, "validation.json"), "w"), indent=1, default=str)
        summary.write(res, a.out)
        bad = [r["name"] for r in res if not r["ok"]]
        print("validation:", "all OK" if not bad else f"FAILED {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
