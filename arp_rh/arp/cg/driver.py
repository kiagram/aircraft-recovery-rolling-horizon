"""Column-generation driver: build network + master, generate columns, price-and-branch, convert to a Plan."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..costs import itinerary_info
from ..evaluate import evaluate
from ..instance import Instance
from ..model import COARSE_GRID, Params, initial_state
from ..policies import occ_rule, shared_extra_deps
from .master import Master
from .network import AircraftClass, Network
from .pricing import price_all, price_sequential


def make_params(grid: str = "fine") -> Params:
    p = Params()
    if grid == "coarse":
        p.delay_grid = COARSE_GRID
    return p


def build(I: Instance, p: Params, warm_plan: Optional[dict] = None, restrict: str = "auto", verbose: bool = False):
    t0 = time.time()
    S = initial_state(I, known_all=True)
    itins = itinerary_info(I)
    rs = (len(I.modifiable) > 1000) if restrict == "auto" else (restrict == "yes")
    net = Network(I, S, p, itins, warm_plan, restrict_singletons=rs)
    t1 = time.time()
    master = Master(net, itins, p, verbose=verbose)
    return net, master, dict(build_net=t1 - t0, build_master=time.time() - t1)


# ----------------------------------------------------------------------------------------------
def tail_routes(net: Network, plan: dict) -> Dict[str, List[int]]:
    """Per-tail route (copy ids) of the open flights of a plan {f: (dep, arr, tail) | 'C'}."""
    by_tail = defaultdict(list)
    for f in net.flights:
        v = plan.get(f)
        if isinstance(v, (tuple, list)):
            by_tail[v[2]].append((v[0], f))
    out = {}
    for t, lst in by_tail.items():
        lst.sort()
        out[t] = [net.copy_id[(f, dep)] for dep, f in lst]
    return out


def class_of_tail(net: Network) -> Dict[str, AircraftClass]:
    return {t: c for c in net.classes for t in c.tails}


def load_plan_columns(net: Network, master: Master, plan: dict, tag: str, ground: bool = True):
    """Add each tail's route as a column (plus a ground-only column per class when `ground`).  Returns the
    multiplicity map {pool index: count} of the plan and the list of tails whose route had to be dropped."""
    routes = tail_routes(net, plan)
    cof = class_of_tail(net)
    mult: Dict[int, float] = defaultdict(float)
    dropped = []
    for t, cls in cof.items():
        r = routes.get(t, [])
        while True:
            try:
                net.route_info(cls, r)
                break
            except (ValueError, KeyError):
                if r is routes.get(t) or t not in dropped:
                    dropped.append(t)
                r = r[:-1]          # truncate at the last valid prefix (flights after it are cancelled)
        master.add_column(cls, r, tag, keep=True)
        mult[master.pool_keys[(cls.cid, tuple(r))]] += 1.0
    if ground:
        for cls in net.classes:
            master.add_column(cls, [], "ground", keep=True)
    return dict(mult), dropped


def column_generation(net: Network, master: Master, time_limit: float = 600.0, tol: float = 1e-4, alpha: float = 0.8,
                      purge_thr: Optional[float] = 50.0, purge_min: int = 500, simplex_strategy: int = 1,
                      n_alt: int = 2, merge_sep: bool = True, sep_limit: Optional[int] = None, n_seq: int = 2,
                      seq_beta: float = 1.0, seed: int = 1, natural_center: bool = True, box: Optional[float] = None, box_s0: float = 1000.0,
                      true_every: int = 5, center0: Optional[np.ndarray] = None, tighten: bool = True, purge_age: int = 5, verbose: bool = True, log=None):
    """Column generation with dual smoothing (price at alpha*center + (1-alpha)*LP duals) and pricing also at the true
    LP duals (convergence test, Lagrangian bound).  Per class and round: the best route at each dual vector plus up to
    `n_alt` single-deviation alternatives of the smoothed-dual route.  Lazy connection rows are separated once per
    round (`merge_sep`: rows and columns enter the same LP re-solve) instead of to exhaustion.  Non-basic columns with
    large reduced cost are purged to keep the RMP small.  Converged = no negative-reduced-cost column at the true duals
    and no violated lazy row."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    import random as _random
    prng = _random.Random(seed)
    master.h.setOptionValue("simplex_strategy", simplex_strategy)
    if merge_sep:
        obj, rd = master.solve_once()
    else:
        obj, rd = master.solve_lp()
    if tighten and merge_sep:
        rd = master.tighten(master.last_rd_full, master.last_x)
    center = center0.copy() if center0 is not None else (master.natural_duals() if natural_center else rd.copy())
    t_price = t_add = t_purge = t_sep = 0.0
    hist = []
    best_lb = -np.inf
    it = 0
    n_purged = 0
    status = "converged"
    while True:
        tp = time.time()
        if box is None:
            center = alpha * center + (1 - alpha) * rd
        else:       # box-step: the pricing duals stay within a trust region around the previous ones
            delta = box * (np.abs(center) + box_s0)
            center = center + np.clip(rd - center, -delta, delta)
        res_t = price_all(net, master, rd)
        res_c = price_all(net, master, center, n_alt) if (alpha > 0 or box is not None) else []
        seq = []
        for _ in range(n_seq):
            seq += price_sequential(net, master, center, seq_beta, prng)
        t_price += time.time() - tp
        lagr = obj + sum(cls.n * min(0.0, rc) for cls, _, rc in res_t)
        best_lb = max(best_lb, lagr)
        neg_t = [(cls, route) for cls, route, rc in res_t if rc < -tol]
        ts = time.time()
        n_rows = master.separate(master.last_x, limit=sep_limit) if merge_sep else 0
        t_sep += time.time() - ts
        hist.append((it, time.time() - t0, obj, lagr, len(neg_t), len(master.active)))
        if verbose and (it % 10 == 0 or (not neg_t and not n_rows)):
            msg = (f"  it {it:4d}  t={time.time() - t0:7.1f}s  obj={obj:14.2f}  lagr={lagr:14.2f}  neg={len(neg_t):3d}"
                   f"  cols={len(master.active)}  rows+={n_rows}  purged={n_purged}")
            print(msg, flush=True)
            if log:
                log.write(msg + "\n")
                log.flush()
        if not neg_t and not n_rows:
            break
        ta = time.time()
        added = 0
        use_true = box is None or it % true_every == 0 or not any(rc < -tol for _, _, rc in res_c)
        for cls, route in (neg_t if use_true else []) + [(cls, route) for cls, route, rc in res_c if rc < -tol] + seq:
            added += master.add_column(cls, route, "cg", it=it)
        t_add += time.time() - ta
        if added == 0 and not n_rows:
            status = "no_new_columns"
            break
        it += 1
        if time.time() - t0 > time_limit:
            status = "time_limit"
            break
        master.h.setOptionValue("time_limit", master.h.getRunTime() + max(30.0, time_limit - (time.time() - t0)))
        try:
            if merge_sep:
                obj, rd = master.solve_once()
                if tighten:
                    rd = master.tighten(master.last_rd_full, master.last_x)
            else:
                obj, rd = master.solve_lp()
        except RuntimeError:           # RMP solve hit the remaining time budget: stop with the last valid duals
            status = "time_limit"
            break
        if purge_thr is not None and len(master.active) > purge_min:
            tq = time.time()
            n_purged += master.purge(purge_thr, it, purge_age)
            t_purge += time.time() - tq
    master.h.setOptionValue("time_limit", float("inf"))
    return dict(lp_obj=obj, lagr_lb=float(best_lb), iterations=it, status=status, time=time.time() - t0,
                pricing_time=t_price, lp_time=master.lp_time, add_time=t_add, purge_time=t_purge, sep_time=t_sep,
                cols=len(master.active), pool=len(master.lam), purged=n_purged, lazy_rows=master.n_lazy_rows, hist=hist)


# ----------------------------------------------------------------------------------------------
def solution_to_plan(net: Network, master: Master, lam_x: np.ndarray) -> dict:
    I, S = net.I, net.S
    plan = {}
    for f in I.flights:
        if f.is_transp:
            continue
        plan[f.idx] = S.executed[f.idx] if f.idx in S.executed else "C"
    per_class = defaultdict(list)
    for i, l in enumerate(master.active_columns()):
        k = int(round(lam_x[i]))
        for _ in range(k):
            per_class[l["cid"]].append(l["route"])
    for cls in net.classes:
        routes = per_class[cls.cid]
        assert len(routes) == cls.n, (cls, len(routes))
        for tail, route in zip(cls.tails, routes):
            for c in route:
                f = net.cflt[c]
                plan[f] = (net.cdep[c], net.carr[c], tail)
    return plan


def fh_lp_bound(I: Instance, p: Params, warm_plan: dict, time_limit: float = 900.0) -> Tuple[float, float]:
    """LP bound of the flight-copy model (arp.model) on the same grid, warm copies included.  Runs the model's own
    LP-guided restriction (LP solved first, then a 1-gap MIP that stops at the warm incumbent)."""
    from ..model import solve_window
    S = initial_state(I, known_all=True)
    itins = itinerary_info(I)
    q = replace(p, time_limit=time_limit, lp_restrict=True, rc_keep=50.0, mip_gap=1.0)
    t0 = time.time()
    res = solve_window(I, S, I.modifiable, itins, q, final=True, warm=warm_plan)
    # R-8 (F-09 addendum): only an LP solved to optimality is reported as the flow-LP value
    return (res.bound if res.cert.get("bound_source") == "LP optimal" else None), time.time() - t0


def add_clean_columns(net: Network, master: Master, rd: np.ndarray) -> int:
    """For every class with a maintenance requirement add the best route under duals `rd` that meets the location and
    flight-hour requirements *hard* (soft penalties made prohibitive).  Guarantees a penalty-free column per class
    whenever one exists."""
    from .pricing import price_labels
    pi_link, _ = master.duals(rd)
    n = 0
    for cls in net.classes:
        comm = net.comm_of[cls.kid]
        if not net.maint_active(comm):
            continue
        G = net.graphs[cls.kid]
        rc = (G.kcost_np + pi_link[G.cid_np]).tolist()
        cr = [master.pos_credit(rd, ap, comm.kind) for ap in G.aps]
        _, route = price_labels(net, G, rc, cr, cls, comm.maint, net.S.tails[comm.tails[0]].flown, big=1e9, hour_pen=1e6)
        if net.route_info(cls, route, with_pen=True)[2] > 0:
            continue                          # no penalty-free route exists for this class
        n += master.add_column(cls, route, "clean", keep=True)
    return n


def clean_start(net: Network, master: Master, mult: Dict[int, float]) -> Dict[int, float]:
    """Make a start penalty-free.  Each column that relies on a soft maintenance slack is replaced by a repair route of
    its class: the best *elementary* route (natural duals) that meets the maintenance requirement hard, avoids
    saturated hourly quotas, and avoids the flights of the other columns of the start; if no such route exists the
    clashing tails are truncated at their first clash (flights after it are cancelled)."""
    from .pricing import price_labels
    rd = master.natural_duals()
    pi_link, _ = master.duals(rd)
    out = defaultdict(float, mult)
    covered = defaultdict(int)
    usage = defaultdict(int)

    def hours(c):
        return ((net.corig[c], net.cdep[c] // 60 * 60, "d"), (net.cdest[c], net.carr[c] // 60 * 60, "a"))

    def book(route, k):
        for c in route:
            covered[net.cflt[c]] += k
            for key in hours(c):
                usage[key] += k

    for pid, k in mult.items():
        book(master.lam[pid]["route"], int(round(k)))

    def full(c):
        return any(key in master.cap_val and usage[key] + 1 > master.cap_val[key] for key in hours(c))

    def hard_route(cls, blocked):
        comm = net.comm_of[cls.kid]
        G = net.graphs[cls.kid]
        cr = [master.pos_credit(rd, ap, comm.kind) for ap in G.aps]
        ban = np.array([1e9 if (net.cflt[c] in blocked or full(c)) else 0.0 for c in G.cid])
        for _ in range(20):                          # make the route elementary (no flight twice)
            rc = (G.kcost_np + pi_link[G.cid_np] + ban).tolist()
            _, route = price_labels(net, G, rc, cr, cls, comm.maint, net.S.tails[comm.tails[0]].flown, big=1e9, hour_pen=1e6)
            seen = {}
            dup = []
            for c in route:
                f = net.cflt[c]
                if f in seen:
                    dup.append(f)
                seen.setdefault(f, c)
            if not dup:
                return route if net.route_info(cls, route, with_pen=True)[2] <= 0 else None
            for f in dup:
                for c in net.copies_of_flight[f]:
                    if c != seen[f] and c in G.arc_of_copy:
                        ban[G.arc_of_copy[c]] = 1e9
        return None

    for pid in [q for q in mult if master.lam[q]["pen"] > 0]:
        rec = master.lam[pid]
        k = int(round(mult[pid]))
        cls = net.classes[rec["cid"]]
        book(rec["route"], -k)
        own = {net.cflt[c] for c in rec["route"]}
        blocked = {f for f, n in covered.items() if n > 0}
        route = hard_route(cls, blocked)
        clash = set()
        if route is None:                            # allow clashes with other tails, truncate those tails
            route = hard_route(cls, set())
            if route is not None:
                clash = {net.cflt[c] for c in route} & blocked
        if route is None:
            book(rec["route"], k)
            continue                                 # cannot be repaired
        for pid2 in [q for q in list(out) if out[q] > 0 and q != pid]:
            r2 = master.lam[pid2]
            cut = next((i for i, c in enumerate(r2["route"]) if net.cflt[c] in clash), None)
            if cut is None:
                continue
            cls2 = net.classes[r2["cid"]]
            trunc = r2["route"][:cut]
            master.add_column(cls2, trunc, "repair", keep=True)
            np2 = master.pool_keys[(cls2.cid, tuple(trunc))]
            k2 = out[pid2]
            book(r2["route"], -int(round(k2)))
            book(trunc, int(round(k2)))
            out[pid2] -= k2
            out[np2] += k2
        master.add_column(cls, route, "repair", keep=True)
        new_pid = master.pool_keys[(cls.cid, tuple(route))]
        book(route, k)
        out[pid] -= mult[pid]
        out[new_pid] += mult[pid]
    return {q: v for q, v in out.items() if v > 0}


def read_plan(path: str) -> dict:
    """Read-only: plan {f: (dep, arr, tail) | 'C'} from a results JSON."""
    import json
    d = json.load(open(path))
    return {int(k): (tuple(v) if v != "C" else "C") for k, v in d["plan"].items()}


def run(I: Instance, grid: str = "fine", cg_tl: float = 600.0, mip_tl: float = 300.0, mip_gap: float = 1e-4,
        restrict: str = "auto", with_fh_lp: bool = True, verbose: bool = True, seeds: Optional[List[str]] = None,
        log_path: Optional[str] = None, cg_kwargs: Optional[dict] = None, rmp_lp: str = "simplex") -> dict:
    from ..policies import plan_to_warm
    T0 = time.time()
    p = make_params(grid)
    # same singleton-neighbourhood rule as experiments.py (restricted only for > 1000 flights)
    p.restrict_singletons = (len(I.modifiable) > 1000) if restrict == "auto" else (restrict == "yes")
    occ = occ_rule(I)
    warm = plan_to_warm(occ.plan)
    p.extra_deps = shared_extra_deps(I, occ.plan)      # R-3: same departure set as the flow model
    seed_plans = [read_plan(sp) for sp in (seeds or [])]
    net, master, tb = build(I, p, [warm] + seed_plans if seed_plans else warm, "yes" if p.restrict_singletons else "no")
    log = open(log_path, "w") if log_path else None
    def say(msg):
        print(msg, flush=True)
        if log:
            log.write(msg + "\n")
            log.flush()
    say(f"network: {net.ncopies} copies, {len(net.classes)} classes, {len(net.comms)} commodities; master "
        f"{master.n_rows} rows x {master.n_static} static cols; build {tb}")
    starts = {}
    mult, dropped = load_plan_columns(net, master, {f: v for f, v in occ.plan.items()}, "occ")
    starts["occ"] = mult
    out_dropped = {"occ": dropped}
    for name, sp in zip(seeds or [], seed_plans):
        m2, d2 = load_plan_columns(net, master, sp, "seed", ground=False)
        starts[name] = m2
        out_dropped[name] = d2
    n_clean = add_clean_columns(net, master, master.natural_duals())
    say(f"clean maintenance columns added: {n_clean}")
    start_obj = {}
    for name, mu in starts.items():
        start_obj[name], _ = master.fixed_lp(mu)
        say(f"start {name}: RMP objective {start_obj[name]:.2f}, tails truncated {len(out_dropped[name])}")
    master.set_rmp_lp(rmp_lp)
    cg = column_generation(net, master, cg_tl, verbose=verbose, log=log, **(cg_kwargs or {}))
    master.set_rmp_lp("simplex")
    out_rmp_lp = rmp_lp
    out = dict(instance=I.name, grid=grid, build=tb, cg={k: v for k, v in cg.items() if k != "hist"},
               hist=cg["hist"], n_copies=net.ncopies, n_classes=len(net.classes), master_rows=master.n_rows,
               master_static_cols=master.n_static, start_obj=start_obj,
               seeds=seeds or [], rmp_lp=out_rmp_lp, start_truncated={k: len(v) for k, v in out_dropped.items()})
    say(f"CG {cg['status']}: LP={cg['lp_obj']:.2f} Lagrangian LB={cg['lagr_lb']:.2f} iterations={cg['iterations']} "
        f"time={cg['time']:.0f}s")
    # price-and-branch ------------------------------------------------------------------------------
    clean = {k: clean_start(net, master, mu) for k, mu in starts.items()}
    clean_obj = {}
    for k, mu in clean.items():
        try:
            clean_obj[k] = master.fixed_lp(mu)[0]
        except RuntimeError:
            say(f"start {k}: no feasible penalty-free repair")
    if clean_obj:
        best = min(clean_obj, key=clean_obj.get)
        inc_obj, inc_x = master.fixed_lp(clean[best])
        left_start_pen = any(master.lam[pid]["pen"] > 0 for pid in clean[best])
    else:
        best, inc_obj, inc_x, left_start_pen = None, None, None, True
    left_pen = left_start_pen
    n_forb = master.forbid_penalised()
    say(f"MIP start {best}: objective {inc_obj}, penalty-free start={not left_pen}; penalised columns fixed to 0: {n_forb}")
    out["mip_start"], out["mip_start_obj"], out["start_obj_clean"] = best, inc_obj, clean_obj
    out["mip_start_penalty_free"] = not left_pen
    t_main = mip_tl if clean_obj and not left_pen else 0.67 * mip_tl
    x, ub, db, st, mt = master.solve_mip(t_main, mip_gap, start=(None if left_pen else inc_x))
    out["mip_stage"] = 1
    if x is None:          # no penalty-free integer solution in the pool: re-allow the soft slack, start from the repaired start
        say("stage 1 (penalty-free) found no solution; stage 2 re-allows soft maintenance slack")
        master.unforbid()
        x, ub, db, st, mt2 = master.solve_mip(max(1.0, mip_tl - mt), mip_gap, start=inc_x)   # total MIP wall time stays <= mip_tl
        mt += mt2
        out["mip_stage"] = 2
    out["mip"] = dict(obj=ub, dual_bound=db, status=st, time=mt)
    lp = cg["lp_obj"]
    out["lp_bound"] = lp
    out["lb_valid"] = cg["lagr_lb"] if cg["status"] != "converged" else lp
    out["ub"] = ub if x is not None else None
    out["gap"] = (ub - out["lb_valid"]) / abs(ub) if x is not None else None
    say(f"MIP {st}: UB={ub} dual bound over pool={db} time={mt:.0f}s")
    if x is not None:
        plan = solution_to_plan(net, master, x[master.n_static:])
        e = evaluate(I, plan)
        out["eval"] = {k: v for k, v in e.items()}
        out["plan"] = {str(k): v for k, v in plan.items()}
    if with_fh_lp:
        b, t = fh_lp_bound(I, p, warm)
        out["fh_lp_bound"], out["fh_lp_time"] = b, t
    out["runtime"] = time.time() - T0
    return out
