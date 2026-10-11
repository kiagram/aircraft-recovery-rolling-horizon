"""Independent solution checker and cost evaluator (does not reuse the MIP)."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union

from .costs import flight_loads, itinerary_info, spill
from .instance import Instance
from .model import BIG_SLACK, MAINT_MINUTE_PENALTY, MIN_CONNECT, capacity

Plan = Dict[int, Union[str, Tuple[int, int, str]]]   # f -> 'C' or (dep, arr, tail)


def evaluate(I: Instance, plan: Plan, tol_hours_before_start: bool = True, revelation=None) -> Dict[str, float]:
    """Decision D5 (audit Stage 2, R-6), one rule for every plan of a dynamic scenario: a flight that departs before an
    airport-capacity reduction or an aircraft outage is announced is exempt from it (as airborne flights are exempt
    from ground delay programmes); it still counts against the nominal quota.  `revelation` gives the announcement
    times.  Without it (static ROADEF instances) nothing is exempt: every movement counts (ROADEF rule)."""
    alpha, beta, gamma = I.weights
    viol = Counter()
    NO = -10**9
    if revelation is not None and not getattr(revelation, "exempt", True):
        revelation = None           # static revelation: no exemption (ROADEF rule)
    ann_ap = (lambda ap, k: revelation.airport.get((ap, k), I.start)) if revelation is not None else (lambda ap, k: NO)
    ann_ac = (lambda t, k: revelation.aircraft.get((t, k), I.start)) if revelation is not None else (lambda t, k: NO)
    # ---------------- per-tail feasibility
    by_tail = defaultdict(list)
    for f in I.flights:
        if f.is_transp:
            continue
        v = plan.get(f.idx)
        if v is None:
            viol["missing_decision"] += 1
            continue
        if v == "C":
            if f.dep < I.start and I.flight_disr.get(f.idx, 0) >= 0:
                viol["cancel_fixed_flight"] += 1
            continue
        dep, arr, tail = v
        by_tail[tail].append((dep, arr, f))
        earliest = f.dep + max(I.flight_disr.get(f.idx, 0), 0)
        if f.dep < I.start and (dep != earliest or tail != f.aircraft):
            viol["modified_fixed_flight"] += 1
        if dep < earliest:
            viol["early_departure"] += 1
        if I.flight_disr.get(f.idx, 0) < 0 and f.dep >= I.start:
            viol["flew_cancelled_flight"] += 1
        a = I.aircraft[tail]
        if a.family != I.aircraft[f.aircraft].family:
            viol["family_swap"] += 1
        if f.dur > a.range_min:
            viol["range"] += 1
        if arr - dep != f.dur:
            viol["duration"] += 1
    flown_before_maint = Counter()
    for t, lst in by_tail.items():
        lst.sort(key=lambda x: x[0])
        a = I.aircraft[t]
        loc, ready, prev_f = a.orig, -10**9, None
        for dep, arr, f in lst:
            if f.orig != loc:
                viol["continuity"] += 1
            need = a.transit if (prev_f is not None and f.prev == prev_f.idx) else a.turn
            if dep < ready + (need if prev_f is not None else 0) - 1e-9 and prev_f is not None:
                viol["turnaround"] += 1
            for k, (s, e) in enumerate(I.ac_unavail.get(t, ())):
                if dep < e and arr > s and (dep if f.dep >= I.start else NO + 1) >= ann_ac(t, k):
                    viol["unavailability"] += 1
            if a.maint:
                ap_m, s_m, e_m, rem = a.maint
                if dep < e_m and arr > s_m:
                    viol["maint_overlap"] += 1
                if dep < s_m and dep >= I.start:
                    flown_before_maint[t] += f.dur
            loc, ready, prev_f = f.dest, arr, f
    maint_excess_minutes = 0
    for t, a in I.aircraft.items():
        if a.is_transp or not a.maint:
            continue
        ap_m, s_m, e_m, rem = a.maint
        if s_m < I.start:
            continue
        before = [x for x in by_tail.get(t, []) if x[0] < s_m]
        loc = max(before, key=lambda x: x[0])[2].dest if before else a.orig
        if loc != ap_m:
            viol["maint_location"] += 1
        if flown_before_maint[t] > rem:
            viol["maint_hours"] += 1
            maint_excess_minutes += flown_before_maint[t] - rem

    # ---------------- airport capacity (hours inside the recovery period)
    use = Counter()
    for f in I.flights:
        if f.is_transp:
            continue
        v = plan.get(f.idx)
        if v is None or v == "C":
            continue
        dep, arr, _ = v
        use[(f.orig, dep // 60 * 60, "d")] += 1
        use[(f.dest, arr // 60 * 60, "a")] += 1
    cap_excess = 0
    exempt_used = 0
    deps_in = defaultdict(list)
    for f in I.flights:
        v = plan.get(f.idx)
        if f.is_transp or v is None or v == "C":
            continue
        d0 = v[0] if f.dep >= I.start else NO + 1     # fixed pre-T0 flight: exempt from announced reductions (D5)
        deps_in[(f.orig, v[0] // 60 * 60, "d")].append(d0)
        deps_in[(f.dest, v[1] // 60 * 60, "a")].append(d0)
    for (ap, h, kd), u in use.items():
        if tol_hours_before_start and h + 60 <= I.start:
            continue
        qd, qa = capacity(I, {}, ap, h)
        qn = qd if kd == "d" else qa
        hits = [k for k, (s, e, d, a) in enumerate(I.alt_cap.get(ap, ())) if s <= h < e]
        c = min(I.alt_cap[ap][k][2 if kd == "d" else 3] for k in hits) if hits else qn
        excess = max(0, u - max(qn, c))         # nominal quota (or a raised alternative quota): every movement
        if c < qn:
            tau = min(ann_ap(ap, k) for k in hits)                          # announcement of the reduction(s)
            nonexempt = sum(1 for dep in deps_in[(ap, h, kd)] if dep >= tau)
            excess = max(excess, nonexempt - c)                             # reduced quota: non-exempt movements
            exempt_used += max(0, min(u - c, u - nonexempt))
        cap_excess += excess
    if cap_excess:
        viol["capacity_excess"] = cap_excess

    # ---------------- costs
    op = 0.0
    op_orig = 0.0
    n_cancel = n_del15 = 0
    tot_delay = prim_delay = 0
    swaps = 0
    for f in I.flights:
        if f.is_transp or f.dep < I.start:
            continue
        op_orig += alpha * I.aircraft[f.aircraft].cost_h * f.dur / 60
        v = plan.get(f.idx)
        if v == "C" or v is None:
            n_cancel += 1
            continue
        dep, arr, tail = v
        op += alpha * I.aircraft[tail].cost_h * f.dur / 60
        d = dep - f.dep
        tot_delay += d
        prim_delay += min(d, max(I.flight_disr.get(f.idx, 0), 0))
        n_del15 += d >= 15
        swaps += tail != f.aircraft

    infos = itinerary_info(I)
    pax_cost = pax_cancel_cost = pax_delay_cost = 0.0
    pax_broken = pax_total = 0
    pax_delay_min = 0.0
    for inf in infos:
        legs = [f for f, _ in inf.it.legs]
        if all(I.flights[f].dep < I.start for f in legs):
            continue   # itinerary fully before the recovery period
        pax_total += inf.it.count
        broken = any(plan.get(f) in ("C", None) for f in legs)
        if not broken:
            for l1, l2 in zip(legs, legs[1:]):
                f1, f2 = I.flights[l1], I.flights[l2]
                mct = min(MIN_CONNECT, f2.dep - f1.arr)
                if plan[l2][0] < plan[l1][1] + mct:
                    broken = True
                    break
        if broken:
            started = I.flights[legs[0]].dep < I.start
            c = inf.cancel_cost_in if (inf.it.kind == "R" or started) else inf.cancel_cost_out
            pax_cancel_cost += inf.it.count * c
            pax_broken += inf.it.count
        else:
            dl = plan[legs[-1]][1] - inf.sched_arr
            pax_delay_cost += inf.it.count * inf.delay_cost(dl)
            pax_delay_min += inf.it.count * max(dl, 0)
    # seat spill
    loads = flight_loads(I)
    legs_of = defaultdict(list)
    for inf in infos:
        for f, _ in inf.it.legs:
            legs_of[f].append(inf)
    spill_cost = 0.0
    spilled = 0
    for f in I.flights:
        v = plan.get(f.idx)
        if f.is_transp or v in ("C", None) or f.dep < I.start:
            continue
        sp = spill(loads.get(f.idx, (0, 0, 0)), I.aircraft[v[2]].config)
        if sp:
            lst = legs_of[f.idx]
            pax = sum(i.it.count for i in lst)
            spill_cost += sp * sum(i.it.count * i.cancel_cost_out for i in lst) / pax
            spilled += sp
    pax_cost = pax_cancel_cost + pax_delay_cost + spill_cost

    # ---------------- end-of-recovery positions
    present = Counter()
    for t, a in I.aircraft.items():
        if a.is_transp:
            continue
        lst = [x for x in by_tail.get(t, []) if x[0] <= I.end]
        if not lst:
            present[(a.orig, a.kind)] += 1
            continue
        dep, arr, f = max(lst, key=lambda x: x[0])
        if arr <= I.end:
            present[(f.dest, a.kind)] += 1
    fam_of = {a.model: a.family for a in I.aircraft.values()}
    nb_fam = nb_mod = nb_cfg = 0
    for ap, reqs in I.positions.items():
        R = Counter({(m, c): n for m, c, n in reqs})
        F = Counter({k: v for (a2, k), v in present.items() if a2 == ap})
        u_cfg = sum(max(0, R[k] - F[k]) for k in R)
        Rm, Fm, Rf, Ff = Counter(), Counter(), Counter(), Counter()
        for (m, c), n in R.items():
            Rm[m] += n
            Rf[fam_of.get(m, m)] += n
        for (m, c), n in F.items():
            Fm[m] += n
            Ff[fam_of.get(m, m)] += n
        u_mod = sum(max(0, Rm[m] - Fm[m]) for m in Rm)
        u_fam = sum(max(0, Rf[m] - Ff[m]) for m in Rf)
        nb_fam += u_fam
        nb_mod += u_mod - u_fam
        nb_cfg += u_cfg - u_mod
    cf, cm, cc = I.pos_pen
    pos_cost = gamma * (nb_fam * cf + nb_mod * cm + nb_cfg * cc)

    # decision D2 (audit Stage 2, R-4): maintenance requirements are priced soft constraints with the same prices as
    # in the optimisation models; the price is part of the total.  Every other violation is a hard-constraint breach:
    # the plan is reported infeasible (`feasible` False) and its total is not comparable with feasible plans.
    maint_penalty = BIG_SLACK * viol.get("maint_location", 0) + MAINT_MINUTE_PENALTY * maint_excess_minutes
    hard = {k: v for k, v in viol.items() if k not in ("maint_location", "maint_hours")}
    cost_ops = op + pax_cost + pos_cost
    total = cost_ops + maint_penalty
    n_mod = sum(1 for f in I.flights if not f.is_transp and f.dep >= I.start)
    out = dict(total=total, cost_excl_maint_penalty=cost_ops, maint_penalty=maint_penalty,
               maint_excess_minutes=maint_excess_minutes, feasible=not hard, n_hard_violations=sum(hard.values()), op_cost=op, op_delta=op - op_orig, pax_cost=pax_cost, pax_cancel_cost=pax_cancel_cost,
               pax_delay_cost=pax_delay_cost, spill_cost=spill_cost, pos_cost=pos_cost,
               n_flights=n_mod, n_cancel=n_cancel, n_delayed15=n_del15, total_delay=tot_delay,
               primary_delay=prim_delay, propagated_delay=tot_delay - prim_delay, swaps=swaps,
               pax_total=pax_total, pax_broken=pax_broken, pax_delay_min=pax_delay_min, spilled=spilled,
               pos_missing=nb_fam + nb_mod + nb_cfg, exempt_airborne=exempt_used, n_violations=sum(viol.values()))
    out["violations"] = dict(viol)
    return out
