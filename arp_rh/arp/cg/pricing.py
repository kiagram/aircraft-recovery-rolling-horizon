"""Pricing: resource-constrained shortest path per aircraft class on the flight-copy time-space network.

Arc reduced cost = kind cost of the flight + dual of the copy-linking row.  Route-level terms:
  * end-of-recovery position credit  -(sum of duals of the position rows matched by the end airport),
  * convexity dual of the class.
Without maintenance requirements the network is a DAG: one backward pass per commodity (shared by all its classes).
With a maintenance requirement (location at s_m and flight-minute budget) a forward label-setting algorithm with
flown minutes as the resource and Pareto dominance is used.

End position rule (identical to model.py / evaluate.py): the tail is "present" at the destination of its last flight
with dep <= I.end if that flight has landed by I.end, or at its supply airport if it flies nothing up to I.end.
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

from ..model import BIG_SLACK
from .network import MAINT_HOUR_PENALTY, AircraftClass, CommGraph, Network

INF = 1.0e30


class DagTables:
    """Result of one backward pass; `best(...)` reads the optimal route for any supply node."""

    def __init__(self, G: CommGraph, rc: List[float], cr: List[float], t_end: int):
        n = G.n
        dep, arr, o, d, pos, succ, m = G.dep, G.arr, G.o, G.d, G.pos, G.succ, G.m
        Spre = [[INF] * (len(l) + 1) for l in G.ap_arcs]
        Apre = [[-1] * (len(l) + 1) for l in G.ap_arcs]
        Spost = [[INF] * (len(l) + 1) for l in G.ap_arcs]
        Apost = [[-1] * (len(l) + 1) for l in G.ap_arcs]
        nxt = [-1] * n
        tot_l = [0.0] * n
        for a in range(n - 1, -1, -1):
            dd = d[a]
            j = succ[a]
            md = m[dd]
            pj = j if j > md else md
            q = Spost[dd][pj]
            qa = Apost[dd][pj]
            if q > 0.0:
                q = 0.0
                qa = -1
            opt1 = (cr[dd] if arr[a] <= t_end else 0.0) + q
            opt2 = Spre[dd][j]
            if opt2 < opt1:
                v = opt2
                nxt[a] = Apre[dd][j]
            else:
                v = opt1
                nxt[a] = qa
            tot = rc[a] + v
            tot_l[a] = tot
            oo = o[a]
            p = pos[a]
            if dep[a] <= t_end:
                S, A = Spre[oo], Apre[oo]
            else:
                S, A = Spost[oo], Apost[oo]
            if tot <= S[p + 1]:
                S[p] = tot
                A[p] = a
            else:
                S[p] = S[p + 1]
                A[p] = A[p + 1]
        self.G, self.cr, self.t_end = G, cr, t_end
        self.rc, self.tot = rc, tot_l
        self.Spre, self.Apre, self.Spost, self.Apost, self.nxt = Spre, Apre, Spost, Apost, nxt

    def best(self, ap: str, ready: int) -> Tuple[float, List[int]]:
        G = self.G
        ai = G.ap_index.get(ap)
        cr0 = 0.0
        if ai is None:
            return 0.0, []
        if ready <= self.t_end:
            cr0 = self.cr[ai]
        j = bisect.bisect_left(G.ap_deps[ai], ready)
        pj = max(j, G.m[ai])
        q = self.Spost[ai][pj]
        qa = self.Apost[ai][pj]
        if q > 0.0:
            q, qa = 0.0, -1
        opt1 = cr0 + q
        opt2 = self.Spre[ai][j]
        if opt2 < opt1:
            v, a = opt2, self.Apre[ai][j]
        else:
            v, a = opt1, qa
        route = []
        while a >= 0:
            route.append(G.cid[a])
            a = self.nxt[a]
        return v, route


def _alternatives(self, ap, ready, route, M):
    """Single-deviation alternatives of the best route: at each node of the route take another pre-end departure
    b and then continue optimally.  Returns up to M (estimated value, route) pairs, best first."""
    G = self.G
    ai = G.ap_index.get(ap)
    if ai is None or M <= 0:
        return []
    arcs = [G.arc_of_copy[c] for c in route]
    cands = []
    prefix = 0.0
    state_ai, state_ready = ai, ready
    for k in range(len(arcs) + 1):
        chosen = arcs[k] if k < len(arcs) else -1
        j = bisect.bisect_left(G.ap_deps[state_ai], state_ready)
        lst = G.ap_arcs[state_ai]
        for pos in range(j, G.m[state_ai]):
            b = lst[pos]
            if b != chosen:
                cands.append((prefix + self.tot[b], k, b))
        if k < len(arcs):
            a = arcs[k]
            if G.dep[a] > self.t_end:
                break
            prefix += self.rc[a]
            state_ai, state_ready = G.d[a], G.ready[a]
    cands.sort()
    out = []
    for val, k, b in cands[:M]:
        r = [G.cid[a] for a in arcs[:k]]
        a = b
        while a >= 0:
            r.append(G.cid[a])
            a = self.nxt[a]
        out.append((val, r))
    return out


DagTables.alternatives = _alternatives


def price_labels(net: Network, G: CommGraph, rc: List[float], cr: List[float], cls: AircraftClass,
                 maint: Optional[tuple], flown0: int = 0, big: float = BIG_SLACK,
                 hour_pen: float = MAINT_HOUR_PENALTY) -> Tuple[float, List[int]]:
    """Forward label-setting (cost, flown minutes) with Pareto dominance for one singleton class.

    `maint` = (airport, start, end, remaining flight minutes) or None.  The location requirement at s_m costs
    big when violated; flown minutes (dep < s_m) beyond the budget cost hour_pen each.
    """
    t_end = net.I.end
    n = G.n
    s_m = maint[1] if maint else None
    ap_m = G.ap_index.get(maint[0]) if maint else None
    budget = max(0, maint[3] - flown0) if maint else 0
    ai0 = G.ap_index.get(cls.ap)
    if ai0 is None:
        return 0.0, []
    lab_cost: List[float] = []
    lab_r: List[int] = []
    lab_par: List[int] = []
    lab_arc: List[int] = []

    def new(cost, r, par, arc):
        lab_cost.append(cost)
        lab_r.append(r)
        lab_par.append(par)
        lab_arc.append(arc)
        return len(lab_cost) - 1

    def pareto(ids):
        ids = sorted(ids, key=lambda i: (lab_r[i], lab_cost[i]))
        out = []
        best = INF
        for i in ids:
            if lab_cost[i] < best - 1e-9:
                out.append(i)
                best = lab_cost[i]
        return out

    pools: List[List[int]] = [[] for _ in G.aps]
    pend: Dict[int, List[int]] = {}
    ev = list(G.events())
    ev.append((cls.ready, 0, -1))                  # supply becomes available
    if maint:
        ev.append((s_m, 1, -1))
    ev.append((t_end, 3, -1))
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    root = new(0.0, 0, -1, -1)
    for time_, typ, a in ev:
        if typ == 0:
            if a < 0:
                c0 = 0.0
                if maint and cls.ready > s_m:
                    c0 += big
                if cls.ready > t_end:
                    c0 += cr[ai0]
                lab_cost[root] = c0
                pools[ai0] = pareto(pools[ai0] + [root])
            else:
                ids = pend.pop(a, None)
                if not ids:
                    continue
                dd = G.d[a]
                add = cr[dd] if (G.ready[a] > t_end and G.arr[a] <= t_end) else 0.0
                for i in ids:
                    lab_cost[i] += add
                    if maint and time_ >= s_m:
                        lab_r[i] = 0
                pools[dd] = pareto(pools[dd] + ids)
        elif typ == 1:                              # maintenance start: location check, resource no longer needed
            for ai in range(len(pools)):
                for i in pools[ai]:
                    if ai != ap_m:
                        lab_cost[i] += big
                    lab_r[i] = 0
                pools[ai] = pareto(pools[ai])
        elif typ == 2:
            pool = pools[G.o[a]]
            if not pool:
                continue
            ids = []
            dur = G.arr[a] - G.dep[a]
            track = maint is not None and G.dep[a] < s_m
            for i in pool:
                c = lab_cost[i] + rc[a]
                r = lab_r[i]
                if track:
                    over = max(0, r + dur - budget) - max(0, r - budget)
                    c += hour_pen * over
                    r = min(budget, r + dur)
                ids.append(new(c, r, i, a))
            pend[a] = pareto(ids)
        else:                                       # end of recovery: position credit for tails on the ground
            for ai in range(len(pools)):
                for i in pools[ai]:
                    lab_cost[i] += cr[ai]
    best_i, best_c = -1, INF
    for pool in pools:
        for i in pool:
            if lab_cost[i] < best_c:
                best_i, best_c = i, lab_cost[i]
    route = []
    i = best_i
    while i >= 0 and lab_arc[i] >= 0:
        route.append(G.cid[lab_arc[i]])
        i = lab_par[i]
    route.reverse()
    return best_c, route


def price_all(net: Network, master, rd, n_alt: int = 0):
    """One pricing round.  Returns list of (class, route, reduced cost) and the Lagrangian term."""
    import numpy as np
    pi_link, pi_conv = master.duals(rd)
    t_end = net.I.end
    out = []
    credit = {}
    classes_of: Dict[int, List[AircraftClass]] = {}
    for cls in net.classes:
        classes_of.setdefault(cls.kid, []).append(cls)
    for kid, classes in classes_of.items():
        G = net.graphs[kid]
        comm = net.comm_of[kid]
        rc = (G.kcost_np + pi_link[G.cid_np]).tolist()
        cr = []
        for ap in G.aps:
            key = (ap, comm.kind)
            if key not in credit:
                credit[key] = master.pos_credit(rd, ap, comm.kind)
            cr.append(credit[key])
        if net.maint_active(comm):
            for cls in classes:
                v, route = price_labels(net, G, rc, cr, cls, comm.maint, net.S.tails[comm.tails[0]].flown)
                out.append((cls, route, v - pi_conv[cls.cid]))
        else:
            tab = DagTables(G, rc, cr, t_end)
            for cls in classes:
                v, route = tab.best(cls.ap, cls.ready)
                out.append((cls, route, v - pi_conv[cls.cid]))
                if n_alt:
                    for va, ra in tab.alternatives(cls.ap, cls.ready, route, n_alt):
                        out.append((cls, ra, va - pi_conv[cls.cid]))
    return out


def price_sequential(net: Network, master, rd, beta: float, rng):
    """Greedy sequential pricing: classes in random order; flights taken by an earlier class lose (a fraction `beta`
    of) their dual bonus for the later ones, so the routes of one round are largely disjoint (a near-feasible
    schedule).  Returns [(class, route)] (not filtered by reduced cost)."""
    import numpy as np
    pi_link, pi_conv = master.duals(rd)
    bonus = np.maximum(0.0, -pi_link)
    pen = np.zeros(len(pi_link))
    t_end = net.I.end
    credit = {}
    order = list(net.classes)
    rng.shuffle(order)
    out = []
    for cls in order:
        G = net.graphs[cls.kid]
        comm = net.comm_of[cls.kid]
        rc = (G.kcost_np + pi_link[G.cid_np] + pen[G.cid_np]).tolist()
        cr = []
        for ap in G.aps:
            key = (ap, comm.kind)
            if key not in credit:
                credit[key] = master.pos_credit(rd, ap, comm.kind)
            cr.append(credit[key])
        if net.maint_active(comm):
            _, route = price_labels(net, G, rc, cr, cls, comm.maint, net.S.tails[comm.tails[0]].flown)
        else:
            _, route = DagTables(G, rc, cr, t_end).best(cls.ap, cls.ready)
        out.append((cls, route))
        for c in route:
            cs = net.copies_of_flight[net.cflt[c]]
            pen[cs] += beta * bonus[cs]
    return out
