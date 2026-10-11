"""Time-space network for the path (column-generation) formulation.

Uses exactly the flight copies, eligibility rules and arc costs of ``arp.model.solve_window`` (full horizon,
``terminal='none'``): a column is one aircraft route = a chain of flight copies (f, dep).  Aircraft with the same
(kind, supply airport, ready time) form a class; tails with maintenance / outages are singleton classes.
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..costs import ItinInfo, flight_loads, spill
from ..instance import Instance
from ..model import BIG_SLACK, MAINT_MINUTE_PENALTY, Commodity, Params, State, build_commodities, departure_set

MAINT_HOUR_PENALTY = MAINT_MINUTE_PENALTY     # model.py: cost of one flown minute above the maintenance budget


@dataclass
class AircraftClass:
    cid: int
    kid: int
    ap: str
    ready: int
    tails: List[str]

    @property
    def n(self) -> int:
        return len(self.tails)


class CommGraph:
    """Arcs (flight copies) usable by one commodity, sorted by departure, with the static index structure
    used by the pricing DP (per-airport departure lists and successor positions)."""

    def __init__(self, net: "Network", comm: Commodity, copy_ids: List[int], kcost: Dict[int, float]):
        I = net.I
        self.comm = comm
        self.kid = comm.kid
        self.turn = comm.turn
        cp = sorted(copy_ids, key=lambda c: (net.cdep[c], c))
        self.n = len(cp)
        self.cid = cp
        self.dep = [net.cdep[c] for c in cp]
        self.arr = [net.carr[c] for c in cp]
        self.ready = [a + comm.turn for a in self.arr]
        self.kcost = [kcost[net.cflt[c]] for c in cp]
        aps = sorted({net.corig[c] for c in cp} | {net.cdest[c] for c in cp})
        self.ap_index = {a: i for i, a in enumerate(aps)}
        self.aps = aps
        self.o = [self.ap_index[net.corig[c]] for c in cp]
        self.d = [self.ap_index[net.cdest[c]] for c in cp]
        self.ap_arcs: List[List[int]] = [[] for _ in aps]
        self.pos = [0] * self.n
        for a in range(self.n):
            lst = self.ap_arcs[self.o[a]]
            self.pos[a] = len(lst)
            lst.append(a)
        self.ap_deps = [[self.dep[a] for a in lst] for lst in self.ap_arcs]
        self.m = [bisect.bisect_right(ds, I.end) for ds in self.ap_deps]    # number of pre-end departures
        self.succ = [bisect.bisect_left(self.ap_deps[self.d[a]], self.ready[a]) for a in range(self.n)]
        self.cid_np = np.array(cp, dtype=np.int64)
        self.kcost_np = np.array(self.kcost, dtype=float)
        self.arc_of_copy = {c: a for a, c in enumerate(cp)}
        self._events = None

    def events(self):
        """Time-ordered events for the label-setting pricer: (time, type, arc); add=0 < maint=1 < dep=2 < end=3."""
        if self._events is None:
            ev = []
            for a in range(self.n):
                ev.append((self.dep[a], 2, a))
                ev.append((self.ready[a], 0, a))
            self._events = ev
        return self._events


class Network:
    def __init__(self, I: Instance, S: State, p: Params, itins: List[ItinInfo], warm_plan: Optional[dict] = None,  # unused (R-3)
                 restrict_singletons: Optional[bool] = None):
        self.I, self.S, self.p = I, S, p
        alpha = I.weights[0]
        flights = list(I.modifiable)
        self.flights = flights
        fset = set(flights)
        horizon_end = max([I.flights[f].dep for f in flights] + [S.now]) + max(p.delay_grid) + 600
        rs = p.restrict_singletons if restrict_singletons is None else restrict_singletons
        self.comms = build_commodities(I, S, horizon_end, rs)
        loads = flight_loads(I)
        # R-3: station sets and departure times are exactly those of the flow model (arp.model); warm / seed plans no
        # longer widen them (a seed route leaving the set is truncated by driver.load_plan_columns)
        # passenger average cancellation cost per flight (spill pricing)
        legs_of: Dict[int, List[ItinInfo]] = defaultdict(list)
        for inf in itins:
            for f, _ in inf.it.legs:
                legs_of[f].append(inf)
        avg_cancel: Dict[int, float] = {}
        for f in flights:
            lst = legs_of.get(f, [])
            pax = sum(i.it.count for i in lst)
            avg_cancel[f] = (sum(i.it.count * i.cancel_cost_out for i in lst) / pax) if pax else 0.0
        self.forced_cancel = {f for f in flights if S.delays.get(f, 0) < 0}
        # copies --------------------------------------------------------------------------------
        self.cflt: List[int] = []
        self.cdep: List[int] = []
        self.carr: List[int] = []
        self.corig: List[str] = []
        self.cdest: List[str] = []
        self.copy_id: Dict[Tuple[int, int], int] = {}
        self.copies_of_flight: Dict[int, List[int]] = defaultdict(list)
        used_by: Dict[int, List[int]] = defaultdict(list)     # kid -> copy ids
        kcost: Dict[int, Dict[int, float]] = {k.kid: {} for k in self.comms}
        for f in flights:
            if f in self.forced_cancel:
                continue
            fo = I.flights[f]
            d = S.delays.get(f, 0)
            deps = departure_set(fo, d, S.now, p)
            for k in self.comms:
                if k.family != I.aircraft[fo.aircraft].family or fo.dur > k.range_min:
                    continue
                if k.stations is not None and fo.aircraft not in k.tails and (
                        fo.orig not in k.stations or fo.dest not in k.stations):
                    continue
                sp = spill(loads.get(f, (0, 0, 0)), k.kind[1])
                kcost[k.kid][f] = alpha * k.cost_h * fo.dur / 60.0 + sp * avg_cancel[f]
                for dep in deps:
                    arr = dep + fo.dur
                    ready = arr + k.turn
                    if any(dep < e and ready > s for s, e in k.blocks):
                        continue
                    cid = self.copy_id.get((f, dep))
                    if cid is None:
                        cid = len(self.cflt)
                        self.copy_id[(f, dep)] = cid
                        self.cflt.append(f)
                        self.cdep.append(dep)
                        self.carr.append(arr)
                        self.corig.append(fo.orig)
                        self.cdest.append(fo.dest)
                        self.copies_of_flight[f].append(cid)
                    used_by[k.kid].append(cid)
        self.ncopies = len(self.cflt)
        self.graphs: Dict[int, CommGraph] = {k.kid: CommGraph(self, k, used_by[k.kid], kcost[k.kid]) for k in self.comms}
        self.kcost = kcost
        # classes ---------------------------------------------------------------------------------
        self.classes: List[AircraftClass] = []
        for k in self.comms:
            groups: Dict[Tuple[str, int], List[str]] = defaultdict(list)
            for t in k.tails:
                ts = S.tails[t]
                groups[(ts.ap, ts.ready)].append(t)
            for (ap, ready), tails in sorted(groups.items()):
                self.classes.append(AircraftClass(len(self.classes), k.kid, ap, ready, sorted(tails)))
        self.comm_of = {k.kid: k for k in self.comms}
        self.copy_extra = None

    def set_copy_cost(self, extra) -> None:
        """Add a separable per-copy cost (passenger delay costs) to every arc cost; route_info includes it."""
        self.copy_extra = np.asarray(extra, dtype=float)
        for G in self.graphs.values():
            add = self.copy_extra[G.cid_np]
            G.kcost_np = G.kcost_np + add
            G.kcost = G.kcost_np.tolist()

    # ------------------------------------------------------------------------------------------
    def maint_active(self, comm: Commodity) -> bool:
        """Maintenance with a location + flight-hour requirement (starts inside the recovery period)."""
        return bool(comm.maint) and comm.maint[1] >= self.S.now

    def route_info(self, cls: AircraftClass, route: List[int], with_pen: bool = False):
        """Validate a route (list of copy ids) for a class; return (kind cost incl. maintenance terms, end airport
        or None).  Raises ValueError on infeasibility."""
        I, S = self.I, self.S
        comm = self.comm_of[cls.kid]
        G = self.graphs[cls.kid]
        elig = G.arc_of_copy
        loc, ready = cls.ap, cls.ready
        cost = 0.0
        for c in route:
            if c not in elig:
                raise ValueError("copy not usable by commodity")
            if self.corig[c] != loc:
                raise ValueError("continuity")
            if self.cdep[c] < ready:
                raise ValueError("turnaround")
            cost += self.kcost[cls.kid][self.cflt[c]] + (self.copy_extra[c] if self.copy_extra is not None else 0.0)
            loc, ready = self.cdest[c], self.carr[c] + G.turn
        m = comm.maint if self.maint_active(comm) else None
        pen = 0.0
        if m:
            # on the ground at the maintenance airport at s_m (soft, BIG_SLACK) and flight-hour budget (soft)
            loc2, ready2, flown = cls.ap, cls.ready, 0
            for c in route:
                if self.cdep[c] < m[1]:
                    loc2, ready2 = self.cdest[c], self.carr[c] + G.turn
                    flown += self.carr[c] - self.cdep[c]
            if not (loc2 == m[0] and ready2 <= m[1]):
                cost += BIG_SLACK
                pen += BIG_SLACK
            budget = max(0, m[3] - S.tails[comm.tails[0]].flown)
            cost += MAINT_HOUR_PENALTY * max(0, flown - budget)
            pen += MAINT_HOUR_PENALTY * max(0, flown - budget)
        # end-of-recovery position
        end = None
        last = None
        for c in route:
            if self.cdep[c] <= I.end:
                last = c
        if last is None:
            end = cls.ap
        elif self.carr[last] <= I.end:
            end = self.cdest[last]
        return (cost, end, pen) if with_pen else (cost, end)
