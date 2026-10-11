"""Flight-copy / fleet time-space network MIP for aircraft recovery with
hourly airport departure/arrival quotas (ROADEF 2009 rules).

One call builds and solves the recovery problem for a *window* of open
flights given the current operational *state* (aircraft positions and ready
times, flights already executed, capacity already consumed, information known
so far).  The same builder is used by the full-horizon model (one window
covering the whole recovery period) and by every rolling-horizon policy.
"""
from __future__ import annotations

import bisect
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import highspy
import numpy as np

from .costs import ItinInfo, flight_loads, spill
from .instance import Instance

MIN_CONNECT = 30
BIG_SLACK = 1.0e5            # price of a missed maintenance location (decision D2: priced soft constraint, R-4)
MAINT_MINUTE_PENALTY = 100.0  # price per flown minute above the maintenance budget (same in evaluator and path model)


# ----------------------------------------------------------------------------
@dataclass
class TailState:
    ap: str
    ready: int
    flown: int = 0          # flight minutes since recovery start (maintenance budget)


@dataclass
class State:
    now: int
    tails: Dict[str, TailState]
    executed: Dict[int, Tuple[int, int, str]] = field(default_factory=dict)  # f -> dep, arr, tail
    cancelled: Set[int] = field(default_factory=set)
    cap_used: Counter = field(default_factory=Counter)                         # (ap, hour, 'd'/'a')
    delays: Dict[int, int] = field(default_factory=dict)                       # known flight disruptions
    unavail: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)    # known aircraft outages
    alt_cap: Dict[str, List[Tuple[int, int, int, int]]] = field(default_factory=dict)  # known reductions
    # announcement times (decision D5, R-6): a flight departing before the announcement of a reduction / an outage is
    # exempt from it.  Missing keys mean "no exemption" (static ROADEF instances: every movement counts, ROADEF rule).
    cap_ann: Dict[Tuple[str, int, int], int] = field(default_factory=dict)      # (airport, start, end) -> time
    unavail_ann: Dict[Tuple[str, int, int], int] = field(default_factory=dict)  # (tail, start, end) -> time

    def copy(self) -> "State":
        return State(self.now, {k: TailState(v.ap, v.ready, v.flown) for k, v in self.tails.items()},
                     dict(self.executed), set(self.cancelled), Counter(self.cap_used), dict(self.delays),
                     {k: list(v) for k, v in self.unavail.items()},
                     {k: list(v) for k, v in self.alt_cap.items()}, dict(self.cap_ann), dict(self.unavail_ann))


def capacity(I: Instance, alt: Dict[str, list], ap: str, h: int) -> Tuple[int, int]:
    hits = [(d, a) for s, e, d, a in alt.get(ap, ()) if s <= h < e]
    if hits:
        return min(d for d, _ in hits), min(a for _, a in hits)
    tod = h % 1440
    for s, e, d, a in I.base_cap.get(ap, ()):
        ee = e if e > s else 1440
        if s <= tod < ee:
            return d, a
    return 10**6, 10**6


NO_EXEMPTION = -10**9          # announcement time of a disruption known "for ever" (static instance): nothing exempt
FIXED_DEP = NO_EXEMPTION + 1   # departure reported for flights fixed before T0 (exempt from every announced disruption)


def quota_rows(I: Instance, S: State, ap: str, h: int, kd: str,
               exec_deps: Dict[Tuple[str, int, str], List[int]]) -> List[Tuple[Optional[int], int]]:
    """Quota rows of one airport-hour cell under decision D5 (audit Stage 2, R-6), shared by the flow model, the path
    master and (in the same form) the evaluator.  Returns [(tau, cap)]: the new movements of the cell that depart at
    or after `tau` (None = all of them) may use at most `cap` slots.
      nominal row : all movements (executed + new)              <= nominal quota
      reduced row : movements departing at or after the announcement tau of the (known) reduction <= reduced quota
    Movements departing before the announcement are exempt from the reduction, as airborne flights are exempt from a
    ground delay programme.  Without announcement times (static instances) nothing is exempt: ROADEF rule."""
    qd, qa = capacity(I, {}, ap, h)
    qn = qd if kd == "d" else qa
    used = S.cap_used.get((ap, h, kd), 0)
    hits = [(s, e, d, a) for s, e, d, a in S.alt_cap.get(ap, ()) if s <= h < e]
    qr = min(d if kd == "d" else a for _, _, d, a in hits) if hits else qn
    # an alternative quota above the base quota is a capacity change, not a reduction: it applies to every movement
    rows: List[Tuple[Optional[int], int]] = [(None, max(max(qn, qr) - used, 0))]
    if qr < qn:
        tau = min(S.cap_ann.get((ap, s, e), NO_EXEMPTION) for s, e, _, _ in hits)
        used_r = sum(1 for dp in exec_deps.get((ap, h, kd), ()) if dp >= tau)
        rows.append((tau, max(qr - used_r, 0)))
    return rows


def executed_departures(I: Instance, S: State) -> Dict[Tuple[str, int, str], List[int]]:
    """cell -> departure times of the executed (committed) movements counted in S.cap_used.  A flight scheduled before
    T0 is fixed data, not a decision: it is reported with departure FIXED_DEP, so it is exempt from every announced
    reduction (D5: fixed pre-T0 flights count as exempt) but still binds where nothing is exempt (static ROADEF rule,
    announcement NO_EXEMPTION < FIXED_DEP)."""
    out: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    for f, (dep, arr, _) in S.executed.items():
        fo = I.flights[f]
        d = FIXED_DEP if fo.dep < I.start else dep
        out[(fo.orig, dep // 60 * 60, "d")].append(d)
        out[(fo.dest, arr // 60 * 60, "a")].append(d)
    return out


def initial_state(I: Instance, known_all: bool = True) -> State:
    """State at the start of the recovery period: flights departing before it are executed as planned."""
    tails = {a.id: TailState(a.orig, I.start) for a in I.aircraft.values() if not a.is_transp}
    st = State(I.start, tails)
    by_tail = defaultdict(list)
    for f in I.flights:
        if f.is_transp:
            continue
        if f.dep < I.start:
            d = I.flight_disr.get(f.idx, 0)
            if d < 0:
                st.cancelled.add(f.idx)
                continue
            dep, arr = f.dep + d, f.arr + d
            st.executed[f.idx] = (dep, arr, f.aircraft)
            by_tail[f.aircraft].append((dep, arr, f))
            st.cap_used[(f.orig, dep // 60 * 60, "d")] += 1
            st.cap_used[(f.dest, arr // 60 * 60, "a")] += 1
    for t, lst in by_tail.items():
        dep, arr, last = max(lst, key=lambda x: x[1])
        tails[t].ap = last.dest
        tails[t].ready = max(I.start, arr + I.aircraft[t].turn)
    if known_all:
        st.delays = dict(I.flight_disr)
        st.unavail = {k: list(v) for k, v in I.ac_unavail.items()}
        st.alt_cap = {k: list(v) for k, v in I.alt_cap.items()}
    return st


# ----------------------------------------------------------------------------
COARSE_GRID = tuple(list(range(0, 61, 10)) + list(range(80, 181, 20)) + list(range(210, 361, 30)))


@dataclass
class Params:
    delay_grid: Tuple[int, ...] = tuple(list(range(0, 61, 5)) + list(range(70, 121, 10)) +
                                        list(range(135, 241, 15)) + list(range(270, 481, 30)))
    hour_aligned: bool = True
    time_limit: float = 600.0
    mip_gap: float = 1e-4
    terminal: str = "none"          # 'none' | 'targets' | 'boundary'
    boundary_rate: Optional[float] = None   # euro per aircraft-minute of shortage (default: data-driven)
    bnd_conn: bool = True           # boundary component: passenger-connection protection
    bnd_need: bool = True           # boundary component: time-indexed range-class aircraft needs
    bnd_kind: bool = True           # boundary component: kind-level (model/config) end-of-window targets
    bnd_maint: bool = True          # boundary component: tail-specific targets for aircraft due for maintenance
    bnd_slot: bool = True           # boundary component: slot reservation for scheduled flights beyond the window
    slot_penalty: Optional[float] = None
    bnd_value: bool = True          # price family-level targets by the value of the tail's next rotation
    maint_target_penalty: float = 50000.0
    target_penalty: Optional[float] = None   # per missing aircraft at window end; default C_family
    threads: int = 1
    verbose: bool = False
    heur_effort: float = 0.05
    lp_restrict: bool = False       # LP-guided restriction: solve LP, keep copies with LP value>0 or small reduced cost
    rc_keep: float = 0.0
    restrict_singletons: bool = True
    lp_method: str = "choose"      # HiGHS 'solver' option for the LP relaxation ('simplex' / 'ipm' / 'choose')
    extra_deps: Optional[Dict[int, Tuple[int, ...]]] = None   # shared extra departure times per flight (R-3: the
    #                                OCC warm-start times, given to every method so that all share one decision set)
    delay_link: str = "aggregate"  # broken-itinerary delay linking: 'aggregate' (one row per itinerary) or 'copy' (v_pc >= y_c - z_p;
    #                                theoretically tighter LP; identical LP bounds on A01/A02/A04/A09/US03, twice the LP time)


@dataclass
class Commodity:
    kid: int
    kind: Tuple[str, Tuple[int, int, int]]
    family: str
    turn: int
    range_min: int
    cost_h: float
    tails: List[str]
    blocks: List[Tuple[int, int]]            # forbidden intervals (singleton only)
    maint: Optional[Tuple[str, int, int, int]] = None
    stations: Optional[Set[str]] = None      # singleton neighbourhood restriction (None = unrestricted)


@dataclass
class WindowResult:
    chosen: Dict[int, Tuple[int, int, int]]   # f -> (dep, arr, commodity id)
    cancelled: Set[int]
    obj: float
    bound: float
    gap: float
    status: str
    build_time: float
    solve_time: float
    n_cols: int
    n_rows: int
    commodities: List[Commodity]
    tail_of: Dict[int, str] = field(default_factory=dict)
    cert: Dict = field(default_factory=dict)      # LP / MIP status record (audit Stage 2, R-8)


class _Builder:
    def __init__(self):
        self.cost: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.integ: List[int] = []
        self.rows: List[List[Tuple[int, float]]] = []
        self.rlb: List[float] = []
        self.rub: List[float] = []
        self.pen: List[int] = []
        self.start = None
        self.ydef: Dict[int, List[int]] = {}
        self.absorb: Dict[int, int] = {}     # row -> the penalty column meant to satisfy it (complete_start)

    def var(self, cost, lb=0.0, ub=1.0, integer=True, penalty=False) -> int:
        if penalty:
            self.pen.append(len(self.cost))
        self.cost.append(cost)
        self.lb.append(lb)
        self.ub.append(ub)
        self.integ.append(1 if integer else 0)
        return len(self.cost) - 1

    def row(self, terms, lo, hi):
        self.rows.append(terms)
        self.rlb.append(lo)
        self.rub.append(hi)

    def complete_start(self, x0: np.ndarray) -> np.ndarray:
        """Given integer columns of a start, set penalty (continuous) columns to their minimal feasible value.
        Penalty columns are set in creation order; a row may contain penalty columns created earlier (e.g. the delay
        link v >= y - z contains z), whose values are then already known and are used.  A row registered in
        `absorb` is ignored by every penalty column except its designated one (z must not be raised to satisfy the
        delay link: v absorbs it)."""
        pen = set(self.pen)
        rows_of = defaultdict(list)
        for r, terms in enumerate(self.rows):
            for c, v in terms:
                if c in pen:
                    rows_of[c].append(r)
        done = set()
        for c in self.pen:
            need = 0.0
            for r in rows_of[c]:
                if self.absorb.get(r, c) != c:
                    continue
                coef = 0.0
                rest = 0.0
                for cc, v in self.rows[r]:
                    if cc == c:
                        coef += v
                    elif cc not in pen or cc in done:
                        rest += v * x0[cc]
                if coef > 0 and self.rlb[r] > -np.inf:
                    need = max(need, (self.rlb[r] - rest) / coef)
                if coef < 0 and self.rub[r] < np.inf:
                    need = max(need, (self.rub[r] - rest) / coef)
            x0[c] = min(max(need, self.lb[c]), self.ub[c])
            done.add(c)
        return x0

    def solve(self, time_limit, gap, threads, verbose, heur=0.05, integrality=True, fixed_ub=None, lp_method="choose"):
        h = highspy.Highs()
        if not integrality:
            h.setOptionValue("solver", lp_method)
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("mip_heuristic_effort", float(heur))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", float(gap))
        h.setOptionValue("threads", int(threads))
        lp = highspy.HighsLp()
        n = len(self.cost)
        lp.num_col_ = n
        lp.num_row_ = len(self.rows)
        lp.col_cost_ = np.array(self.cost, dtype=float)
        lp.col_lower_ = np.array(self.lb, dtype=float)
        lp.col_upper_ = np.array(self.ub if fixed_ub is None else fixed_ub, dtype=float)
        lp.row_lower_ = np.array(self.rlb, dtype=float)
        lp.row_upper_ = np.array(self.rub, dtype=float)
        # row-wise -> CSC
        cols = defaultdict(list)
        for r, terms in enumerate(self.rows):
            for c, v in terms:
                cols[c].append((r, v))
        start, index, value = [0], [], []
        for c in range(n):
            for r, v in cols.get(c, ()):
                index.append(r)
                value.append(v)
            start.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(start, dtype=np.int32)
        lp.a_matrix_.index_ = np.array(index, dtype=np.int32)
        lp.a_matrix_.value_ = np.array(value, dtype=float)
        if integrality:
            lp.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                               for i in self.integ]
        h.passModel(lp)
        if self.start is not None:
            hs = highspy.HighsSolution()
            hs.col_value = list(map(float, self.start))
            hs.value_valid = True
            h.setSolution(hs)
        t0 = time.time()
        h.run()
        st = time.time() - t0
        info = h.getInfo()
        status = h.modelStatusToString(h.getModelStatus())
        # status record for the certificate (R-8); primal_solution_status 2 = feasible point
        self.last_info = dict(status=status, objective=float(info.objective_function_value),
                              primal_solution_status=int(info.primal_solution_status),
                              dual_solution_status=int(info.dual_solution_status),
                              max_primal_infeasibility=float(info.max_primal_infeasibility),
                              max_dual_infeasibility=float(info.max_dual_infeasibility),
                              mip_dual_bound=float(info.mip_dual_bound) if integrality else None,
                              mip_gap=float(info.mip_gap) if integrality else None, time=st,
                              method=None if integrality else lp_method, threads=int(threads),
                              time_limit=float(time_limit))
        sol = np.array(h.getSolution().col_value) if info.primal_solution_status > 0 else None
        if not integrality:
            rc = np.array(h.getSolution().col_dual) if sol is not None else None
            return sol, info.objective_function_value, rc, status, st
        return sol, info.objective_function_value, info.mip_dual_bound, info.mip_gap, status, st


# ----------------------------------------------------------------------------
def build_commodities(I: Instance, S: State, horizon_end: int, restrict_singletons: bool = True) -> List[Commodity]:
    groups: Dict[Tuple, List[str]] = defaultdict(list)
    singles = []
    for t in S.tails:
        a = I.aircraft[t]
        blocks = [(s, e) for s, e in S.unavail.get(t, ()) if e > S.now]
        # a tail with a pending maintenance is a singleton in every window, whatever the window length, so that FH
        # and every RH stage restrict it in the same way (audit Stage 2, R-3); horizon_end is no longer used
        m = a.maint if (a.maint and a.maint[2] > S.now) else None
        if blocks or m:
            singles.append((t, blocks, m))
        else:
            groups[a.kind].append(t)
    comms = []
    for kind, tails in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        a = I.aircraft[tails[0]]
        comms.append(Commodity(len(comms), kind, a.family, a.turn, a.range_min, a.cost_h, sorted(tails), []))
    for t, blocks, m in singles:
        a = I.aircraft[t]
        bl = list(blocks)
        if m:
            bl.append((m[1], m[2]))
        st = None
        if restrict_singletons:
            # station set fixed from the recovery start (not the epoch) plus the tail's current airport, which lies
            # in it by induction: RH stage sets are therefore subsets of the FH set (R-3)
            st = {S.tails[t].ap} | {f.orig for f in I.flights if f.aircraft == t and f.dep >= I.start} |                 {f.dest for f in I.flights if f.aircraft == t and f.dep >= I.start}
            if m:
                st.add(m[0])
        comms.append(Commodity(len(comms), a.kind, a.family, a.turn, a.range_min, a.cost_h, [t], bl, m, st))
    return comms


def fixed_broken(I: Instance, S: State, legs: List[int]) -> bool:
    """Itinerary broken whatever the open decisions: a leg is already cancelled, or two already executed legs miss
    their connection (minimum connecting time as in the evaluator).  Its cancellation cost is a constant of the window;
    the model must neither charge it again nor charge its final-leg delay (audit Stage 2, R-2)."""
    if any(f in S.cancelled for f in legs):
        return True
    for l1, l2 in zip(legs, legs[1:]):
        if l1 in S.executed and l2 in S.executed:
            if S.executed[l2][0] < S.executed[l1][1] + min(MIN_CONNECT, I.flights[l2].dep - I.flights[l1].arr):
                return True
    return False


def constant_offset(I: Instance) -> float:
    """C: cost of the itineraries broken before T0 (a leg cancelled before T0 or a connection missed between two legs
    flown before T0).  The evaluator charges it whatever the plan; the models omit it (R-1/R-2), so that for every
    plan  model price = evaluated total - C."""
    from .costs import itinerary_info
    S = initial_state(I, known_all=True)
    c = 0.0
    for inf in itinerary_info(I):
        legs = [f for f, _ in inf.it.legs]
        if all(I.flights[f].dep < I.start for f in legs) or not fixed_broken(I, S, legs):
            continue
        started = I.flights[legs[0]].dep < I.start
        c += inf.it.count * (inf.cancel_cost_in if (inf.it.kind == "R" or started) else inf.cancel_cost_out)
    return c


def departure_set(f, delay: int, now: int, p: "Params") -> List[int]:
    """Departure times of open flight `f` available at decision time `now` (audit Stage 2, R-3, decision D3).

    The set is anchored at the flight's physical earliest departure e = t_f + max(d_f, 0) -- never at the decision
    epoch -- so it is the same for the full-horizon model and every rolling-horizon stage: the grid and clock-hour
    copies of copy_times(e), plus the shared extra times p.extra_deps[f] (the OCC warm start, given to every method),
    restricted to times >= now.  The maximum delay is therefore measured from e, and an RH stage at epoch tau uses
    exactly the FH set intersected with [tau, inf)."""
    e = f.dep + max(delay, 0)
    ts = set(copy_times(e, f.dur, p))
    if p.extra_deps:
        ts.update(t for t in p.extra_deps.get(f.idx, ()) if t >= e)
    return sorted(t for t in ts if t >= now)


def copy_times(earliest: int, dur: int, p: Params) -> List[int]:
    ts = {earliest + d for d in p.delay_grid}
    hi = earliest + max(p.delay_grid)
    if p.hour_aligned:
        h = (earliest // 60 + 1) * 60
        while h <= hi:
            ts.add(h)                      # departs at the start of an hour
            if h - dur > earliest:
                ts.add(h - dur)            # arrives at the start of an hour
            h += 60
    return sorted(ts)


def original_targets(I: Instance, S: State, t_end: int, window_flights: Set[int]) -> Counter:
    """Aircraft needed per (airport, family) at the end of a look-ahead window:
    origin of each tail's first original flight beyond the window."""
    need = Counter()
    by_tail = defaultdict(list)
    for f in I.flights:
        if (not f.is_transp and f.dep >= S.now and f.idx not in window_flights
                and f.idx not in S.executed and f.idx not in S.cancelled):
            by_tail[f.aircraft].append(f)
    last_dest = {}
    for f in I.flights:
        if not f.is_transp and f.dep < I.end:
            if f.aircraft not in last_dest or f.dep > last_dest[f.aircraft][0]:
                last_dest[f.aircraft] = (f.dep, f.dest)
    for t in S.tails:
        lst = [f for f in by_tail.get(t, []) if f.dep < I.end]
        fam = I.aircraft[t].family
        if lst:
            need[(min(lst, key=lambda f: f.dep).orig, fam)] += 1
        else:   # no more flights planned in the recovery period: end-of-day location
            need[(last_dest.get(t, (0, I.aircraft[t].orig))[1], fam)] += 1
    return need


def original_needs(I: Instance, S: State, window_flights: Set[int], itins=None) -> list:
    """Time-indexed aircraft needs beyond the look-ahead window: for each tail, (airport, family, time)
    of its first original flight after the window, or its end-of-day location at the recovery end."""
    by_tail = defaultdict(list)
    last_dest = {}
    for f in I.flights:
        if f.is_transp:
            continue
        if f.dep < I.end and (f.aircraft not in last_dest or f.dep > last_dest[f.aircraft][0]):
            last_dest[f.aircraft] = (f.dep, f.dest)
        if (f.dep >= S.now - 24 * 60 and f.dep < I.end and f.idx not in window_flights
                and f.idx not in S.executed and f.idx not in S.cancelled and S.delays.get(f.idx, 0) >= 0):
            by_tail[f.aircraft].append(f)
    out = []
    for t in S.tails:
        fam = I.aircraft[t].family
        lst = by_tail.get(t)
        if lst:
            f = min(lst, key=lambda f: f.dep)
            out.append((f.orig, fam, max(S.now, f.dep + max(S.delays.get(f.idx, 0), 0)), f.dur, I.aircraft[t].kind,
                        _rotation_value(I, sorted(lst, key=lambda g: g.dep), itins)))
        else:
            out.append((last_dest.get(t, (0, I.aircraft[t].orig))[1], fam, I.end, 0, I.aircraft[t].kind,
                        I.weights[2] * I.pos_pen[0]))
    return out


_RATE_CACHE: Dict[str, float] = {}
_FCANCEL_CACHE: Dict[str, Dict[int, float]] = {}


def _flight_cancel_value(I: Instance, itins) -> Dict[int, float]:
    """Net cost of cancelling each flight in isolation: passenger cancellation costs minus saved operating cost."""
    key = I.name + getattr(I, "regime", "")
    if key not in _FCANCEL_CACHE:
        v = Counter()
        for inf in itins or []:
            for f, _ in inf.it.legs:
                v[f] += inf.it.count * inf.cancel_cost_out
        alpha = I.weights[0]
        _FCANCEL_CACHE[key] = {f.idx: v.get(f.idx, 0.0) - alpha * I.aircraft[f.aircraft].cost_h * f.dur / 60.0
                               for f in I.flights if not f.is_transp}
    return _FCANCEL_CACHE[key]


def _rotation_value(I: Instance, chain, itins) -> float:
    """Value of a tail's next rotation: cancelling it until the aircraft is back at the chain's origin."""
    vals = _flight_cancel_value(I, itins)
    if not chain:
        return 0.0
    home = chain[0].orig
    tot = 0.0
    for f in chain:
        tot += max(vals.get(f.idx, 0.0), 0.0)
        if f.dest == home:
            break
    return tot


def _default_rate(I: Instance, itins: List[ItinInfo]) -> float:
    """Average passenger delay cost per minute of a flight (euro/min)."""
    if I.name not in _RATE_CACHE:
        per_f = Counter()
        for inf in itins:
            for f, _ in inf.it.legs:
                per_f[f] += inf.it.count * inf.delay_rate
        vals = [v for f, v in per_f.items()] or [1.0]
        _RATE_CACHE[I.name] = float(np.mean(vals))
    return _RATE_CACHE[I.name]


def solve_window(I: Instance, S: State, flights: List[int], itins: List[ItinInfo], p: Params,
                 final: bool, warm: Optional[Dict[int, object]] = None) -> WindowResult:
    """Optimise decisions of `flights` (open) given state S.  `final` = window reaches recovery end."""
    t_build = time.time()
    alpha, beta, gamma = I.weights
    B = _Builder()
    fl = [I.flights[f] for f in flights]
    fset = set(flights)
    horizon_end = max([f.dep for f in fl] + [S.now]) + max(p.delay_grid) + 600
    comms = build_commodities(I, S, horizon_end, p.restrict_singletons)
    loads = flight_loads(I)

    # passenger bookkeeping ------------------------------------------------
    legs_of: Dict[int, List[ItinInfo]] = defaultdict(list)
    for inf in itins:
        for f, _ in inf.it.legs:
            legs_of[f].append(inf)
    avg_cancel: Dict[int, float] = {}
    for f in flights:
        lst = legs_of.get(f, [])
        pax = sum(i.it.count for i in lst)
        avg_cancel[f] = (sum(i.it.count * i.cancel_cost_out for i in lst) / pax) if pax else 0.0

    # copies and arcs ----------------------------------------------------------
    def ann_block(k, s0, e0):
        t0 = k.tails[0]
        if k.maint and (s0, e0) == (k.maint[1], k.maint[2]) and (t0, s0, e0) not in S.unavail_ann:
            return -10**9
        return S.unavail_ann.get((t0, s0, e0), NO_EXEMPTION)

    nodes: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    arcs = []   # (col, f, dep, arr, kid)
    x_of_f: Dict[int, List[int]] = defaultdict(list)
    L = {}
    for f in fl:
        d = S.delays.get(f.idx, 0)
        if d < 0:          # cancelled by the disruption
            L[f.idx] = B.var(0.0, 1.0, 1.0)
            continue
        deps = departure_set(f, d, S.now, p)     # R-3: warm-start times outside the set are not added
        for k in comms:
            if k.family != I.aircraft[f.aircraft].family or f.dur > k.range_min:
                continue
            if k.stations is not None and f.aircraft not in k.tails and                     (f.orig not in k.stations or f.dest not in k.stations):
                continue
            sp = spill(loads.get(f.idx, (0, 0, 0)), k.kind[1])
            for dep in deps:
                arr = dep + f.dur
                ready = arr + k.turn
                # an outage blocks the copy unless the copy departs before the outage is announced (D5, R-6;
                # maintenance blocks are never exempt)
                if any(dep < e and ready > s and dep >= ann_block(k, s, e) for s, e in k.blocks):
                    continue
                c = alpha * k.cost_h * f.dur / 60.0 + sp * avg_cancel[f.idx]
                col = B.var(c)
                arcs.append((col, f.idx, dep, arr, k.kid))
                x_of_f[f.idx].append(len(arcs) - 1)
                nodes[(k.kid, f.orig)].add(dep)
                nodes.setdefault((k.kid, f.dest), set())
        L[f.idx] = B.var(0.0)
    # supplies
    supply = Counter()
    for k in comms:
        for t in k.tails:
            ts = S.tails[t]
            supply[(k.kid, ts.ap, ts.ready)] += 1
            nodes[(k.kid, ts.ap)].add(ts.ready)
        if k.maint and k.maint[1] >= S.now:
            nodes[(k.kid, k.maint[0])].add(k.maint[1])
    if final:
        for key in list(nodes):
            nodes[key].add(I.end)
    needs = []
    if p.terminal == "boundary" and not final:
        needs = original_needs(I, S, fset, itins)
        fam_comms = defaultdict(list)
        for k in comms:
            fam_comms[k.family].append(k.kid)
        for ap, fam, t, _, _, _ in needs:
            for kid in fam_comms[fam]:
                nodes[(kid, ap)].add(t)
    # node compression: arrivals are attached to the first node at or after their ready time
    SINK_T = 10**8
    for key in list(nodes):
        nodes[key].add(SINK_T)
    sorted_nodes = {key: sorted(ts) for key, ts in nodes.items()}

    def arr_node(kid, ap, ready):
        ts = sorted_nodes[(kid, ap)]
        return ts[bisect.bisect_left(ts, ready)]

    # ground arcs and flow conservation -------------------------------------
    node_idx: Dict[Tuple[int, str, int], int] = {}
    node_terms: List[List[Tuple[int, float]]] = []
    node_rhs: List[float] = []
    ground_from: Dict[Tuple[int, str, int], int] = {}
    terminal_var: Dict[Tuple[int, str], int] = {}
    ntails = {k.kid: len(k.tails) for k in comms}
    for (kid, ap), ts in sorted_nodes.items():
        for t in ts:
            node_idx[(kid, ap, t)] = len(node_terms)
            node_terms.append([])
            node_rhs.append(supply.get((kid, ap, t), 0))
        for a, b in zip(ts, ts[1:]):
            g = B.var(0.0, 0.0, ntails[kid])
            ground_from[(kid, ap, a)] = g
            node_terms[node_idx[(kid, ap, a)]].append((g, -1.0))
            node_terms[node_idx[(kid, ap, b)]].append((g, 1.0))
        g = B.var(0.0, 0.0, ntails[kid])
        terminal_var[(kid, ap)] = g
        ground_from[(kid, ap, ts[-1])] = g
        node_terms[node_idx[(kid, ap, ts[-1])]].append((g, -1.0))
    kturn = {k.kid: k.turn for k in comms}
    for col, f, dep, arr, kid in arcs:
        fo = I.flights[f]
        node_terms[node_idx[(kid, fo.orig, dep)]].append((col, -1.0))
        node_terms[node_idx[(kid, fo.dest, arr_node(kid, fo.dest, arr + kturn[kid]))]].append((col, 1.0))
    # inflow - outflow = -supply  <=>  outflow - inflow = supply
    for terms, rhs in zip(node_terms, node_rhs):
        B.row(terms, -rhs, -rhs)

    # cover ---------------------------------------------------------------------
    for f in fl:
        terms = [(arcs[a][0], 1.0) for a in x_of_f.get(f.idx, [])] + [(L[f.idx], 1.0)]
        B.row(terms, 1.0, 1.0)

    # airport capacity ---------------------------------------------------------
    # aggregated copy-selection variables y[f,dep] = sum_k x[f,dep,k] (reduce nonzeros)
    ycols: Dict[Tuple[int, int], int] = {}
    ymem: Dict[int, List[int]] = defaultdict(list)
    for col, f, dep, arr, kid in arcs:
        key = (f, dep)
        if key not in ycols:
            ycols[key] = B.var(0.0, 0.0, 1.0, integer=False)
        ymem[ycols[key]].append(col)
    for yc, members in ymem.items():
        B.row([(yc, 1.0)] + [(c, -1.0) for c in members], 0.0, 0.0)
    B.ydef = ymem
    use = defaultdict(list)
    for (f, dep), yc in ycols.items():
        fo = I.flights[f]
        use[(fo.orig, dep // 60 * 60, "d")].append((yc, f, dep))
        use[(fo.dest, (dep + fo.dur) // 60 * 60, "a")].append((yc, f, dep))
    reserve = Counter()
    if p.terminal == "boundary" and p.bnd_slot and not final:
        # slot reservation: capacity that scheduled flights beyond the window will need
        for fo in I.flights:
            if (fo.is_transp or fo.idx in fset or fo.idx in S.executed or fo.idx in S.cancelled
                    or fo.dep >= I.end or S.delays.get(fo.idx, 0) < 0):
                continue
            e0 = max(fo.dep + max(S.delays.get(fo.idx, 0), 0), S.now)
            reserve[(fo.orig, e0 // 60 * 60, "d")] += 1
            reserve[(fo.dest, (e0 + fo.dur) // 60 * 60, "a")] += 1
        lam_slot = p.slot_penalty if p.slot_penalty is not None else 120.0 * _default_rate(I, itins)
    exec_deps = executed_departures(I, S)
    for (ap, h, kd), cols in use.items():
        # D5 (R-6): nominal row over all movements, reduced row over movements departing at/after the announcement;
        # rows with the same left-hand side are merged (the usual case: every copy departs after the announcement)
        lhs_cap: Dict[Tuple[int, ...], int] = {}
        for tau, cap in quota_rows(I, S, ap, h, kd, exec_deps):
            lhs = tuple(sorted(c for c, _, dep in cols if tau is None or dep >= tau))
            if lhs:
                lhs_cap[lhs] = min(cap, lhs_cap.get(lhs, cap))
        for lhs, cap in lhs_cap.items():
            r = min(reserve.get((ap, h, kd), 0), cap)
            if cap - r < len({f for c, f, _ in cols if c in set(lhs)}):
                terms = [(c, 1.0) for c in lhs]
                if r > 0:
                    sv = B.var(lam_slot, 0.0, r, integer=False, penalty=True)
                    terms.append((sv, -1.0))
                B.row(terms, -np.inf, cap - r)

    # maintenance ----------------------------------------------------------------
    for k in comms:
        if not k.maint:
            continue
        ap_m, s_m, e_m, rem = k.maint
        t = k.tails[0]
        win_last = max((f.dep for f in fl), default=S.now)
        # the location requirement is imposed only if maintenance starts inside the look-ahead window
        # (beyond it, the positioning targets of the tail's next original flights take over)
        if s_m < S.now:
            continue
        if final or s_m <= win_last + 60:
            key = (k.kid, ap_m, s_m)
            sl = B.var(BIG_SLACK, 0.0, 1.0, integer=False, penalty=True)
            if key in ground_from:
                B.row([(ground_from[key], 1.0), (sl, 1.0)], 1.0, np.inf)
            else:
                B.row([(sl, 1.0)], 1.0, np.inf)
        elif p.terminal == "boundary" and p.bnd_maint:
            # tail-specific positioning target: end the window where the tail's next planned flight departs
            nxt = [fo for fo in I.flights if fo.aircraft == t and not fo.is_transp and fo.idx not in fset
                   and fo.idx not in S.executed and fo.idx not in S.cancelled and win_last < fo.dep < s_m]
            X = min(nxt, key=lambda fo: fo.dep).orig if nxt else ap_m
            sl = B.var(p.maint_target_penalty, 0.0, 1.0, integer=False, penalty=True)
            tv = terminal_var.get((k.kid, X))
            B.row(([(tv, 1.0)] if tv is not None else []) + [(sl, 1.0)], 1.0, np.inf)
        slh = B.var(MAINT_MINUTE_PENALTY, 0.0, np.inf, integer=False, penalty=True)
        terms = [(col, float(I.flights[f].dur)) for col, f, dep, arr, kid in arcs if kid == k.kid and dep < s_m]
        if terms:
            B.row(terms + [(slh, -1.0)], -np.inf, max(0, rem - S.tails[t].flown))

    # passengers ------------------------------------------------------------------
    arc_by_f = defaultdict(list)       # f -> [(y col, dep, arr)]
    for (f, dep), yc in sorted(ycols.items(), key=lambda kv: kv[0]):
        arc_by_f[f].append((yc, dep, dep + I.flights[f].dur))
    xarc_by_f = defaultdict(list)      # f -> [(x col, dep, arr)] for costs
    for col, f, dep, arr, kid in arcs:
        xarc_by_f[f].append((col, dep, arr))
    for inf in itins:
        legs = [f for f, _ in inf.it.legs]
        open_legs = [f for f in legs if f in fset]
        if not open_legs:
            continue
        if fixed_broken(I, S, legs):
            continue   # already broken: cost fixed
        started = legs[0] in S.executed
        ccost = inf.it.count * (inf.cancel_cost_in if (inf.it.kind == "R" or started) else inf.cancel_cost_out)
        last = legs[-1]
        if len(legs) == 1:
            # single-leg itinerary: the delay cost sits on the copies of the leg (a cancelled leg flies no copy)
            if last in fset:
                for col, dep, arr in xarc_by_f.get(last, []):
                    dc = inf.delay_cost(arr - inf.sched_arr)
                    if dc:
                        B.cost[col] += inf.it.count * dc
            B.cost[L[last]] += ccost
            continue
        z = B.var(ccost, 0.0, 1.0, integer=False, penalty=True)
        for f in open_legs:
            B.row([(z, 1.0), (L[f], -1.0)], 0.0, np.inf)
        # delay cost of a multi-leg itinerary is charged only if it is intact (ROADEF rule, as in the evaluator):
        # v_pc >= y_{last,c} - z_p.  Exact because raising z_p voluntarily never pays: its cancellation cost is at
        # least the largest delay cost it could save (checked here, audit Stage 2 R-1).
        if last in fset:
            for yc, dep, arr in arc_by_f.get(last, []):
                dc = inf.it.count * inf.delay_cost(arr - inf.sched_arr)
                if not dc:
                    continue
                if dc > ccost + 1e-6:
                    raise ValueError(f"itinerary {inf.it.id}: delay cost {dc:.2f} exceeds cancellation cost {ccost:.2f}; "
                                     "the broken-itinerary delay linking v >= y - z would not be exact")
                if p.delay_link == "aggregate":
                    continue
                v = B.var(dc, 0.0, 1.0, integer=False, penalty=True)
                B.absorb[len(B.rows)] = v
                B.row([(v, 1.0), (z, 1.0), (yc, -1.0)], 0.0, np.inf)
            if p.delay_link == "aggregate":
                terms = [(yc, inf.it.count * inf.delay_cost(arr - inf.sched_arr)) for yc, dep, arr in arc_by_f.get(last, [])]
                terms = [(yc, dc) for yc, dc in terms if dc]
                if terms:
                    dmax = max(dc for _, dc in terms)
                    w = B.var(1.0, 0.0, np.inf, integer=False, penalty=True)
                    B.absorb[len(B.rows)] = w
                    B.row([(w, 1.0), (z, dmax)] + [(yc, -dc) for yc, dc in terms], 0.0, np.inf)
        for l1, l2 in zip(legs, legs[1:]):
            f1, f2 = I.flights[l1], I.flights[l2]
            mct = min(MIN_CONNECT, f2.dep - f1.arr)
            if l2 not in fset:
                if (p.terminal == "boundary" and p.bnd_conn and l1 in fset and l2 not in S.executed and l2 not in S.cancelled
                        and S.delays.get(l2, 0) >= 0):
                    ref = max(f2.dep + max(S.delays.get(l2, 0), 0), S.now)
                    late = [c for c, dep, arr in arc_by_f.get(l1, []) if arr + mct > ref]
                    if late:
                        B.row([(z, 1.0)] + [(c, -1.0) for c in late], 0.0, np.inf)
                continue
            c2 = arc_by_f.get(l2, [])
            if l1 in S.executed:
                a1 = S.executed[l1][1]
                bad = [(c, 1.0) for c, dep, arr in c2 if dep < a1 + mct]
                if bad:
                    B.row([(z, 1.0)] + [(c, -1.0) for c, _ in bad], 0.0, np.inf)
                continue
            if l1 not in fset:
                continue
            c1 = arc_by_f.get(l1, [])
            thresholds = sorted({arr for _, _, arr in c1})
            min_dep2 = min((dep for _, dep, _ in c2), default=None)
            for a in thresholds:
                if min_dep2 is None or min_dep2 >= a + mct:
                    continue
                late1 = [c for c, _, arr in c1 if arr >= a]
                early2 = [c for c, dep, _ in c2 if dep < a + mct]
                if late1 and early2:
                    B.row([(z, 1.0)] + [(c, -1.0) for c in late1] + [(c, -1.0) for c in early2], -1.0, np.inf)

    # terminal conditions -----------------------------------------------------------
    kinds_by_comm = {k.kid: k for k in comms}
    if final:
        cc = I.pos_pen  # family, model, config
        c_fam = gamma * (cc[0] - cc[1])
        c_mod = gamma * (cc[1] - cc[2])
        c_cfg = gamma * cc[2]
        present = defaultdict(list)   # (ap, model, config) -> cols
        for (kid, ap), ts in nodes.items():
            k = kinds_by_comm[kid]
            key = (kid, ap, I.end)
            if key in ground_from:
                present[(ap, k.kind)].append(ground_from[key])
        for col, f, dep, arr, kid in arcs:
            k = kinds_by_comm[kid]
            if arr <= I.end < arr + k.turn:
                present[(I.flights[f].dest, k.kind)].append(col)
        for ap, reqs in I.positions.items():
            by_model = defaultdict(int)
            by_fam = defaultdict(int)
            cols_model = defaultdict(list)
            cols_fam = defaultdict(list)
            fam_of_model = {}
            for a in I.aircraft.values():
                fam_of_model[a.model] = a.family
            for model, cfg, cnt in reqs:
                u = B.var(c_cfg, 0.0, np.inf, integer=False, penalty=True)
                cols = present.get((ap, (model, cfg)), [])
                B.row([(u, 1.0)] + [(c, 1.0) for c in cols], cnt, np.inf)
                by_model[model] += cnt
                by_fam[fam_of_model.get(model, model)] += cnt
            for (ap2, kind), cols in present.items():
                if ap2 != ap:
                    continue
                cols_model[kind[0]] += cols
                cols_fam[fam_of_model.get(kind[0], kind[0])] += cols
            for model, cnt in by_model.items():
                u = B.var(c_mod, 0.0, np.inf, integer=False, penalty=True)
                B.row([(u, 1.0)] + [(c, 1.0) for c in cols_model.get(model, [])], cnt, np.inf)
            for fam, cnt in by_fam.items():
                u = B.var(c_fam, 0.0, np.inf, integer=False, penalty=True)
                B.row([(u, 1.0)] + [(c, 1.0) for c in cols_fam.get(fam, [])], cnt, np.inf)
    elif p.terminal == "targets":
        lam = p.target_penalty if p.target_penalty is not None else gamma * I.pos_pen[0]
        need = original_targets(I, S, 0, fset)
        endcols = defaultdict(list)
        for (kid, ap), g in terminal_var.items():
            endcols[(ap, kinds_by_comm[kid].family)].append(g)
        for (ap, fam), cnt in need.items():
            u = B.var(lam, 0.0, np.inf, integer=False, penalty=True)
            B.row([(u, 1.0)] + [(c, 1.0) for c in endcols.get((ap, fam), [])], cnt, np.inf)

    elif p.terminal == "boundary" and needs:
        rate = p.boundary_rate if p.boundary_rate is not None else _default_rate(I, itins)
        lam_end = gamma * I.pos_pen[0]
        by_af = defaultdict(list)
        for ap, fam, t, dur, _, _ in needs:
            by_af[(ap, fam)].append((t, dur))
        comm_of_fam = defaultdict(list)
        for k in comms:
            comm_of_fam[k.family].append(k)
        for (ap, fam), lst in (by_af.items() if p.bnd_need else ()):
            ranges = sorted({k.range_min for k in comm_of_fam[fam]})
            # eligibility class of a need = smallest range covering its duration (nested classes -> Hall conditions)
            cls = []
            for t, dur in lst:
                r = next((x for x in ranges if x >= dur), None)
                if r is not None:
                    cls.append((t, r))
            for r in ranges:
                members = [k.kid for k in comm_of_fam[fam] if k.range_min >= r]
                ts = [t for t, rr in cls if rr >= r]
                if not ts or (r != ranges[0] and all(rr == ranges[0] for _, rr in cls)):
                    continue
                cnt = Counter(ts)
                levels = sorted(cnt)
                cum = 0
                for j, t in enumerate(levels):
                    cum += cnt[t]
                    nxt = levels[j + 1] if j + 1 < len(levels) else I.end
                    w = lam_end if t >= I.end else rate * max(nxt - t, 5)
                    u = B.var(w, 0.0, np.inf, integer=False, penalty=True)
                    terms = [(ground_from[(kid, ap, t)], 1.0) for kid in members if (kid, ap, t) in ground_from]
                    B.row([(u, 1.0)] + terms, cum, np.inf)

    if p.terminal == "boundary" and needs and p.bnd_kind:
        # kind-level hierarchical targets on end-of-window locations (ROADEF positioning penalty structure)
        cc = I.pos_pen
        w_cfg, w_mod, w_fam = gamma * cc[2], gamma * (cc[1] - cc[2]), gamma * (cc[0] - cc[1])
        R_k, R_m, R_f = Counter(), Counter(), Counter()
        V_f = defaultdict(list)
        for ap, fam, t, dur, kind, val in needs:
            R_k[(ap, kind)] += 1
            R_m[(ap, kind[0])] += 1
            R_f[(ap, fam)] += 1
            V_f[(ap, fam)].append(val)
        P_k, P_m, P_f = defaultdict(list), defaultdict(list), defaultdict(list)
        for (kid, ap), g in terminal_var.items():
            k = kinds_by_comm[kid]
            P_k[(ap, k.kind)].append(g)
            P_m[(ap, k.kind[0])].append(g)
            P_f[(ap, k.family)].append(g)
        for R, P, w in ((R_k, P_k, w_cfg), (R_m, P_m, w_mod)) + (((R_f, P_f, w_fam),) if not p.bnd_need else ()):
            for key, cnt in R.items():
                if w is w_fam and p.bnd_value:
                    # rotation-value-priced family targets: shortages hit the cheapest rotations first
                    us = [B.var(max(v, w_fam), 0.0, 1.0, integer=False, penalty=True) for v in sorted(V_f[key])]
                    B.row([(u, 1.0) for u in us] + [(c, 1.0) for c in P.get(key, [])], cnt, np.inf)
                    continue
                u = B.var(w, 0.0, np.inf, integer=False, penalty=True)
                B.row([(u, 1.0)] + [(c, 1.0) for c in P.get(key, [])], cnt, np.inf)

    if warm:
        B.start = _warm_vector(I, S, B, comms, arcs, L, sorted_nodes, ground_from, supply, warm, fl, arr_node)
    build_time = time.time() - t_build
    lp_time = 0.0
    lp_bound = None
    fixed_ub = None
    cert = dict(lp=None, mip=None)
    n_fixed = 0
    if p.lp_restrict:
        # LP-guided restriction: keep copies used by the LP or with small reduced cost (+ the warm start)
        B.last_info = None
        lsol, lobj, rc, lstat, lp_time = B.solve(p.time_limit, p.mip_gap, p.threads, p.verbose, integrality=False,
                                                  lp_method=p.lp_method)
        cert["lp"] = dict(getattr(B, "last_info", None) or dict(status=lstat, objective=lobj, time=lp_time))
        cert["lp"]["status"] = lstat
        # R-8: the LP value is a lower bound only if the LP was solved to optimality (F-09)
        if lsol is not None and lstat == "Optimal":
            lp_bound = lobj
        if lsol is not None:        # the restriction itself is a heuristic and may use any LP point
            fixed_ub = list(B.ub)
            for col, f, dep, arr, kid in arcs:
                keep = lsol[col] > 1e-6 or rc[col] <= p.rc_keep
                if B.start is not None and B.start[col] > 0.5:
                    keep = True
                if not keep:
                    fixed_ub[col] = 0.0
                    n_fixed += 1
    B.last_info = None
    sol, obj, bound, gap, status, st = B.solve(max(1.0, p.time_limit - lp_time), p.mip_gap, p.threads, p.verbose,
                                              heur=p.heur_effort, fixed_ub=fixed_ub)
    cert["mip"] = dict(getattr(B, "last_info", None) or dict(status=status), restricted=n_fixed > 0,
                       n_columns_fixed=n_fixed, incumbent_model_obj=obj if sol is not None else None)
    st += lp_time
    # R-8 (F-07): the MIP's dual bound and 'Optimal' status refer to the restricted model if columns were fixed; they
    # are then no bound for the model.  Valid bound = optimal LP value, or the dual bound of the unrestricted MIP.
    cands = ([lp_bound] if lp_bound is not None else []) + \
            ([bound] if n_fixed == 0 and bound is not None and np.isfinite(bound) and sol is not None else [])
    bound = max(cands) if cands else None
    gap = (obj - bound) / max(abs(obj), 1e-9) if (sol is not None and bound is not None) else None
    cert["bound"], cert["bound_source"] = bound, (None if bound is None else
                                                  "LP optimal" if bound == lp_bound else "unrestricted MIP dual bound")
    chosen, canc = {}, set()
    if sol is not None:
        for col, f, dep, arr, kid in arcs:
            if sol[col] > 0.5:
                chosen[f] = (dep, arr, kid)
        for f, v in L.items():
            if sol[v] > 0.5 and f not in chosen:
                canc.add(f)
    else:
        canc = set(f.idx for f in fl)
    res = WindowResult(chosen, canc, obj, bound, gap, status, build_time, st, len(B.cost), len(B.rows), comms)
    res.cert = cert
    res.tail_of = assign_tails(I, S, res)
    return res


def assign_tails(I: Instance, S: State, res: WindowResult) -> Dict[int, str]:
    """Decompose commodity flows into tail rotations (FIFO on ready times)."""
    avail: Dict[int, List[List]] = defaultdict(list)   # kid -> [[ready, ap, tail]]
    for k in res.commodities:
        for t in k.tails:
            ts = S.tails[t]
            avail[k.kid].append([ts.ready, ts.ap, t])
    kturn = {k.kid: k.turn for k in res.commodities}
    tail_of = {}
    for f, (dep, arr, kid) in sorted(res.chosen.items(), key=lambda kv: kv[1][0]):
        fo = I.flights[f]
        cands = [c for c in avail[kid] if c[1] == fo.orig and c[0] <= dep]
        if not cands:
            raise RuntimeError(f"flow decomposition failed for flight {fo.num} ({f})")
        c = min(cands, key=lambda c: c[0])
        tail_of[f] = c[2]
        c[0], c[1] = arr + kturn[kid], fo.dest
    return tail_of


def _warm_vector(I, S, B, comms, arcs, L, nodes, ground_from, supply, warm, fl, arr_node):
    """Translate a plan {f: (dep, tail) | 'C'} into a full start vector (None if not representable)."""
    x0 = np.zeros(len(B.cost))
    kid_of_tail = {t: k.kid for k in comms for t in k.tails}
    kturn = {k.kid: k.turn for k in comms}
    col_of = {(f, dep, kid): col for col, f, dep, arr, kid in arcs}
    events = Counter()   # (kid, ap, t) -> net outflow
    for f in fl:
        w = warm.get(f.idx, "C")
        if w == "C" or w is None:
            x0[L[f.idx]] = 1.0
            continue
        dep, tail = w
        kid = kid_of_tail.get(tail)
        col = col_of.get((f.idx, dep, kid))
        if col is None:
            x0[L[f.idx]] = 1.0
            continue
        x0[col] = 1.0
        events[(kid, f.orig, dep)] += 1
        events[(kid, f.dest, arr_node(kid, f.dest, dep + f.dur + kturn[kid]))] -= 1
    for (kid, ap), ts in nodes.items():
        level = 0
        for t in ts:
            level += supply.get((kid, ap, t), 0) - events.get((kid, ap, t), 0)
            if level < 0:
                return None     # start not flow-feasible (e.g. tail mapping differs); skip warm start
            x0[ground_from[(kid, ap, t)]] = level
    for yc, members in B.ydef.items():
        x0[yc] = sum(x0[c] for c in members)
    return B.complete_start(x0)
