"""Recovery policies: full-horizon MIP, rolling-horizon MIP (with look-ahead,
intermediate positioning targets and progressive information), and a
rule-based OCC baseline (delay propagation with capacity-aware slotting)."""
from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .costs import itinerary_info
from .instance import Instance
from .model import Params, State, capacity, departure_set, executed_departures, initial_state, quota_rows, solve_window


# ----------------------------------------------------------------------------
# Information revelation protocol
# ----------------------------------------------------------------------------
@dataclass
class Revelation:
    """Time at which each disruption becomes known to the OCC."""
    flight: Dict[int, int] = field(default_factory=dict)
    aircraft: Dict[Tuple[str, int], int] = field(default_factory=dict)
    airport: Dict[Tuple[str, int], int] = field(default_factory=dict)
    # decision D5 (R-6): exemptions by announcement time exist only for dynamically revealed disruptions (scenarios,
    # progressive information); on a static instance every disruption binds every movement (ROADEF rule)
    exempt: bool = False

    @staticmethod
    def static(I: Instance) -> "Revelation":
        r = Revelation()
        for f in I.flight_disr:
            r.flight[f] = I.start
        for t, lst in I.ac_unavail.items():
            for k, _ in enumerate(lst):
                r.aircraft[(t, k)] = I.start
        for ap, lst in I.alt_cap.items():
            for k, _ in enumerate(lst):
                r.airport[(ap, k)] = I.start
        return r

    @staticmethod
    def progressive(I: Instance, lead: Tuple[int, int] = (60, 180), seed: int = 0, epoch: int = 60) -> "Revelation":
        """Each disruption is announced `lead` minutes (uniform) before it bites,
        but never before the start of the recovery period.  Reductions and outages are announced at a decision
        epoch (rounded up to a multiple of `epoch` minutes after T0; audit Stage 2, R-6)."""
        rng = random.Random(seed)
        r = Revelation(exempt=True)
        up = lambda t: t if epoch <= 0 else I.start + -(-(t - I.start) // epoch) * epoch
        for f, d in sorted(I.flight_disr.items()):
            r.flight[f] = max(I.start, I.flights[f].dep - rng.randint(*lead))
        for t, lst in sorted(I.ac_unavail.items()):
            for k, (s, e) in enumerate(lst):
                r.aircraft[(t, k)] = up(max(I.start, s - rng.randint(*lead)))
        for ap, lst in sorted(I.alt_cap.items()):
            for k, (s, e, d, a) in enumerate(lst):
                r.airport[(ap, k)] = up(max(I.start, s - rng.randint(*lead)))
        return r

    def apply(self, I: Instance, S: State, now: int) -> None:
        S.delays = {f: I.flight_disr[f] for f, t in self.flight.items() if t <= now}
        S.unavail = defaultdict(list)
        for (t, k), tt in self.aircraft.items():
            if tt <= now:
                S.unavail[t].append(I.ac_unavail[t][k])
        S.unavail = dict(S.unavail)
        S.alt_cap = defaultdict(list)
        for (ap, k), tt in self.airport.items():
            if tt <= now:
                S.alt_cap[ap].append(I.alt_cap[ap][k])
        S.alt_cap = dict(S.alt_cap)
        self.set_announcements(I, S, known_only=now)

    def set_announcements(self, I: Instance, S: State, known_only: Optional[int] = None) -> None:
        """Announcement times of reductions and outages (decision D5, R-6: a flight departing before an announcement
        is exempt from it).  `known_only` = only those announced by then."""
        S.cap_ann, S.unavail_ann = {}, {}
        if not self.exempt:
            return
        for (ap, k), tt in self.airport.items():
            if known_only is None or tt <= known_only:
                s, e = I.alt_cap[ap][k][:2]
                S.cap_ann[(ap, s, e)] = min(tt, S.cap_ann.get((ap, s, e), tt))
        for (t, k), tt in self.aircraft.items():
            if known_only is None or tt <= known_only:
                s, e = I.ac_unavail[t][k]
                S.unavail_ann[(t, s, e)] = min(tt, S.unavail_ann.get((t, s, e), tt))


# ----------------------------------------------------------------------------
@dataclass
class RunResult:
    plan: Dict
    runtime: float
    stages: int
    solve_time: float
    max_stage_time: float
    log: List[Dict] = field(default_factory=list)
    bound: Optional[float] = None
    obj: Optional[float] = None
    gap: Optional[float] = None
    fallbacks: int = 0
    cert: Dict = field(default_factory=dict)      # FH: LP / MIP status record of the single model (R-8)


def _plan_from_state(I: Instance, S: State) -> Dict:
    plan = {}
    for f in I.flights:
        if f.is_transp:
            continue
        if f.idx in S.executed:
            plan[f.idx] = S.executed[f.idx]
        else:
            plan[f.idx] = "C"
    return plan


def _commit(I: Instance, S: State, f: int, dep: int, arr: int, tail: str) -> None:
    fo = I.flights[f]
    S.executed[f] = (dep, arr, tail)
    ts = S.tails[tail]
    ts.ap = fo.dest
    ts.ready = arr + I.aircraft[tail].turn
    a = I.aircraft[tail]
    if a.maint and dep < a.maint[1]:
        ts.flown += fo.dur
    S.cap_used[(fo.orig, dep // 60 * 60, "d")] += 1
    S.cap_used[(fo.dest, arr // 60 * 60, "a")] += 1


def plan_to_warm(plan: Dict) -> Dict:
    return {f: (v[0], v[2]) if isinstance(v, tuple) else "C" for f, v in plan.items()}


def shared_extra_deps(I: Instance, occ_plan: Dict) -> Dict[int, Tuple[int, ...]]:
    """Extra departure times shared by every method (audit Stage 2, R-3): the departures of the rule-based OCC plan,
    which warm-starts the full-horizon model.  Giving them to FH *and* to every RH stage keeps one common decision set."""
    return {f: (v[0],) for f, v in occ_plan.items() if isinstance(v, tuple) and I.start <= I.flights[f].dep < I.end}


def night_break(I: Instance) -> int:
    """Time of day (min) at which the longest overnight gap without departures starts."""
    busy = {(f.dep % 1440) // 60 for f in I.flights if not f.is_transp}
    best, best_len = 0, -1
    for h in range(24):
        if h in busy or (h - 1) % 24 not in busy:
            continue
        n = 0
        while (h + n) % 24 not in busy and n < 24:
            n += 1
        if n > best_len:
            best, best_len = h, n
    return best * 60


def full_horizon(I: Instance, p: Params, warm: Optional[Dict] = None, revelation: Optional[Revelation] = None) -> RunResult:
    """Full-horizon model with all information.  With `revelation` (perfect-information reference of the dynamic study,
    R-6) every disruption is known at T0, but a flight departing before the announcement of a reduction or an outage
    is exempt from it, exactly as for the online policies and in the evaluator (decision D5)."""
    t0 = time.time()
    S = initial_state(I, known_all=True)
    if revelation is not None:
        revelation.set_announcements(I, S)
    itins = itinerary_info(I)
    res = solve_window(I, S, I.modifiable, itins, p, final=True, warm=warm)
    for f, (dep, arr, kid) in sorted(res.chosen.items(), key=lambda kv: kv[1][0]):
        _commit(I, S, f, dep, arr, res.tail_of[f])
    S.cancelled |= res.cancelled
    rt = time.time() - t0
    log = [dict(now=I.start, n_flights=len(I.modifiable), obj=res.obj, bound=res.bound, gap=res.gap,
                status=res.status, build=res.build_time, solve=res.solve_time, cols=res.n_cols, rows=res.n_rows,
                lp_status=(res.cert.get("lp") or {}).get("status"), restricted=(res.cert.get("mip") or {}).get("restricted"))]
    out = RunResult(_plan_from_state(I, S), rt, 1, res.solve_time, res.build_time + res.solve_time, log,
                    res.bound, res.obj, res.gap)
    out.cert = res.cert
    return out


def rolling_horizon(I: Instance, p: Params, step: int = 60, lookahead: int = 60, commit: str = "exec",
                    revelation: Optional[Revelation] = None, stage_time_limit: Optional[float] = None) -> RunResult:
    """Generic rolling-horizon policy.

    step       -- re-optimisation period Delta (minutes)
    lookahead  -- optimisation window Lambda >= step (minutes); math.inf = whole remaining horizon
    commit     -- 'exec': decisions become irrevocable when executed (departure < now+step);
                  'stage': thesis-style staging, every flight *scheduled* in the stage is fixed
    p.terminal -- 'none' (myopic) or 'targets' (intermediate positioning targets)
    """
    t0 = time.time()
    rev = revelation or Revelation.static(I)
    S = initial_state(I, known_all=False)
    itins = itinerary_info(I)
    pp = replace(p, time_limit=stage_time_limit or p.time_limit)
    open_f = set(I.modifiable)
    now = I.start
    day_break = night_break(I) if lookahead == "day" else None
    prev_plan: Dict[int, Tuple[int, str]] = {}
    fallbacks = 0
    log = []
    stages = 0
    tot_solve = 0.0
    max_stage = 0.0
    while open_f:
        S.now = now
        rev.apply(I, S, now)
        # disruption cancellations known now are applied immediately
        for f in list(open_f):
            if S.delays.get(f, 0) < 0 and I.flights[f].dep < now + step:
                S.cancelled.add(f)
                open_f.discard(f)
        if lookahead == "day":
            wend = now + ((day_break - now) % 1440 or 1440)
        else:
            wend = now + lookahead
        window = sorted(f for f in open_f if I.flights[f].dep < wend)
        if not window:
            nxt = min(I.flights[f].dep for f in open_f)
            now = max(now + step, (nxt - I.start) // step * step + I.start)
            continue
        final = all(I.flights[f].dep < wend for f in open_f)
        warm = {f: prev_plan[f] for f in window if f in prev_plan} or None
        if warm is not None:
            for f in window:
                warm.setdefault(f, "C")
        res = solve_window(I, S, window, itins, pp, final=final, warm=warm)
        failed = not res.chosen and len(res.cancelled) == len(window) and "ptimal" not in res.status
        stages += 1
        tot_solve += res.solve_time
        max_stage = max(max_stage, res.build_time + res.solve_time)
        log.append(dict(now=now, n_flights=len(window), obj=res.obj, bound=res.bound, gap=res.gap,
                        status=res.status, build=res.build_time, solve=res.solve_time, cols=res.n_cols,
                        rows=res.n_rows, final=final, lp_status=(res.cert.get("lp") or {}).get("status"),
                        restricted=(res.cert.get("mip") or {}).get("restricted")))
        horizon = now + step
        if failed:
            # fallback (audit Stage 2, R-7): (1) re-solve the stage for the flights that can depart in the commit
            # period only, myopically; (2) if that also fails, rule-based restoration of the commit period
            fallbacks += 1
            due = [f for f in window if _can_depart_before(I, S, f, now, horizon, pp)]
            if due:
                res = solve_window(I, S, due, itins, replace(pp, terminal="none"), final=False)
                tot_solve += res.solve_time
            if due and not (not res.chosen and len(res.cancelled) == len(due) and "ptimal" not in res.status):
                log[-1]["fallback"] = dict(kind="commit-period MIP", n_flights=len(due), status=res.status)
            else:
                n_fb = _restore(I, S, due, now, horizon, open_f, pp)
                log[-1]["fallback"] = dict(kind="rule", n_flights=len(due), dispatched=n_fb)
                now = horizon
                continue
        prev_plan = {f: (res.chosen[f][0], res.tail_of[f]) for f in res.chosen}
        # commit decisions
        to_commit = []
        confirmed = []
        for f in window:
            fo = I.flights[f]
            if f in res.chosen:
                dep, arr, kid = res.chosen[f]
                if (commit == "exec" and dep < horizon) or (commit == "stage" and fo.dep < horizon) or final and commit == "stage":
                    to_commit.append((dep, f, arr))
            else:
                # decision D4 (R-9), as in the manuscript: a planned cancellation is confirmed only if the flight has
                # no departure option at or after the stage boundary; otherwise it stays open (tentative)
                if (commit == "exec" and not _can_depart_after(I, S, f, now, horizon, pp)) or \
                        (commit == "stage" and fo.dep < horizon) or (final and commit == "stage"):
                    S.cancelled.add(f)
                    open_f.discard(f)
                    confirmed.append(f)
        log[-1]["confirmed_cancellations"] = confirmed
        for dep, f, arr in sorted(to_commit):
            _commit(I, S, f, dep, arr, res.tail_of[f])
            open_f.discard(f)
        now = horizon
        if now > I.end + 24 * 60:   # safety
            for f in open_f:
                S.cancelled.add(f)
            break
    rt = time.time() - t0
    out = RunResult(_plan_from_state(I, S), rt, stages, tot_solve, max_stage, log)
    out.fallbacks = fallbacks
    return out


def _can_depart_after(I: Instance, S: State, f: int, now: int, horizon: int, p: Params) -> bool:
    return any(t >= horizon for t in departure_set(I.flights[f], S.delays.get(f, 0), now, p))


def _can_depart_before(I: Instance, S: State, f: int, now: int, horizon: int, p: Params) -> bool:
    return S.delays.get(f, 0) < 0 or any(t < horizon for t in departure_set(I.flights[f], S.delays.get(f, 0), now, p))


def _restore(I: Instance, S: State, flights: List[int], now: int, horizon: int, open_f, p: Params) -> int:
    """Rule-based restoration of the commit period [now, horizon) when no stage MIP returns a solution (audit Stage 2,
    R-7; replaces the former greedy dispatch).  Flights in order of scheduled departure; a flight is flown on its
    planned tail if possible, else on the earliest-ready tail of its family at its origin, at the first departure time
    of the common decision set (`departure_set`) before the boundary at which
      the tail is ready, no known outage or maintenance window overlaps [dep, ready) (model rule), the maintenance
      flown-minute budget is not exceeded, a maintenance tail is not moved onto another rotation, and every quota row
      of the departure and arrival cells (decision D5) has room.
    Otherwise the flight stays open if it can still depart after the boundary, else it is cancelled."""
    exec_deps = executed_departures(I, S)
    n = 0

    def room(ap, h, kd, dep):
        return all(cap >= 1 for tau, cap in quota_rows(I, S, ap, h, kd, exec_deps) if tau is None or dep >= tau)

    for f in sorted(flights, key=lambda g: I.flights[g].dep):
        fo = I.flights[f]
        d = S.delays.get(f, 0)
        if d < 0:
            S.cancelled.add(f)
            open_f.discard(f)
            continue
        deps = [t for t in departure_set(fo, d, now, p) if t < horizon]
        fam = I.aircraft[fo.aircraft].family
        cands = [t for t, ts in S.tails.items() if ts.ap == fo.orig and I.aircraft[t].family == fam
                 and I.aircraft[t].range_min >= fo.dur]
        cands.sort(key=lambda t: (t != fo.aircraft, S.tails[t].ready))
        done = False
        for t in cands:
            a, ts = I.aircraft[t], S.tails[t]
            blocks = [(s0, e0) for s0, e0 in S.unavail.get(t, ()) if e0 > now]
            m = a.maint if (a.maint and a.maint[2] > now) else None
            if m:
                blocks.append((m[1], m[2]))
                if t != fo.aircraft and fo.dest != m[0]:
                    continue
            for dep in deps:
                arr = dep + fo.dur
                if dep < ts.ready or any(dep < e0 and arr + a.turn > s0 for s0, e0 in blocks):
                    continue
                if m and dep < m[1] and ts.flown + fo.dur > m[3]:
                    break
                if not (room(fo.orig, dep // 60 * 60, "d", dep) and room(fo.dest, arr // 60 * 60, "a", dep)):
                    continue
                _commit(I, S, f, dep, arr, t)
                exec_deps[(fo.orig, dep // 60 * 60, "d")].append(dep)
                exec_deps[(fo.dest, arr // 60 * 60, "a")].append(dep)
                open_f.discard(f)
                n += 1
                done = True
                break
            if done:
                break
        if not done and not _can_depart_after(I, S, f, now, horizon, p):
            S.cancelled.add(f)
            open_f.discard(f)
    return n


# ----------------------------------------------------------------------------
def occ_rule(I: Instance, max_delay: int = 480, revelation: Optional[Revelation] = None) -> RunResult:
    """Rule-based baseline mimicking manual OCC practice: keep original tails,
    propagate delays, slot each flight into the first hour with free departure
    and arrival capacity, cancel when the delay exceeds `max_delay` (and cancel
    the rest of the rotation until the aircraft is back where it should be)."""
    t0 = time.time()
    S = initial_state(I, known_all=True)
    by_tail = defaultdict(list)
    for f in I.modifiable:
        by_tail[I.flights[f].aircraft].append(f)
    for t in by_tail:
        by_tail[t].sort(key=lambda f: I.flights[f].dep)
    queue = sorted(I.modifiable, key=lambda f: I.flights[f].dep)
    ptr = {t: 0 for t in by_tail}
    # process flights in order of scheduled departure, tail by tail
    events = []
    for f in queue:
        events.append(f)
    blocked_until = {}
    for f in events:
        fo = I.flights[f]
        t = fo.aircraft
        ts = S.tails[t]
        a = I.aircraft[t]
        d = I.flight_disr.get(f, 0)
        if d < 0 or ts.ap != fo.orig:
            S.cancelled.add(f)
            continue
        dep = max(fo.dep + max(d, 0), ts.ready)
        for s, e in I.ac_unavail.get(t, ()):
            if dep < e and dep + fo.dur > s:
                dep = max(dep, e)
        if a.maint:
            ap_m, s_m, e_m, rem = a.maint
            if dep < e_m and dep + fo.dur + a.turn > s_m and dep >= s_m - fo.dur - a.turn:
                dep = max(dep, e_m)
        # capacity-aware slotting
        while dep - fo.dep <= max_delay:
            hd, ha = dep // 60 * 60, (dep + fo.dur) // 60 * 60
            cd = capacity(I, I.alt_cap, fo.orig, hd)[0] - S.cap_used[(fo.orig, hd, "d")]
            ca = capacity(I, I.alt_cap, fo.dest, ha)[1] - S.cap_used[(fo.dest, ha, "a")]
            if cd > 0 and ca > 0:
                break
            dep = (dep // 60 + 1) * 60 if cd <= 0 else dep + 5
        if dep - fo.dep > max_delay:
            S.cancelled.add(f)
            continue
        _commit(I, S, f, dep, dep + fo.dur, t)
    return RunResult(_plan_from_state(I, S), time.time() - t0, 0, 0.0, 0.0)
