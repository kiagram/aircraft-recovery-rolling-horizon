"""ROADEF-2009-consistent cost components used by both the MIP and the evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .instance import Instance, Itinerary

CABIN_RANK = {"F": 0, "B": 1, "E": 2}
TYPE_RANK = {"P": -1, "D": 0, "C": 1, "I": 2}


def meal_threshold(trip_dur: int) -> int:
    if trip_dur < 120:
        return 120
    if trip_dur < 270:
        return 180
    return 240


def compensation(trip_dur: int) -> float:
    if trip_dur < 120:
        return 250.0
    if trip_dur < 270:
        return 400.0
    return 600.0


@dataclass
class ItinInfo:
    it: Itinerary
    ref_cabin: str
    itype: str
    trip_dur: int
    sched_arr: int
    delay_rate: float        # beta * euro / pax / minute
    cancel_cost_out: float   # alpha*(price+compensation) + beta*cancel_out, per pax
    cancel_cost_in: float    # same, with inbound / started-trip disutility

    def delay_cost(self, delay: float) -> float:
        """Cost per passenger of an arrival delay (minutes) at the final destination."""
        if delay <= 0:
            return 0.0
        a = 0.0
        if delay > meal_threshold(self.trip_dur):
            a += 15.0
        if delay > 300:
            a += 60.0
        if self._regime == "eu261" and delay >= 180:          # Reg. (EC) 261/2004 delay compensation
            a += compensation(self.trip_dur)
        elif self._regime == "eu2025":                          # Council position of 5 June 2025
            if self.trip_dur < 270 and delay >= 240:
                a += 300.0
            elif self.trip_dur >= 270 and delay >= 360:
                a += 500.0
        return self.delay_rate * delay + self._alpha * a

    _alpha: float = 1.0
    _regime: str = "roadef"


def itinerary_info(I: Instance) -> List[ItinInfo]:
    alpha, beta, _ = I.weights
    out = []
    for it in I.itineraries:
        legs = [I.flights[f] for f, _ in it.legs]
        ref = min((c for _, c in it.legs), key=lambda c: CABIN_RANK[c])
        types = [l.ftype if l.ftype != "P" else "D" for l in legs]
        itype = max(types, key=lambda t: TYPE_RANK[t])
        trip = sum(l.dur for l in legs)
        rate = beta * I.delay_cost.get((ref, itype), 0.0)
        base = alpha * (it.price + compensation(trip))
        cout = base + beta * I.cancel_out.get((ref, itype), 0.0)
        cin = base + beta * I.cancel_in.get((ref, itype), 0.0)
        info = ItinInfo(it, ref, itype, trip, legs[-1].arr, rate, cout, cin)
        info._alpha = alpha
        info._regime = getattr(I, "regime", "roadef")
        out.append(info)
    return out


def flight_loads(I: Instance) -> Dict[int, Tuple[int, int, int]]:
    """Booked passengers per flight and cabin (F, B, E)."""
    load: Dict[int, List[int]] = {}
    for it in I.itineraries:
        for f, c in it.legs:
            v = load.setdefault(f, [0, 0, 0])
            v[CABIN_RANK[c]] += it.count
    return {k: tuple(v) for k, v in load.items()}


def spill(load: Tuple[int, int, int], config: Tuple[int, int, int]) -> int:
    """Passengers that cannot be seated (upgrades into free higher cabins allowed)."""
    free_hi = 0
    sp = 0
    for cab in range(3):
        cap = config[cab] if config[cab] >= 0 else 10**6
        demand = load[cab]
        seats = cap + free_hi
        if demand > seats:
            sp += demand - seats
            free_hi = 0
        else:
            # remaining seats in higher cabins can absorb lower-cabin demand
            free_hi = seats - demand
    return sp
