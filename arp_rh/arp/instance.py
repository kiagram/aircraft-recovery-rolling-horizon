"""Parser for ROADEF 2009 challenge instances (Palpant et al., 2009).

All times are converted to absolute minutes measured from 00:00 of the
earliest date appearing in the instance (recovery window or rotations).
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CABINS = ("F", "B", "E")


def _lines(path: str) -> List[List[str]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="latin-1") as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith("%"):
                continue
            if s.startswith("#"):
                break
            rows.append(s.split())
    return rows


def _date(s: str) -> _dt.date:
    d, m, y = s.split("/")
    return _dt.date(2000 + int(y), int(m), int(d))


def _time(s: str) -> int:
    """'hh:mm', 'hh:mm+1' or 'hh:mm-1' -> minutes (may be <0 or >=1440)."""
    shift = 0
    if s.endswith("+1"):
        shift, s = 1440, s[:-2]
    elif s.endswith("-1"):
        shift, s = -1440, s[:-2]
    h, m = s.split(":")
    return int(h) * 60 + int(m) + shift


@dataclass
class Aircraft:
    id: str
    model: str
    family: str
    config: Tuple[int, int, int]
    range_min: int
    cost_h: float
    turn: int
    transit: int
    orig: str
    maint: Optional[Tuple[str, int, int, int]] = None  # airport, start, end, remaining flight minutes

    @property
    def is_transp(self) -> bool:
        return self.family == "TranspCom" or self.model == "TranspCom"

    @property
    def kind(self) -> Tuple[str, Tuple[int, int, int]]:
        return (self.model, self.config)


@dataclass
class Flight:
    idx: int
    num: int
    date: _dt.date
    orig: str
    dest: str
    dep: int
    arr: int
    prev: Optional[int]      # index of preceding leg (multi-leg flight)
    aircraft: str            # aircraft in the original rotation
    ftype: str               # D / C / I / P
    is_transp: bool = False

    @property
    def dur(self) -> int:
        return self.arr - self.dep


@dataclass
class Itinerary:
    id: int
    kind: str                # 'A' outbound, 'R' inbound
    price: float
    count: int
    legs: List[Tuple[int, str]]   # (flight idx, cabin)


@dataclass
class Instance:
    name: str
    day0: _dt.date
    start: int
    end: int
    aircraft: Dict[str, Aircraft]
    flights: List[Flight]
    itineraries: List[Itinerary]
    positions: Dict[str, List[Tuple[str, Tuple[int, int, int], int]]]
    dist: Dict[Tuple[str, str], Tuple[int, str]]
    base_cap: Dict[str, List[Tuple[int, int, int, int]]]  # airport -> [(start,end,dep,arr)] typical day
    alt_cap: Dict[str, List[Tuple[int, int, int, int]]]   # airport -> [(abs start, abs end, dep, arr)]
    flight_disr: Dict[int, int]                            # flight idx -> delay (>0) or -1 cancel
    ac_unavail: Dict[str, List[Tuple[int, int]]]
    delay_cost: Dict[Tuple[str, str], float]
    cancel_out: Dict[Tuple[str, str], float]
    cancel_in: Dict[Tuple[str, str], float]
    downgrade: Dict[Tuple[str, str, str], float]
    pos_pen: Tuple[float, float, float]                    # family, model, config
    weights: Tuple[float, float, float]                    # alpha, beta, gamma
    airports: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def cap(self, airport: str, hour_start: int) -> Tuple[int, int]:
        """(dep, arr) capacity of the one-hour interval [hour_start, hour_start+60)."""
        for s, e, d, a in self.alt_cap.get(airport, ()):
            if s <= hour_start < e:
                return d, a
        tod = hour_start % 1440
        for s, e, d, a in self.base_cap.get(airport, ()):
            ee = e if e > s else 1440
            if s <= tod < ee:
                return d, a
        return 10**6, 10**6  # airport without data: unconstrained

    @property
    def modifiable(self) -> List[int]:
        return [f.idx for f in self.flights if not f.is_transp and self.start <= f.dep < self.end]


def load_instance(folder: str, name: Optional[str] = None) -> Instance:
    j = lambda fn: os.path.join(folder, fn)
    cfg = _lines(j("config.csv"))
    rot = _lines(j("rotations.csv"))
    dates = [_date(cfg[0][0]), _date(cfg[0][2])] + [_date(r[1]) for r in rot]
    day0 = min(dates)
    absd = lambda ds: (_date(ds) - day0).days * 1440
    start = absd(cfg[0][0]) + _time(cfg[0][1])
    end = absd(cfg[0][2]) + _time(cfg[0][3])

    def triples(row, n):
        out = {}
        for k in range(0, len(row), n):
            chunk = row[k:k + n]
            if len(chunk) == n:
                out[tuple(chunk[:-1])] = float(chunk[-1])
        return out

    delay_cost = triples(cfg[1], 3)
    cancel_out = triples(cfg[2], 3)
    cancel_in = triples(cfg[3], 3)
    downgrade = triples(cfg[4], 4)
    pos_pen = tuple(float(v) for v in cfg[5][:3])
    weights = tuple(float(v) for v in cfg[6][:3]) if len(cfg) > 6 else (1.0, 1.0, 1.0)

    aircraft: Dict[str, Aircraft] = {}
    for r in _lines(j("aircraft.csv")):
        cf = tuple(int(v) for v in r[3].split("/"))
        maint = None
        if r[9] != "NULL":
            p = r[9].split("-")
            maint = (p[0], absd(p[1]) + _time(p[2]), absd(p[3]) + _time(p[4]), int(p[5]))
        aircraft[r[0]] = Aircraft(r[0], r[1], r[2], cf, int(r[4]), float(r[5]), int(r[6]), int(r[7]), r[8], maint)

    dist = {(r[0], r[1]): (int(r[2]), r[3]) for r in _lines(j("dist.csv"))}
    ftab = {int(r[0]): r for r in _lines(j("flights.csv"))}

    flights: List[Flight] = []
    key2idx: Dict[Tuple[int, str], int] = {}
    by_ac: Dict[str, List[int]] = {}
    for r in rot:
        num, ds, ac = int(r[0]), r[1], r[2]
        fr = ftab[num]
        dep = absd(ds) + _time(fr[3])
        arr = absd(ds) + _time(fr[4])
        if arr < dep:  # safety for inconsistent +1 flags
            arr += 1440
        ftype = dist.get((fr[1], fr[2]), (0, "D"))[1]
        f = Flight(len(flights), num, _date(ds), fr[1], fr[2], dep, arr, None, ac, ftype,
                   aircraft[ac].is_transp if ac in aircraft else False)
        key2idx[(num, ds)] = f.idx
        flights.append(f)
        by_ac.setdefault(ac, []).append(f.idx)
    # multi-leg links: previous leg = preceding flight of same aircraft with number PrevFlight
    for ac, lst in by_ac.items():
        lst.sort(key=lambda i: flights[i].dep)
        for a, b in zip(lst, lst[1:]):
            pn = int(ftab[flights[b].num][5])
            if pn and flights[a].num == pn:
                flights[b].prev = a

    itins: List[Itinerary] = []
    for r in _lines(j("itineraries.csv")):
        legs = []
        ok = True
        for k in range(4, len(r), 3):
            key = (int(r[k]), r[k + 1])
            if key not in key2idx:
                ok = False
                break
            legs.append((key2idx[key], r[k + 2]))
        if ok and legs:
            itins.append(Itinerary(int(r[0]), r[1], float(r[2]), int(r[3]), legs))

    positions = {}
    for r in _lines(j("position.csv")):
        # each entry is a number of required aircraft; an (airport, model, configuration) key that occurs twice (B/XB
        # instances) requires the sum (ROADEF spec 5.3; audit Stage 2, F-18: the evaluator kept the last entry, the
        # model the largest one at configuration level and the sum at model/family level)
        agg = {}
        for k in range(1, len(r) - 2, 3):
            key = (r[k], tuple(int(v) for v in r[k + 1].split("/")))
            agg[key] = agg.get(key, 0) + int(r[k + 2])
        positions[r[0]] = [(m, c, n) for (m, c), n in agg.items()]

    base_cap = {}
    for r in _lines(j("airports.csv")):
        lst = []
        for k in range(1, len(r), 4):
            if k + 3 < len(r):
                s, e = _time(r[k + 2]), _time(r[k + 3])
                if e == 0:
                    e = 1440
                if s == e == 1440:
                    s = 0
                lst.append((s, e, int(r[k]), int(r[k + 1])))
        base_cap[r[0]] = lst

    alt_cap: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for r in _lines(j("alt_airports.csv")):
        s = absd(r[1]) + _time(r[2])
        e = absd(r[3]) + _time(r[4])
        alt_cap.setdefault(r[0], []).append((s, e, int(r[5]), int(r[6])))

    flight_disr = {}
    for r in _lines(j("alt_flights.csv")):
        key = (int(r[0]), r[1])
        if key in key2idx:
            flight_disr[key2idx[key]] = int(r[2])

    ac_unavail: Dict[str, List[Tuple[int, int]]] = {}
    for r in _lines(j("alt_aircraft.csv")):
        ac_unavail.setdefault(r[0], []).append((absd(r[1]) + _time(r[2]), absd(r[3]) + _time(r[4])))

    airports = sorted(set(base_cap) | {f.orig for f in flights} | {f.dest for f in flights})
    return Instance(name or os.path.basename(os.path.normpath(folder)), day0, start, end, aircraft, flights,
                    itins, positions, dist, base_cap, alt_cap, flight_disr, ac_unavail, delay_cost,
                    cancel_out, cancel_in, downgrade, pos_pen, weights, airports)


INSTANCE_DIRS = {
    **{f"A{i:02d}": f"A{i:02d}_60885{70 if i <= 5 else 90}" for i in range(1, 11)},
    **{f"B{i:02d}": (f"SetB1/B_{i:02d}" if i <= 5 else f"SetB2/B_{i:02d}") for i in range(1, 11)},
    **{f"XA{i:02d}": f"Copy of Instance_X/XA{i:02d}_DATA" for i in range(1, 5)},
    **{f"XB{i:02d}": f"Copy of Instance_X/XB{i:02d}_DATA" for i in range(1, 5)},
    **{f"X{i:02d}": f"Copy of Instance_X/X{i:02d}_DATA" for i in range(1, 5)},
}

INSTANCE_DIRS.update({n: f"../us2014/instances/{n}" for n in [f"US{i:02d}" for i in range(2, 11)] + ["US02_CRJ700", "US02_A320", "US02_B737", "US02_CRJ200", "US08_EMB120", "US08_CRJ700", "US08_A320", "US02_A320_NQ"]})  # US2014 (rebuilt), see us2014/README.md

def load(name: str, root: str) -> Instance:
    return load_instance(os.path.join(root, INSTANCE_DIRS[name]), name)
