"""Stage 2: scenario windows, rotation repair, maintenance, airport capacity, disruptions (substitutes for the
HP-internal PlanSimulator / MaintenanceFileGen / DisruptionsSimulator / AirportCapacitySimulator)."""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .build_base import fleet
from .make_models import load_models, short_of

# wiki 'Create Scenarios': recovery start 14:00 GMT on day 1, end 23:59 on day 3, scenStart 00:00 of day 1.
# (the wiki labels 4 and 6 "Full 4 days" but their dates span 3 days; dates kept, label noted in README)
WIKI_DATES = {2: "2014-01-21", 3: "2014-01-21", 4: "2014-01-24", 5: "2014-02-03", 6: "2014-04-23",
              7: "2014-06-17", 8: "2014-08-18", 9: "2014-09-10", 10: "2014-10-28"}
WIKI_LABEL = {2: "Full 3 days", 3: "Partial 3 days (EMB-120 Brasilia sub-fleet)", 4: "'Full 4 days' (3-day dates)",
              5: "Full 3 days", 6: "'Full 4 days' (3-day dates)", 7: "Full 3 days", 8: "Full 3 days",
              9: "Full 3 days", 10: "Full 3 days"}
REC_START_H = 14


@dataclass
class Spec:
    name: str
    day0: str
    wiki_scenario: Optional[int] = None
    families: Tuple[str, ...] = ()        # tail filter: aircraft families
    models: Tuple[str, ...] = ()          # ... or models
    carriers: Tuple[str, ...] = ()        # ... and/or reporting carriers
    cap_basis: str = "network"            # 'network' (wiki) or 'fleet' (quotas from the instance's own flights)
    note: str = ""
    seed_name: str = ""                   # draw the random streams of another instance (sensitivity twins); default: own name
    extra: bool = False                   # not part of the 16 benchmark instances: skipped by `build --all`


def specs() -> Dict[str, Spec]:
    out = {}
    for s, d in WIKI_DATES.items():
        if s == 3:
            out["US03"] = Spec("US03", d, 3, models=("EMB120",), note="wiki scenario 3: EMB-120 Brasilia tails of scenario 2")
        else:
            out[f"US{s:02d}"] = Spec(f"US{s:02d}", d, s, note=f"wiki scenario {s}: {WIKI_LABEL[s]}")
    sd = [("US02_CRJ700", "2014-01-21", dict(families=("CRJ700",), carriers=("OO",))),
          ("US02_A320", "2014-01-21", dict(families=("A320",), carriers=("UA",))),
          ("US02_B737", "2014-01-21", dict(families=("B737",), carriers=("UA",))),
          ("US02_CRJ200", "2014-01-21", dict(families=("CRJ200",), carriers=("OO",))),
          ("US08_EMB120", "2014-08-18", dict(families=("EMB120",))),
          ("US08_CRJ700", "2014-08-18", dict(families=("CRJ700",), carriers=("OO",))),
          ("US08_A320", "2014-08-18", dict(families=("A320",), carriers=("UA",)))]
    for n, d, kw in sd:
        out[n] = Spec(n, d, None, cap_basis="fleet", note="scale-down sub-fleet (quotas from own flights)", **kw)
    # sensitivity twin of US02_A320 with the wiki's network-basis quotas; same random streams, so only the quota basis differs
    out["US02_A320_NQ"] = Spec("US02_A320_NQ", "2014-01-21", None, families=("A320",), carriers=("UA",), cap_basis="network",
                               note="sensitivity twin of US02_A320: network-basis quotas (same seed stream)", seed_name="US02_A320", extra=True)
    return out


def seed_of(name: str, seed: int) -> int:
    return (zlib.crc32(name.encode()) ^ (seed * 2654435761)) & 0xFFFFFFFF


def window(day0: str) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    s0 = pd.Timestamp(day0)
    return s0, s0 + pd.Timedelta(hours=REC_START_H), s0 + pd.Timedelta(days=2, hours=23, minutes=59)


def aircraft_table() -> pd.DataFrame:
    fl = fleet()
    fl = fl[fl.has_cfg & (fl.cabF + fl.cabB + fl.cabE > 0)].copy()
    M = load_models()
    fl["model"] = fl.Model.map(lambda m: short_of(m)[0])
    fl["family"] = fl.model.map(M.family)
    return fl


def select_flights(base: pd.DataFrame, ac: pd.DataFrame, spec: Spec) -> Tuple[pd.DataFrame, dict]:
    s0, rec, s1 = window(spec.day0)
    st = {}
    w = base[(base.dep >= s0) & (base.dep <= s1) & (base.arr <= s1)]
    st["window_flights_all"] = len(w)
    w = w[w.in_fleet]
    st["window_flights_in_fleet"] = len(w)
    w = w[w.tailnum.isin(ac.index)]
    st["window_flights_with_config"] = len(w)
    w = w.copy()
    w["model"] = w.tailnum.map(ac.model)
    w["family"] = w.tailnum.map(ac.family)
    if spec.families:
        w = w[w.family.isin(spec.families)]
    if spec.models:
        w = w[w.model.isin(spec.models)]
    if spec.carriers:
        w = w[w.carrier.isin(spec.carriers)]
    st["flights_before_repair"] = len(w)
    return w.reset_index(drop=True), st


def repair_rotations(w: pd.DataFrame, turn: Dict[str, int]) -> Tuple[pd.DataFrame, dict]:
    """Per tail keep the longest chain with continuity (arr airport == next dep airport), no overlap and
    turnaround >= model turn.  Replaces PlanSimulator (10 random trials -> exact DP; ties by earliest flights)."""
    keep = np.zeros(len(w), bool)
    w = w.sort_values(["tailnum", "dep", "arr", "fid"], kind="stable").reset_index(drop=True)
    for tail, idx in w.groupby("tailnum", sort=False).indices.items():
        g = w.iloc[idx]
        o, d = g.orig.to_numpy(), g.dest.to_numpy()
        dep = g.dep.to_numpy().astype("datetime64[m]").astype(np.int64)
        arr = g.arr.to_numpy().astype("datetime64[m]").astype(np.int64)
        tn = turn[g.model.iloc[0]]
        n = len(g)
        # objective: retained flights; ties broken towards flights that were actually flown (scheduled-tail records of
        # cancelled flights are the least reliable part of OTP)
        wt = 1.0 + 0.001 * (g.cancelled.to_numpy() == 0)
        best = wt.copy()
        pred = np.full(n, -1)
        for j in range(n):
            for i in range(j):
                if d[i] == o[j] and dep[j] >= arr[i] + tn and best[i] + wt[j] > best[j]:
                    best[j], pred[j] = best[i] + wt[j], i
        j = int(np.argmax(best))     # first maximum
        while j >= 0:
            keep[idx[j]] = True
            j = pred[j]
    st = {"flights_dropped_by_repair": int((~keep).sum()), "flights_after_repair": int(keep.sum())}
    st["repair_drop_rate"] = st["flights_dropped_by_repair"] / max(len(w), 1)
    return w[keep].reset_index(drop=True), st


def assign_flight_numbers(fl: pd.DataFrame) -> pd.DataFrame:
    """Typical-day flight table ids: (orig, dest, dep time-of-day, arr time-of-day, k); k disambiguates clones on one date."""
    fl = fl.copy()
    day = fl.dep.dt.normalize()
    fl["date"] = day
    fl["dep_tod"] = ((fl.dep - day).dt.total_seconds() // 60).astype(int)
    fl["arr_tod"] = fl.dep_tod + fl.dur
    fl = fl.sort_values(["date", "dep", "orig", "dest", "tailnum"], kind="stable")
    fl["k"] = fl.groupby(["date", "orig", "dest", "dep_tod", "dur"]).cumcount()
    key = fl[["orig", "dest", "dep_tod", "arr_tod", "k"]].drop_duplicates().sort_values(
        ["dep_tod", "orig", "dest", "arr_tod", "k"], kind="stable").reset_index(drop=True)
    key["fnum"] = np.arange(1, len(key) + 1)
    fl = fl.merge(key, on=["orig", "dest", "dep_tod", "arr_tod", "k"], how="left")
    return fl.sort_values(["dep", "fnum"], kind="stable").reset_index(drop=True)


def aircraft_for(fl: pd.DataFrame, ac: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """Per-instance aircraft table: ids MODEL#n (n by tail number), start airport = origin of the tail's first flight."""
    first = fl.sort_values("dep", kind="stable").groupby("tailnum").first()
    last = fl.sort_values("arr", kind="stable").groupby("tailnum").last()
    t = pd.DataFrame(index=sorted(first.index))
    t["model"] = ac.model.reindex(t.index)
    t["family"] = ac.family.reindex(t.index)
    t["cfg"] = list(zip(ac.cabF.reindex(t.index), ac.cabB.reindex(t.index), ac.cabE.reindex(t.index)))
    t["orig"] = first.orig.reindex(t.index)
    t["end_ap"] = last.dest.reindex(t.index)
    t["carrier"] = first.carrier.reindex(t.index)
    t["id"] = t.groupby("model").cumcount().add(1).astype(str)
    t["id"] = t.model + "#" + t.id
    for c in ["range_min", "cost_h", "turn", "transit"]:
        t[c] = t.model.map(M[c])
    return t


def gen_maintenance(fl: pd.DataFrame, ac: pd.DataFrame, rec: pd.Timestamp, s0: pd.Timestamp, s1: pd.Timestamp,
                    rng: np.random.Generator, disr: pd.Series, red_hours: set, outage_tails: set,
                    p: float = 0.15) -> pd.DataFrame:
    """Substitute for MaintenanceFileGen: with probability p a tail gets ONE maintenance block of 4-6 h, placed
    in a ground gap >= 8 h at the airport where the tail sleeps, after the recovery start.

    Reachability rule (root cause of an unrecoverable maintenance-location slack found by the FH gate): maintenance is only
    placed on tails whose rotation up to the gap is untouched by the generated disruptions, i.e. no delayed/cancelled flight
    (fixed ones included), no outage, and no dep/arr in an hour with a reduced airport rate.  Capacity quotas are tight (hours
    without observed movement have rate 0), so a disrupted tail can otherwise be unable to reach the maintenance airport in time."""
    rows = []
    for tail, g in fl.sort_values("dep").groupby("tailnum"):
        dep, dest = g.dep.to_numpy(), g.dest.to_numpy()
        # a fixed flight (departs before the recovery start) arrives late by its historical delay
        late = np.where(g.dep.to_numpy() < np.datetime64(rec), g.fid.map(disr).fillna(0).clip(lower=0).to_numpy(), 0)
        arr = g.arr.to_numpy() + late.astype("timedelta64[m]")
        if tail in outage_tails:
            continue
        hit = np.array([r.fid in disr.index or (r.orig, r.dep.floor("h")) in red_hours or (r.dest, r.arr.floor("h")) in red_hours
                        for r in g.itertuples()])
        clean = ~np.maximum.accumulate(hit)
        gaps = [i for i in range(len(g) - 1) if clean[i] and (dep[i + 1] - arr[i]) / np.timedelta64(1, "m") >= 480 and pd.Timestamp(arr[i]) >= rec]
        if gaps and rng.random() < p:
            i = gaps[int(rng.integers(len(gaps)))]
            start = pd.Timestamp(arr[i]) + pd.Timedelta(minutes=60)
            end = start + pd.Timedelta(minutes=int(rng.integers(240, 361)))
            rows.append(dict(tailnum=tail, ap=dest[i], start=start, end=end,
                             remaining=int((start - s0) / pd.Timedelta(minutes=1))))   # wiki: DATEDIFF(min, scenStart, maintStart)
    return pd.DataFrame(rows, columns=["tailnum", "ap", "start", "end", "remaining"])


# ----------------------------------------------------------------------------- airport capacity
def _hour_counts(d: pd.DataFrame, tcol: str, apcol: str) -> pd.Series:
    t = d[tcol]
    return d.assign(_d=t.dt.normalize(), _h=t.dt.hour).groupby([apcol, "_d", "_h"]).size()


def actual_times(f: pd.DataFrame) -> pd.DataFrame:
    """Wiki FlightsforAirportCapacity: actual = planned if ArrDelayMinutes <= 15, else shifted by ArrDelayMinutes."""
    f = f.copy()
    sh = pd.to_timedelta(np.where(f.ArrDelayMinutes.fillna(0) > 15, f.ArrDelayMinutes.fillna(0), 0), unit="m")
    f["adep"] = f.dep + sh
    f["aarr"] = f.arr + sh
    return f


def capacity_table(f: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (cap[ap,hour]->(dep,arr) typical day), planned[(ap,date,hour)] and actual counts, wiki views
    AirportCapacity / scenarioAirportCapacity_V.  d0,d1 are the scenario dates (inclusive)."""
    f = actual_times(f)
    pd_ = _hour_counts(f, "dep", "orig").rename("pd")
    pa_ = _hour_counts(f, "arr", "dest").rename("pa")
    nc = f[f.cancelled == 0]
    ad_ = _hour_counts(nc, "adep", "orig").rename("ad")
    aa_ = _hour_counts(nc, "aarr", "dest").rename("aa")
    for s in (pd_, pa_, ad_, aa_):
        s.index.names = ["ap", "date", "hour"]
    cnt = pd.concat([pd_, pa_, ad_, aa_], axis=1).fillna(0).reset_index()
    cnt = cnt[(cnt.date >= d0) & (cnt.date <= d1)]
    cnt["dep"] = cnt[["pd", "ad"]].max(axis=1)
    cnt["arr"] = cnt[["pa", "aa"]].max(axis=1)
    cap = cnt.groupby(["ap", "hour"])[["dep", "arr"]].max().astype(int)
    return cap, cnt


def capacity_periods(cap: pd.DataFrame, aps: List[str]) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """(dep, arr, start_min, end_min) periods per airport, equal consecutive hours merged; unobserved hours = 0."""
    out = {}
    for ap in aps:
        try:
            c = cap.loc[ap]
        except KeyError:
            c = pd.DataFrame(columns=["dep", "arr"])
        v = [(int(c.dep.get(h, 0)), int(c.arr.get(h, 0))) for h in range(24)]
        per, s = [], 0
        for h in range(1, 25):
            if h == 24 or v[h] != v[s]:
                per.append((v[s][0], v[s][1], s * 60, h * 60))
                s = h
        out[ap] = per
    return out


def cap_at(per: List[Tuple[int, int, int, int]], hour: int) -> Tuple[int, int]:
    for d, a, s, e in per:
        if s <= hour * 60 < e:
            return d, a
    return 0, 0


# ----------------------------------------------------------------------------- disruptions
def flight_disruptions(fl: pd.DataFrame, rec: pd.Timestamp) -> pd.Series:
    """Wiki FlightDisruptions set 0 + the scenario filter (keep externally-caused delays, all cancellations, and
    everything departing before the recovery start).  Returns Series fid -> delay minutes (>0) or -1."""
    canc = (fl.cancelled == 1) & (fl.ccode != "A")
    late = fl.ArrDelayMinutes.fillna(0) > 15
    base = fl[canc | (late & (fl.cancelled == 0))]
    ext = lambda c: (base[c] <= base.WeatherDelay) | (base[c] <= base.NASDelay) | (base[c] <= base.SecurityDelay)  # noqa: E731
    keep = (base.cancelled == 1) | (base.dep < rec) | (ext("CarrierDelay") & ext("LateAircraftDelay"))
    b = base[keep]
    return pd.Series(np.where(b.cancelled == 1, -1, b.ArrDelayMinutes.fillna(0).round().astype(int)), index=b.fid)


def propagate_fixed_delays(fl: pd.DataFrame, disr: pd.Series, rec: pd.Timestamp, turn: Dict[str, int]) -> Tuple[pd.Series, int]:
    """Fixed flights (departing before the recovery start) are executed with their historical delay.  The wiki applies
    ArrDelayMinutes as the departure delay, which can break the turnaround between two fixed flights of one tail when the
    successor made up time en route.  Propagate delays down each tail's fixed chain so that executed history is
    consistent (substitute; counted in the build stats)."""
    d = disr.copy().astype(float)
    n_new = 0
    for tail, g in fl[fl.dep < rec].sort_values("dep", kind="stable").groupby("tailnum"):
        tn = turn[g.model.iloc[0]]
        prev = None
        for r in g.itertuples():
            cur = max(float(d.get(r.fid, 0)), 0.0)
            if prev is not None:
                slack = (r.dep - prev.arr).total_seconds() / 60 - tn
                need = max(float(d.get(prev.fid, 0)), 0.0) - slack
                if need > cur:
                    n_new += cur <= 0
                    d.loc[r.fid] = np.ceil(need)
            prev = r
    return d.astype(int), n_new


def aircraft_outages(fl: pd.DataFrame, rec: pd.Timestamp, s1: pd.Timestamp, maint: pd.DataFrame,
                     disr: pd.Series) -> pd.DataFrame:
    """Substitute for AircraftDisruptions: carrier-caused cancellations (code 'A', absent from the flight-disruption
    set) are read as aircraft outages: tail unavailable from the cancelled flight's planned departure to its next
    planned departure (or the scenario end), only for flights departing after the recovery start."""
    rows = []
    mt = {r.tailnum: (r.start, r.end) for r in maint.itertuples()}
    for tail, g in fl.sort_values("dep").groupby("tailnum"):
        dep = g.dep.tolist()
        for i, r in enumerate(g.itertuples()):
            if r.cancelled == 1 and r.ccode == "A" and r.dep >= rec:
                e = dep[i + 1] if i + 1 < len(dep) else s1
                st_ = r.dep
                if i > 0:                      # the tail may still be flying its (delayed) fixed previous flight
                    pf = g.iloc[i - 1]
                    if pf.dep < rec:
                        st_ = max(st_, pf.arr + pd.Timedelta(minutes=max(float(disr.get(pf.fid, 0)), 0)))
                if st_ >= e:
                    continue
                if tail in mt and st_ < mt[tail][1] and e > mt[tail][0]:
                    continue
                if rows and rows[-1][0] == tail and rows[-1][2] >= st_:
                    rows[-1][2] = e
                else:
                    rows.append([tail, st_, e])
    return pd.DataFrame(rows, columns=["tailnum", "start", "end"])


def airport_reductions(net: pd.DataFrame, cnt: pd.DataFrame, cap: pd.DataFrame, rec: pd.Timestamp,
                       s1: pd.Timestamp, aps: set, basis: str) -> pd.DataFrame:
    """Substitute for AirportReduction: (airport, hour) after the recovery start where network planned movements
    exceed actual (non-cancelled, un-shifted) movements by >= max(3, 30 % of planned) get a reduced rate."""
    rows = []
    c = cnt[cnt.ap.isin(aps)].copy()
    c["ts"] = c.date + pd.to_timedelta(c.hour, unit="h")
    c = c[(c.ts >= rec) & (c.ts <= s1)]
    for kind, p_, a_ in (("dep", "pd", "ad"), ("arr", "pa", "aa")):
        r = c[(c[p_] - c[a_]) >= np.maximum(3, 0.3 * c[p_])]
        for x in r.itertuples():
            base = int(cap.at[(x.ap, x.hour), kind]) if (x.ap, x.hour) in cap.index else 0
            planned, actual = getattr(x, p_), getattr(x, a_)
            rate = int(min(base, actual)) if basis == "network" else int(np.floor(base * actual / max(planned, 1)))
            if rate < base:
                rows.append((x.ap, x.ts, kind, rate))
    return pd.DataFrame(rows, columns=["ap", "ts", "kind", "rate"])
