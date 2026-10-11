"""Stage 3: passenger itineraries (replaces HP-internal ItineraryGenerator / AutomatedPassengerAllocator /
ItineraryFixer; the original choice parameters 'choiceparams.properties' are not public).

Documented assumptions (see us2014/README.md):
 * route demand: DB1B coupon trips (1 coupon = non-stop trip, 2 coupons without trip break in between = one-stop trip),
   operating carriers UA/EV/OO; monthly demand = DB1B quarterly pax x clip(T-100 monthly / DB1B quarterly, 2, 5)
   (default 3.33 = x10 sample rate / 3 months), spread evenly over the days of the month;
 * candidates: non-stop flights and 2-leg pairs with 30 <= connection <= 180 min, a carrier pair seen in DB1B;
 * allocation: multinomial logit over candidates, then iterative proportional fitting so that flight loads match
   T-100 load factor x seats (cap 95 %); integer rounding with seeded dithering; no cabin is ever overbooked;
 * price = 100 + total miles / 5 (all legs; wiki: first leg only), cabin multipliers E 1.0, B 1.5, F 2.0.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

LF_CAP = 0.95
MULT = {"E": 1.0, "B": 1.5, "F": 2.0}
TOP_K = 6          # candidates kept per (route, date)


def _scales(one: pd.DataFrame, seg: pd.DataFrame, t100: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Wiki MIN_ALPHA_S: per (month, carrier, O, D) T100/DB1B scaling factor clipped to [2, 5]; default 3.33."""
    if t100 is None:
        return pd.DataFrame(columns=["month", "carrier", "orig", "dest", "sf"])
    sq = seg.groupby(["q", "carrier", "orig", "dest"]).pax.sum().rename("db1b").reset_index()
    t = t100.copy()
    t["q"] = (t.month - 1) // 3 + 1
    m = t.merge(sq, on=["q", "carrier", "orig", "dest"])
    m["sf"] = (m.pax / m.db1b).clip(2, 5)
    return m[["month", "carrier", "orig", "dest", "sf"]]


def generate(fl: pd.DataFrame, ns: pd.DataFrame, one: pd.DataFrame, seg: pd.DataFrame, t100: Optional[pd.DataFrame],
             rng: np.random.Generator) -> Tuple[pd.DataFrame, dict]:
    fl = fl.reset_index(drop=True)
    n = len(fl)
    fl["month"] = fl.dep.dt.month
    fl["q"] = (fl.month - 1) // 3 + 1
    seats = fl[["cabF", "cabB", "cabE"]].to_numpy(float)
    tot = seats.sum(1)

    # ---- flight load targets from T-100 (pax / seats per carrier, O, D, month)
    if t100 is not None:
        t = t100.assign(lf=(t100.pax / t100.seats.clip(lower=1)).clip(0.2, LF_CAP))
        f = fl.merge(t[["month", "carrier", "orig", "dest", "lf"]], on=["month", "carrier", "orig", "dest"], how="left")
        cm = t.groupby(["month", "carrier"]).apply(lambda g: g.pax.sum() / max(g.seats.sum(), 1), include_groups=False
                                                   ).clip(0.2, LF_CAP).rename("lf_avg").reset_index()
        f = f.merge(cm, on=["month", "carrier"], how="left")
        lf = f.lf.fillna(f.lf_avg).fillna(0.8).to_numpy()
    else:
        lf = np.full(n, 0.8)
    target = lf * tot

    # ---- connecting share per segment (DB1B): pax riding the segment as part of a one-stop trip / all coupon pax
    segq = seg.groupby(["q", "carrier", "orig", "dest"]).pax.sum().rename("tot").reset_index()
    c1 = one.groupby(["q", "c1", "orig", "conn"]).pax.sum().rename("p1").reset_index().rename(
        columns={"c1": "carrier", "conn": "dest"})
    c2 = one.groupby(["q", "c2", "conn", "dest"]).pax.sum().rename("p2").reset_index().rename(
        columns={"c2": "carrier", "conn": "orig"})
    cs = segq.merge(c1, on=["q", "carrier", "orig", "dest"], how="left").merge(
        c2, on=["q", "carrier", "orig", "dest"], how="left").fillna(0)
    cs["cs"] = ((cs.p1 + cs.p2) / cs.tot.clip(lower=1)).clip(0, 0.9)
    f = fl.merge(cs[["q", "carrier", "orig", "dest", "cs"]], on=["q", "carrier", "orig", "dest"], how="left")
    csf = f.cs.fillna(0.3).to_numpy()

    # ---- connection candidates
    fl["arr_m"] = fl.arr.to_numpy().astype("datetime64[m]").astype(np.int64)
    fl["dep_m"] = fl.dep.to_numpy().astype("datetime64[m]").astype(np.int64)
    A = fl[["orig", "dest", "carrier", "arr_m", "dep_m", "q", "month"]].reset_index().rename(
        columns={"index": "i", "orig": "O", "dest": "C", "carrier": "c1", "arr_m": "a_arr", "dep_m": "a_dep"})
    B = fl[["orig", "dest", "carrier", "dep_m", "arr_m"]].reset_index().rename(
        columns={"index": "j", "orig": "C", "dest": "D", "carrier": "c2", "dep_m": "b_dep", "arr_m": "b_arr"})
    B["hb"] = B.b_dep // 60
    pieces = []
    lo = (A.a_arr + 30) // 60
    hi = (A.a_arr + 180) // 60
    for k in range(int((hi - lo).max()) + 1):
        a = A[lo + k <= hi].copy()
        a["hb"] = lo[a.index] + k
        pieces.append(a.merge(B, on=["C", "hb"]))
    P = pd.concat(pieces, ignore_index=True)
    gap = P.b_dep - P.a_arr
    P = P[(gap >= 30) & (gap <= 180) & (P.D != P.O)].copy()
    P["gap"] = P.b_dep - P.a_arr
    # carrier pair / route must exist in DB1B one-stop demand for that quarter
    routes = one.rename(columns={"orig": "O", "conn": "C", "dest": "D", "pax": "dpax"})
    P = P.merge(routes, on=["q", "O", "C", "D", "c1", "c2"], how="inner")
    sc = _scales(one, seg, t100)
    if len(sc):
        s1 = sc.rename(columns={"carrier": "c1", "orig": "O", "dest": "C", "sf": "s1"})
        s2 = sc.rename(columns={"carrier": "c2", "orig": "C", "dest": "D", "sf": "s2"})
        P = P.merge(s1, on=["month", "c1", "O", "C"], how="left").merge(s2, on=["month", "c2", "C", "D"], how="left")
        P["scale"] = P[["s1", "s2"]].min(axis=1).fillna(10 / 3)
    else:
        P["scale"] = 10 / 3
    # logit utility: elapsed time, connection penalty, night departure
    ph = ((P.a_dep // 60) % 24)
    P["u"] = -0.012 * (P.b_arr - P.a_dep) - 0.6 - 0.5 * ((ph < 5)).astype(float)
    P["date1"] = pd.to_datetime(P.a_dep * 60, unit="s").dt.normalize()
    P["rid"] = P.groupby(["date1", "O", "C", "D", "c1", "c2"]).ngroup()
    P = P.sort_values(["rid", "u"], ascending=[True, False])
    P = P[P.groupby("rid").cumcount() < TOP_K].copy()
    P["w"] = np.exp(P.u - P.groupby("rid").u.transform("max"))
    dim = P.date1.dt.days_in_month
    P["x"] = P.dpax * P.scale / dim * P.w / P.groupby("rid").w.transform("sum")

    # ---- itinerary arrays: non-stop (one per flight) + connections
    l1 = np.concatenate([np.arange(n), P.i.to_numpy()])
    l2 = np.concatenate([np.full(n, -1), P.j.to_numpy()])
    x = np.concatenate([target * (1 - csf), P.x.to_numpy()])
    x = np.maximum(x, 1e-6)
    has2 = l2 >= 0
    legs_n = 1 + has2
    l2s = np.where(has2, l2, 0)
    for it in range(50):
        L = np.bincount(l1, x, n) + np.bincount(l2s[has2], x[has2], n)
        r = np.clip(target / np.maximum(L, 1e-9), 0.05, 20)
        lr = np.log(r)
        g = (lr[l1] + np.where(has2, lr[l2s], 0)) / legs_n
        x = x * np.exp(0.8 * g)
    L = np.bincount(l1, x, n) + np.bincount(l2s[has2], x[has2], n)
    ratio = np.max(np.abs(L - target) / np.maximum(target, 1))
    # ---- integer rounding (seeded dithering) and cabin split
    k = np.floor(x + rng.random(len(x))).astype(int)
    sF = np.minimum(seats[l1, 0] / np.maximum(tot[l1], 1), np.where(has2, seats[l2s, 0] / np.maximum(tot[l2s], 1), 1))
    sB = np.minimum(seats[l1, 1] / np.maximum(tot[l1], 1), np.where(has2, seats[l2s, 1] / np.maximum(tot[l2s], 1), 1))
    # premium cabins are open only if every leg has seats there
    sF = np.where((seats[l1, 0] > 0) & (~has2 | (seats[l2s, 0] > 0)), sF, 0)
    sB = np.where((seats[l1, 1] > 0) & (~has2 | (seats[l2s, 1] > 0)), sB, 0)
    nF = np.floor(k * sF + rng.random(len(k))).astype(int)
    nB = np.floor(k * sB + rng.random(len(k))).astype(int)
    nF = np.minimum(nF, k)
    nB = np.minimum(nB, k - nF)
    nE = k - nF - nB
    cnt = {"F": nF, "B": nB, "E": nE}
    for ci, c in enumerate("FBE"):         # never overbook a cabin
        for _ in range(3):
            cap = seats[:, ci]
            Lc = np.bincount(l1, cnt[c], n) + np.bincount(l2s[has2], cnt[c][has2], n)
            gf = np.minimum(1.0, cap / np.maximum(Lc, 1e-9))
            fac = np.minimum(gf[l1], np.where(has2, gf[l2s], 1.0))
            cnt[c] = np.floor(cnt[c] * fac + 1e-9).astype(int)
    # ---- assemble rows (one itinerary per (flight set, cabin))
    dist1 = fl.dist.to_numpy()[l1] + np.where(has2, fl.dist.to_numpy()[l2s], 0)
    out = []
    fn, dt = fl.fnum.to_numpy(), fl.date.dt.strftime("%d/%m/%y").to_numpy()
    for c in "FBE":
        m = cnt[c] > 0
        out.append(pd.DataFrame({"i": l1[m], "j": l2[m], "cabin": c, "count": cnt[c][m],
                                 "price": (100 + dist1[m] / 5) * MULT[c]}))
    it = pd.concat(out, ignore_index=True).sort_values(["i", "j", "cabin"], kind="stable").reset_index(drop=True)
    it["id"] = np.arange(1, len(it) + 1)
    it["type"] = "A"
    it["f1"], it["d1"] = fn[it.i], dt[it.i]
    j = it.j.to_numpy()
    it["f2"] = np.where(j >= 0, fn[np.maximum(j, 0)], 0)
    it["d2"] = np.where(j >= 0, dt[np.maximum(j, 0)], "")
    st = dict(itineraries=len(it), pax=int(it["count"].sum()), nonstop=int((it.j < 0).sum()),
              two_leg=int((it.j >= 0).sum()), ipf_max_rel_err=float(ratio), cand_pairs=len(P),
              mean_lf=float((np.bincount(l1, k, n) + np.bincount(l2s[has2], k[has2], n)).sum() / max(tot.sum(), 1)))
    return it, st
