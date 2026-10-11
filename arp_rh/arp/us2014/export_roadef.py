"""Stage 4: write a ROADEF-2009-format instance folder readable by arp.instance.load_instance."""
from __future__ import annotations

import hashlib
import os
from typing import Dict

import numpy as np
import pandas as pd

CONFIG_TAIL = [  # penalty block: ROADEF 2009 values (as in A01), P1/P2/P3 = 20000/5000/1000, alpha=beta=gamma=1
    "F D 1.25 F C 1.25 F I 1.25 B D 0.8 B C 0.85 B I 0.9 E D 0.05 E C 0.15 E I 0.25",
    "F D 2500.0 F C 2750.0 F I 3000.0 B D 1500.0 B C 1750.0 B I 2000.0 E D 250.0 E C 600.0 E I 1000.0",
    "F D 7500.0 F C 8250.0 F I 9000.0 B D 4500.0 B C 5250.0 B I 6000.0 E D 750.0 E C 1500.0 E I 3000.0",
    "F B D 150.0 F B C 400.0 F B I 750.0 F E D 200.0 F E C 500.0 F E I 1500.0 B E D 150.0 B E C 400.0 B E I 750.0",
    "20000.0 5000.0 1000.0",
    "1.0 1.0 1.0",
]


def d(ts: pd.Timestamp) -> str:
    return ts.strftime("%d/%m/%y")


def t(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def tod(m: int) -> str:
    return f"{(m % 1440) // 60:02d}:{m % 60:02d}" + ("+1" if m >= 1440 else "")


def flight_type(dist_miles: float, rule: str) -> str:
    if rule == "wiki":
        return "C"                    # wiki: C if both airports in the same country; OTP is US-domestic only
    return "D" if dist_miles < 1000 else ("C" if dist_miles < 3000 else "I")


def write(folder: str, *, rec_start: pd.Timestamp, rec_end: pd.Timestamp, fl: pd.DataFrame, ac: pd.DataFrame,
          maint: pd.DataFrame, positions: Dict[str, list], base_cap: Dict[str, list], alt_ap: list,
          alt_fl: pd.Series, outages: pd.DataFrame, itins: pd.DataFrame, type_rule: str = "wiki") -> Dict[str, str]:
    os.makedirs(folder, exist_ok=True)
    w = lambda fn, lines: open(os.path.join(folder, fn), "w", newline="\n").write("\n".join(lines) + "\n#\n")  # noqa: E731
    w("config.csv", [f"{d(rec_start)} {t(rec_start)} {d(rec_end)} {t(rec_end)}"] + CONFIG_TAIL)
    mm = {r.tailnum: r for r in maint.itertuples()}
    lines = []
    for tail, a in ac.sort_values("id", key=lambda s: s.map(lambda x: (x.split("#")[0], int(x.split("#")[1])))).iterrows():
        mt = "NULL"
        if tail in mm:
            r = mm[tail]
            mt = f"{r.ap}-{d(r.start)}-{t(r.start)}-{d(r.end)}-{t(r.end)}-{r.remaining}"
        lines.append(f"{a.id} {a.model} {a.family} {a.cfg[0]}/{a.cfg[1]}/{a.cfg[2]} {int(a.range_min)} {a.cost_h:.1f} "
                     f"{int(a.turn)} {int(a.transit)} {a.orig} {mt}")
    w("aircraft.csv", lines)
    w("tails.csv", [f"{a.id} {tail} {a.carrier}" for tail, a in ac.iterrows()])      # extra: id -> N-number (ignored by loader)
    fk = fl.drop_duplicates("fnum").sort_values("fnum")
    w("flights.csv", [f"{r.fnum} {r.orig} {r.dest} {tod(r.dep_tod)} {tod(r.arr_tod)} 0" for r in fk.itertuples()])
    idm = ac.id
    w("rotations.csv", [f"{r.fnum} {d(r.date)} {idm[r.tailnum]}" for r in fl.sort_values(["dep", "fnum"]).itertuples()])
    pr = fl.groupby(["orig", "dest"]).dur.mean().round().astype(int)
    dm = fl.groupby(["orig", "dest"]).dist.mean()
    w("dist.csv", [f"{o} {de} {v} {flight_type(dm[(o, de)], type_rule)}" for (o, de), v in pr.items()])
    aps = sorted(base_cap)
    w("airports.csv", [ap + "".join(f" {dd} {aa} {tod(s)} {tod(e) if e < 1440 else '00:00'}" for dd, aa, s, e in base_cap[ap])
                       for ap in aps])
    w("position.csv", [ap + "".join(f" {m} {c[0]}/{c[1]}/{c[2]} {n}" for (m, c), n in sorted(positions[ap].items())) + " "
                       for ap in sorted(positions)])
    w("alt_aircraft.csv", [f"{idm[r.tailnum]} {d(r.start)} {t(r.start)} {d(r.end)} {t(r.end)}" for r in outages.itertuples()])
    w("alt_airports.csv", [f"{ap} {d(s)} {t(s)} {d(e)} {t(e)} {dd} {aa}" for ap, s, e, dd, aa in alt_ap])
    fdl = fl.set_index("fid")
    w("alt_flights.csv", [f"{fdl.at[fid, 'fnum']} {d(fdl.at[fid, 'date'])} {int(v)}" for fid, v in alt_fl.items()])
    w("itineraries.csv", [f"{r.id} {r.type} {r.price:.2f} {r.count} {r.f1} {r.d1} {r.cabin}" +
                          (f" {r.f2} {r.d2} {r.cabin}" if r.f2 else "") for r in itins.itertuples()])
    return checksums(folder)


def checksums(folder: str) -> Dict[str, str]:
    out = {}
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and fn not in ("build_stats.json",):
            out[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out
