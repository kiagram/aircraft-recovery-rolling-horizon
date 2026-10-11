"""Data-derived plane-model table (substitute for the HP-internal Talend 'PlaneModel' table).

    python -m arp.us2014.make_models        # writes us2014/models.csv (committed)

range_min = ceil(1.1 x longest scheduled block time flown by the model in the 2014 OTP data)
turn      = clamp(2nd percentile of scheduled same-airport ground times, 20, class default), multiple of 5
cost_h    = linear in seats, fitted by eye to the ROADEF'09 A/B aircraft table (narrow/regional 1000+6*seats,
            wide-body 4800+3*seats) -- an operating-cost proxy, NOT a Form 41 figure.
family    = aircraft type family (A320, B737, ...); tails only swap within a family (see evaluate.family_swap).
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pandas as pd

from . import US_DIR
from .build_base import build_otp, fleet

# (regex on combinedPlaneData 'Model', short model id, family)
RULES = [
    (r"Airbus A319", "A319", "A320"), (r"Airbus A320", "A320", "A320"),
    (r"737-7", "B737-700", "B737"), (r"737-8", "B737-800", "B737"), (r"737-9", "B737-900", "B737"),
    (r"757-2", "B757-200", "B757"), (r"757-3", "B757-300", "B757"),
    (r"767-3", "B767-300", "B767"), (r"767-4", "B767-400", "B767"),
    (r"777-2", "B777-200", "B777"), (r"747-4", "B747-400", "B747"),
    (r"787-8", "B787-8", "B787"), (r"787-9", "B787-9", "B787"),
    (r"CL-600-2B19", "CRJ200", "CRJ200"), (r"CL-600-2C10", "CRJ700", "CRJ700"),
    (r"EMB-120", "EMB120", "EMB120"), (r"ERJ-135", "ERJ135", "ERJ145"), (r"ERJ-145", "ERJ145", "ERJ145"),
    (r"ERJ-175", "E175", "E175"), (r"ATR 42", "ATR42", "ATR"), (r"ATR 72", "ATR72", "ATR"),
]
WIDE = {"B747", "B767", "B777", "B787"}
CSV = os.path.join(US_DIR, "models.csv")


def short_of(model: str):
    for rx, short, fam in RULES:
        if re.search(rx, model):
            return short, fam
    raise KeyError(model)


def make() -> pd.DataFrame:
    b = build_otp()
    fl = fleet()
    fl = fl[fl.has_cfg].copy()
    fl["model"], fl["family"] = zip(*fl.Model.map(short_of))
    f = b[b.in_fleet].copy()
    f["model"] = f.tailnum.map(fl.model)
    f = f[f.model.notna()].sort_values(["tailnum", "dep"])
    nxt = f.groupby("tailnum")[["orig", "dep"]].shift(-1)
    f["gap"] = (nxt.dep - f.arr).dt.total_seconds() / 60
    f["cont"] = nxt.orig == f.dest
    seats = (fl.cabF + fl.cabB + fl.cabE).groupby(fl.model).mean()
    rows = []
    for m, g in fl.groupby("model"):
        ff = f[f.model == m]
        s = float(seats[m])
        cls = "regional" if s <= 100 else ("wide" if s > 260 or g.family.iloc[0] in WIDE else "narrow")
        dflt = {"regional": 30, "narrow": 40, "wide": 60}[cls]
        gp = ff[ff.cont & (ff.gap >= 0) & (ff.gap < 600)].gap
        q = float(np.percentile(gp, 2)) if len(gp) > 50 else dflt
        turn = int(min(dflt, max(20, 5 * round(q / 5))))
        rng = int(math.ceil(1.1 * ff.dur.max())) if len(ff) else 120
        cost = (4800 + 3.0 * s) if cls == "wide" else (1000 + 6.0 * s)
        rows.append(dict(model=m, family=g.family.iloc[0], size_class=cls, mean_seats=round(s, 1), tails=len(g),
                         flights_2014=len(ff), max_block_min=int(ff.dur.max()) if len(ff) else 0, range_min=rng,
                         cost_h=round(cost, 1), turn=turn, transit=turn, p02_gap_min=round(q, 1)))
    out = pd.DataFrame(rows).sort_values(["family", "model"])
    out.to_csv(CSV, index=False)
    return out


def load_models() -> pd.DataFrame:
    if not os.path.exists(CSV):
        make()
    return pd.read_csv(CSV).set_index("model")


if __name__ == "__main__":
    print(make().to_string())
