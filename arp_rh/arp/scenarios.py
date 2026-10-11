"""Dynamic disruption scenarios layered on top of ROADEF 2009 base instances.

A scenario adds, inside the recovery period,
  * a weather-induced capacity drop at one or two busy airports, possibly
    announced in two steps (an initial forecast that is later extended),
  * random operational (en-route/turnaround) delays of individual flights,
  * aircraft-on-ground (AOG) breakdowns,
together with the time at which each piece of information is revealed.

Audit Stage 2, R-6: announcement times of capacity reductions and aircraft outages are rounded up to the hourly decision
epochs, so that a flight committed by a rolling-horizon stage before an announcement also departs before it (and is
therefore exempt from it under decision D5); the random draws are unchanged.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .instance import Instance
from .model import capacity
from .policies import Revelation


@dataclass
class ScenarioSpec:
    cap_ratio: float = 0.5          # remaining share of nominal hourly capacity during the drop
    drop_hours: Tuple[int, int] = (3, 6)
    extension_prob: float = 0.5     # probability that the drop is extended after announcement
    n_drop_airports: int = 2
    delay_share: float = 0.05       # share of window flights receiving an operational delay
    delay_mean: float = 45.0
    n_aog: int = 2
    aog_hours: Tuple[int, int] = (3, 8)
    lead: Tuple[int, int] = (60, 180)
    ann_epoch: int = 60             # R-6: announcements of reductions/outages are rounded up to the decision epochs
                                    # (multiples of this many minutes after T0; 0 = raw minute)


def _epoch(t: int, start: int, step: int) -> int:
    return t if step <= 0 else start + -(-(t - start) // step) * step


def busiest_airports(I: Instance, k: int) -> List[str]:
    c = Counter()
    for f in I.flights:
        if not f.is_transp and I.start <= f.dep < I.end:
            c[f.orig] += 1
            c[f.dest] += 1
    return [a for a, _ in c.most_common(k)]


def make_scenario(I: Instance, spec: ScenarioSpec, seed: int) -> Tuple[Instance, Revelation]:
    rng = random.Random(seed)
    J = copy.deepcopy(I)
    J.name = f"{I.name}-s{seed}"
    rev = Revelation.static(I)
    rev.exempt = True               # D5: announcement-time exemptions in the dynamic study
    start, end = I.start, I.end
    span = end - start
    # ---- airport capacity drops -------------------------------------------------
    hubs = busiest_airports(I, 4)
    for ap in rng.sample(hubs, spec.n_drop_airports):
        s = start + 60 * rng.randint(1, max(1, span // 60 // 3))
        dur = 60 * rng.randint(*spec.drop_hours)
        ext = 60 * rng.randint(1, 3) if rng.random() < spec.extension_prob else 0
        announce = _epoch(max(start, s - rng.randint(*spec.lead)), start, spec.ann_epoch)
        pieces = [(s, s + dur, announce)]
        if ext:
            pieces.append((s + dur, s + dur + ext, _epoch(max(start, s + dur - rng.randint(spec.lead[0], spec.lead[0] + 30)),
                                                          start, spec.ann_epoch)))
        for ps, pe, ann in pieces:
            h = ps
            while h < pe:
                d, a = capacity(I, J.alt_cap, ap, h)
                J.alt_cap.setdefault(ap, []).append((h, h + 60, int(d * spec.cap_ratio), int(a * spec.cap_ratio)))
                rev.airport[(ap, len(J.alt_cap[ap]) - 1)] = ann
                h += 60
    # ---- operational delays ------------------------------------------------------
    window = [f for f in I.modifiable if f not in I.flight_disr and I.flights[f].dep < end - 120]
    for f in rng.sample(window, int(round(spec.delay_share * len(window)))):
        d = max(10, int(rng.expovariate(1.0 / spec.delay_mean) // 5 * 5))
        J.flight_disr[f] = min(d, 240)
        rev.flight[f] = max(start, I.flights[f].dep - rng.randint(*spec.lead))
    # ---- AOG breakdowns --------------------------------------------------------------
    busy = Counter(I.flights[f].aircraft for f in I.modifiable)
    cands = [t for t, n in busy.items() if n >= 3 and t not in I.ac_unavail and not I.aircraft[t].maint]
    for t in rng.sample(cands, min(spec.n_aog, len(cands))):
        s = start + 60 * rng.randint(1, max(1, span // 60 // 2))
        e = s + 60 * rng.randint(*spec.aog_hours)
        J.ac_unavail.setdefault(t, []).append((s, e))
        rev.aircraft[(t, len(J.ac_unavail[t]) - 1)] = _epoch(max(start, s - rng.randint(*spec.lead)), start, spec.ann_epoch)
    return J, rev
