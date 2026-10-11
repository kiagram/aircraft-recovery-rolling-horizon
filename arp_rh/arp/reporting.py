"""Selection rules shared by the table scripts (audit Stage 2, R-5; decisions D2 and D6).

* A plan is **feasible** if it has no hard violation.  Maintenance location / flown-minute requirements are priced soft
  constraints (D2): their price is part of `eval.total`, they do not make a plan infeasible.
* Plans are ranked lexicographically by `(not feasible, total)`: a reference or a "best" entry is always the cheapest
  *feasible* plan; an infeasible plan is chosen only if no feasible plan exists, and is then labelled as such.
* Runs on different delay grids (fine/coarse) or information protocols (static/progressive) are different methods and
  get different labels, so they are never pooled or used as each other's reference (D6: every method of an instance
  uses the same grid; `grid_of` gives that grid).
* Result files written before R-4 have totals that exclude the maintenance price; `legacy(r)` marks them.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

SOFT = ("maint_location", "maint_hours")

# D6: one grid per instance for every method
COARSE_INSTANCES = {f"B{i:02d}" for i in range(1, 11)} | {f"XB{i:02d}" for i in range(1, 5)} | \
                   {f"X{i:02d}" for i in range(1, 5)} | {"US08_A320", "US02_B737", "US02_CRJ200"}


def grid_of(instance: str) -> str:
    return "coarse" if instance in COARSE_INSTANCES else "fine"


def hard_violations(r: Dict) -> Dict[str, int]:
    return {k: v for k, v in r["eval"]["violations"].items() if k not in SOFT and v}


def feasible(r: Dict) -> bool:
    e = r["eval"]
    return bool(e["feasible"]) if "feasible" in e else not hard_violations(r)


def legacy(r: Dict) -> bool:
    """Evaluated before R-4: `total` excludes the maintenance price (only wrong if maintenance is violated)."""
    return "maint_penalty" not in r["eval"]


def cost(r: Dict) -> float:
    return r["eval"]["total"]


def key(r: Dict):
    return (not feasible(r), cost(r))


def best(records: Iterable[Dict]) -> Optional[Dict]:
    rs = list(records)
    return min(rs, key=key) if rs else None


def base_label(policy: str) -> str:
    """Policy tag -> method label without grid/information suffix."""
    if policy.startswith("fh-pi"):
        return "FH-PI"
    if policy.startswith("fh"):
        return "FH"
    if policy == "occ":
        return "OCC"
    m = re.match(r"rh-(\d+)-(\w+)-(exec|stage)-(none|boundary\[(\w+)\])", policy)
    if m is None:
        return policy
    st, lk, cm, term, bnd = m.groups()
    if cm == "stage":
        return f"STG-{st}"
    if lk == "inf":
        return "RH-inf"
    if term == "none":
        return f"RH-M({st}/{lk})"
    if bnd and bnd != "kms":
        return f"RH-[{bnd}]({st}/{lk})"
    return f"RH-T({st}/{lk})"


def label(r: Dict) -> str:
    """Method label; runs on a grid other than the instance's D6 grid, and progressive-information runs on the
    original (static) instance, get a suffix so they are never compared with the main runs."""
    lab = base_label(r["policy"])
    grid = r.get("grid") or ("coarse" if "coarse" in r["policy"] else "fine")
    if r["policy"] != "occ" and grid != grid_of(r["instance"]):
        lab += f" [{grid}]"
    if r.get("info") == "progressive" and r.get("scenario") is None:
        lab += " [prog]"
    return lab


def group(records: Iterable[Dict]) -> Dict[str, Dict[str, Dict]]:
    """name -> label -> best record (feasibility first, then cost; several runs with one label, e.g. two FH runs)."""
    by: Dict[str, Dict[str, Dict]] = {}
    for r in records:
        r["lab"] = label(r)
        d = by.setdefault(r["name"], {})
        old = d.get(r["lab"])
        if old is None or key(r) < key(old):
            d[r["lab"]] = r
    return by


def reference(d: Dict[str, Dict], labels: Optional[Iterable[str]] = None) -> Optional[Dict]:
    """Best-known reference of one instance: the best record (feasibility first) among `labels` (default: all labels
    without a grid/information suffix)."""
    labs = list(labels) if labels is not None else [k for k in d if "[" not in k.split(")")[-1]]
    return best(d[k] for k in labs if k in d)


def mark(r: Dict) -> str:
    """LaTeX marker for a table cell: dagger = infeasible (hard violations), section sign = legacy total."""
    s = ""
    if not feasible(r):
        s += "$^\\dagger$"
    if legacy(r) and any(k in r["eval"]["violations"] for k in SOFT):
        s += "$^\\S$"
    return s


def describe(name: str, r: Optional[Dict]) -> str:
    if r is None:
        return f"{name}: no reference"
    st = "feasible" if feasible(r) else f"INFEASIBLE {hard_violations(r)}"
    return f"{name}: reference {r['lab']} ({r['policy']}) total {cost(r):,.2f} {st}" + (" legacy" if legacy(r) else "")
