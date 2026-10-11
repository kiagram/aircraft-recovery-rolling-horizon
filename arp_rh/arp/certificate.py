"""Certificate and provenance record stored in every result file (audit Stage 2, R-8 and R-11).

A result may claim
  exact        LP or unrestricted-MIP bound valid (LP status Optimal / MIP dual bound of the unrestricted model),
               incumbent feasible, model price = evaluated total - C, and gap <= 1e-9;
  eps-optimal  as `exact` with gap eps > 1e-9 (eps stored);
  best-known   feasible plan without a valid bound (rolling horizon, or no optimal LP);
  none         infeasible plan.
The claim is valid only for the problem in `valid_for` (instance, scenario, rules, decision set, objective version):
another plan is covered by the bound only if it belongs to that decision set (R-3) and is evaluated by the same rules.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Dict, Optional

import highspy

from .instance import INSTANCE_DIRS

RULES_VERSION = "stage2-D1-D7"          # D1 ROADEF rule, D2 priced maintenance, D3 per-flight grid, D4 manuscript
                                        # commitment, D5 one exemption rule, D6 one grid per instance
OBJECTIVE_VERSION = "R1-R2-R4"          # exact broken-itinerary delay, pre-broken constants, priced maintenance
HERE = os.path.dirname(os.path.abspath(__file__))


def instance_sha256(name: str, root: str) -> Optional[str]:
    folder = os.path.join(root, INSTANCE_DIRS.get(name, name))
    if not os.path.isdir(folder):
        return None
    h = hashlib.sha256()
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and fn.endswith(".csv"):
            h.update(fn.encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def code_commit() -> Dict:
    """git commit of the solver code and whether arp_rh/ had uncommitted changes when the run started."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    root = os.path.join(HERE, "..")
    st = git("status", "--porcelain", "--", HERE, os.path.join(root, "experiments.py"), os.path.join(root, "run_cg.py"))
    return dict(commit=git("rev-parse", "HEAD"), dirty=None if st is None else bool(st.strip()),
                scope="arp_rh/arp, experiments.py, run_cg.py")


def highs_version() -> str:
    return highspy.Highs().version()


def make(problem: Dict, solver: Dict, run, e: Dict, C: float, is_single_model: bool) -> Dict:
    """`run` is a policies.RunResult; `e` the evaluation; C the constant of itineraries broken before T0."""
    feasible = bool(e.get("feasible", not e.get("n_violations")))
    rec = dict(problem=problem, solver=solver, lp=None, mip=None, claim="none" if not feasible else "best-known",
               eps=None, valid_for=dict(problem))
    if is_single_model:
        lp, mip = (run.cert or {}).get("lp"), (run.cert or {}).get("mip")
        rec["lp"], rec["mip"] = lp, mip
        bound = (run.cert or {}).get("bound")
        ident = None if run.obj is None else run.obj - (e["total"] - C)
        rec["incumbent"] = dict(evaluated_total=e["total"], feasible=feasible, violations=e["violations"],
                                model_obj=run.obj, model_minus_eval_minus_C=ident)
        rec["bound"], rec["bound_source"] = bound, (run.cert or {}).get("bound_source")
        # the plan's model price is E - C (identity of R-1/R-2/R-4); the incumbent objective may exceed it when the
        # solver stops with continuous penalty variables above their minimum (ident > 0).  Only ident < 0 would
        # contradict the identity.  The gap is that of the plan's own price to the bound.
        if feasible and bound is not None and ident is not None and ident >= -1e-6 * max(1.0, abs(e["total"])):
            price = e["total"] - C
            gap = max(0.0, (price - bound) / max(abs(price), 1e-9))
            rec["claim"], rec["eps"] = ("exact", 0.0) if gap <= 1e-9 else ("eps-optimal", gap)
            rec["evaluated_lower_bound"] = bound + C       # lower bound on the evaluated cost of every plan in valid_for
    else:
        st = [x.get("lp_status") for x in run.log]
        rec["stages"] = dict(n=len(run.log), lp_not_optimal=sum(1 for s in st if s not in (None, "Optimal")),
                             mip_status=sorted({x.get("status") for x in run.log if x.get("status")}),
                             fallbacks=run.fallbacks)
        rec["incumbent"] = dict(evaluated_total=e["total"], feasible=feasible, violations=e["violations"])
    return rec
