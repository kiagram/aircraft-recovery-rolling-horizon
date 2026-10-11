"""Command-line experiment runner.  One call = one (instance, scenario, policy) run -> results/<tag>.json

Examples
  python experiments.py --inst A01 --policy fh
  python experiments.py --inst A01 --policy rh --step 60 --look 180 --terminal boundary
  python experiments.py --inst A01 --scenario 3 --policy rh --step 60 --look 120 --info progressive
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from arp import certificate
from arp.evaluate import evaluate
from arp.instance import load
from arp.model import COARSE_GRID, Params, constant_offset
from arp.policies import Revelation, full_horizon, occ_rule, plan_to_warm, rolling_horizon, shared_extra_deps
from arp.scenarios import ScenarioSpec, make_scenario

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roadef2009")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--scenario", type=int, default=None, help="seed of a dynamic scenario (None = original)")
    ap.add_argument("--cap_ratio", type=float, default=0.5)
    ap.add_argument("--policy", choices=["fh", "fhpi", "rh", "occ"], required=True,
                    help="fhpi = perfect-information full horizon of a dynamic scenario (R-6)")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--look", default="60", help="minutes, 'inf', or 'day' (until the next night break)")
    ap.add_argument("--commit", default="exec")
    ap.add_argument("--terminal", default="none")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--bnd", default="kms", help="boundary components: c=connections, n=needs, k=kind targets, m=maintenance tails, s=slot reservation")
    ap.add_argument("--regime", choices=["roadef", "eu261", "eu2025"], default="roadef")
    ap.add_argument("--lead", default="60,180", help="announcement lead-time range (min) for dynamic scenarios")
    ap.add_argument("--restrict", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--info", choices=["static", "progressive"], default="static")
    ap.add_argument("--grid", choices=["fine", "coarse"], default="fine")
    ap.add_argument("--tl", type=float, default=600.0, help="time limit (whole FH run or each RH stage)")
    ap.add_argument("--lpm", default="choose")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results-dir", default=None, help="output folder (default: results/)")
    a = ap.parse_args(argv)
    OUT = os.path.abspath(a.results_dir) if a.results_dir else globals()["OUT"]

    I = load(a.inst, ROOT)
    I.regime = a.regime
    rev = None
    rev_eval = None             # announcement times used by the evaluator (D5: one rule for every plan)
    name = a.inst + ("" if a.regime == "roadef" else f"-{a.regime}")
    if a.scenario is not None:
        lo, hi = (int(x) for x in a.lead.split(","))
        spec = ScenarioSpec(cap_ratio=a.cap_ratio, lead=(lo, hi))
        I, rev_prog = make_scenario(I, spec, a.scenario)
        name = f"{a.inst}-s{a.scenario}-c{a.cap_ratio}-l{lo}" + ("" if a.regime == "roadef" else f"-{a.regime}")
        I.regime = a.regime
        rev = rev_prog if a.info == "progressive" else Revelation.static(I)
        rev_eval = rev_prog
    elif a.info == "progressive":
        rev = Revelation.progressive(I)
        rev_eval = rev
    if a.policy == "fhpi" and a.scenario is None:
        ap.error("--policy fhpi needs --scenario")
    p = Params(time_limit=a.tl, lp_restrict=True, rc_keep=50.0, terminal=a.terminal, boundary_rate=a.rate,
               lp_method=a.lpm)
    if a.grid == "coarse":
        p.delay_grid = COARSE_GRID
    p.bnd_conn, p.bnd_need, p.bnd_kind = "c" in a.bnd, "n" in a.bnd, "k" in a.bnd
    p.bnd_maint, p.bnd_slot, p.bnd_value = "m" in a.bnd, "s" in a.bnd, "v" in a.bnd
    p.restrict_singletons = (len(I.modifiable) > 1000) if a.restrict == "auto" else (a.restrict == "yes")
    look = "day" if a.look == "day" else (math.inf if a.look == "inf" else int(float(a.look)))
    if a.policy == "occ":
        pol0 = "occ"
    elif a.policy in ("fh", "fhpi"):
        pol0 = f"{'fh-pi' if a.policy == 'fhpi' else 'fh'}-{a.grid}"
    else:
        term0 = a.terminal + (f"[{a.bnd}]" if a.terminal == "boundary" else "")
        pol0 = f"rh-{a.step}-{'inf' if look == math.inf else look}-{a.commit}-{term0}" +                (f"-r{a.rate:g}" if a.rate is not None else "") + f"-{a.info}-{a.grid}"
    tag0 = a.tag or f"{name}__{pol0}"
    if os.path.exists(os.path.join(OUT, tag0 + ".json")):
        print("exists", tag0)
        return
    t0 = time.time()
    commit0 = certificate.code_commit()
    occ_plan = occ_rule(I).plan
    p.extra_deps = shared_extra_deps(I, occ_plan)     # R-3: one decision set for FH and every RH stage
    if a.policy == "occ":
        r = occ_rule(I)
        pol = "occ"
    elif a.policy in ("fh", "fhpi"):
        warm = plan_to_warm(occ_plan)
        r = full_horizon(I, p, warm=warm, revelation=rev_eval if a.policy == "fhpi" else None)
        pol = pol0
    else:
        r = rolling_horizon(I, p, step=a.step, lookahead=look, commit=a.commit, revelation=rev)
        term = a.terminal + (f"[{a.bnd}]" if a.terminal == "boundary" else "")
        pol = f"rh-{a.step}-{'inf' if look == math.inf else look}-{a.commit}-{term}" + \
              (f"-r{a.rate:g}" if a.rate is not None else "") + f"-{a.info}-{a.grid}"
    e = evaluate(I, r.plan, revelation=rev_eval)
    tag = a.tag or f"{name}__{pol}"
    rec = dict(regime=a.regime, lead=a.lead, instance=a.inst, scenario=a.scenario, cap_ratio=a.cap_ratio if a.scenario is not None else None,
               name=name, policy=pol, step=a.step, look=a.look, commit=a.commit, terminal=a.terminal, rate=a.rate,
               info=a.info, grid=a.grid, bnd=a.bnd if a.terminal == "boundary" else "", runtime=r.runtime, stages=r.stages, solve_time=r.solve_time,
               max_stage_time=r.max_stage_time, fallbacks=r.fallbacks, bound=r.bound, obj=r.obj, gap=r.gap, eval=e, log=r.log,
               plan={str(k): v for k, v in r.plan.items()})
    # certificate and provenance (audit Stage 2, R-8 / R-11)
    C = constant_offset(I)
    problem = dict(instance=a.inst, instance_sha256=certificate.instance_sha256(a.inst, ROOT), scenario_seed=a.scenario,
                   cap_ratio=a.cap_ratio if a.scenario is not None else None, lead=a.lead if a.scenario is not None else None,
                   regime=a.regime, information=a.info, rules_version=certificate.RULES_VERSION,
                   objective_version=certificate.OBJECTIVE_VERSION, constant_offset_C=C,
                   decision_set=dict(grid=a.grid, delay_grid=list(p.delay_grid), dmax=max(p.delay_grid),
                                     anchoring="e_f = t_f + max(d_f, 0)", hour_aligned=p.hour_aligned,
                                     warm_additions="OCC departures, shared by every method",
                                     restrict_singletons=p.restrict_singletons))
    solver = dict(name="HiGHS", version=certificate.highs_version(), threads=p.threads, time_limit=a.tl,
                  lp_method=a.lpm, mip_rel_gap=p.mip_gap, lp_restrict=p.lp_restrict, rc_keep=p.rc_keep,
                  code=commit0, command=" ".join(sys.argv[1:] if argv is None else argv))
    rec["certificate"] = certificate.make(problem, solver, r, e, C, is_single_model=a.policy in ("fh", "fhpi"))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, tag + ".json"), "w") as fh:
        json.dump(rec, fh, default=str)
    print(tag, f"total={e['total']:.0f} viol={e['n_violations']} rt={r.runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
