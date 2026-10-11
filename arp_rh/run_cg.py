"""Column-generation (path formulation) solver for the aircraft recovery problem.

  python run_cg.py --inst A01 --grid fine --tl 600
Output: results_cg/<inst>[-regime]__cg-<grid>.json   (never writes into results/)
"""
import argparse
import json
import os
import sys

from arp.cg.driver import run
from arp.instance import load

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roadef2009")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cg")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", required=True)
    ap.add_argument("--grid", choices=["fine", "coarse"], default="fine")
    ap.add_argument("--regime", choices=["roadef", "eu261", "eu2025"], default="roadef")
    ap.add_argument("--tl", type=float, default=600.0, help="column-generation time limit (s)")
    ap.add_argument("--mip-tl", type=float, default=300.0, help="price-and-branch MIP time limit (s)")
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--restrict", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--no-fh-lp", action="store_true", help="skip recomputing the flow-model LP bound")
    ap.add_argument("--seed-plan", action="append", default=[], help="results JSON (read-only) whose routes are added as initial columns; repeatable")
    ap.add_argument("--box", type=float, default=None, help="box-step stabilisation radius (relative); default: Wentges smoothing")
    ap.add_argument("--rmp-lp", choices=["simplex", "ipm"], default="simplex", help="RMP LP algorithm during CG; ipm = interior point without crossover")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results-dir", default=None, help="output folder (default: results_cg/)")
    a = ap.parse_args(argv)
    OUT = os.path.abspath(a.results_dir) if a.results_dir else globals()["OUT"]
    I = load(a.inst, ROOT)
    I.regime = a.regime
    name = a.inst + ("" if a.regime == "roadef" else f"-{a.regime}")
    tag = a.tag or f"{name}__cg-{a.grid}"
    path = os.path.join(OUT, tag + ".json")
    if os.path.exists(path):
        print("exists", tag)
        return
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    r = run(I, a.grid, a.tl, a.mip_tl, a.mip_gap, a.restrict, not a.no_fh_lp, seeds=a.seed_plan,
            log_path=os.path.join(OUT, "logs", tag + ".log"),
            cg_kwargs=dict(box=a.box) if a.box is not None else None, rmp_lp=a.rmp_lp)
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(r, fh, default=str)
    e = r.get("eval", {})
    print(tag, f"LP={r['lp_bound']:.2f} LB={r['lb_valid']:.2f} UB={r['ub']} gap={r['gap']} eval={e.get('total')} viol={e.get('n_violations')} "
          f"rt={r['runtime']:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
