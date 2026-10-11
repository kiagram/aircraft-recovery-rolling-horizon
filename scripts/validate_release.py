"""Re-evaluate every stored plan of the paper with the independent evaluator of this repository.

    python scripts/validate_release.py              # ROADEF results (needs roadef2009/, see fetch_roadef2009.py)
    python scripts/validate_release.py --us2014     # also US2014-R results (needs the rebuilt us2014/instances/)

For every result file in arp_rh/results_v2/, results_us2014_v2/ and results_cg_v2/ the instance (and, for the
dynamic study, the scenario, rebuilt from its seed) is loaded, its hash is compared with the certificate, the stored
plan is evaluated with arp.evaluate, and the total cost and feasibility are compared with the stored evaluation.
Exit code 1 on any mismatch.  This checks data, scenario generator and evaluator; it does not re-solve the MIPs
(see the README for that).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ARP = os.path.join(ROOT, "arp_rh")
sys.path.insert(0, ARP)

from arp.certificate import instance_sha256  # noqa: E402
from arp.evaluate import evaluate  # noqa: E402
from arp.instance import load  # noqa: E402
from arp.policies import Revelation  # noqa: E402
from arp.scenarios import ScenarioSpec, make_scenario  # noqa: E402

DATA = os.path.join(ROOT, "roadef2009")
_inst, _scen = {}, {}


def instance(name, regime):
    if name not in _inst:
        _inst[name] = load(name, DATA)
    I = _inst[name]
    I.regime = regime
    return I


def setup(r):
    """(instance, revelation used by the evaluator), exactly as experiments.py built them."""
    regime = r.get("regime") or "roadef"
    if r.get("scenario") is None:
        I = instance(r["instance"], regime)
        return I, (Revelation.progressive(I) if r.get("info") == "progressive" else None)
    lo, hi = (int(x) for x in r["lead"].split(","))
    key = (r["instance"], r["scenario"], r["cap_ratio"], lo, hi, regime)
    if key not in _scen:
        I = load(r["instance"], DATA)
        I.regime = regime
        J, rev = make_scenario(I, ScenarioSpec(cap_ratio=r["cap_ratio"], lead=(lo, hi)), r["scenario"])
        J.regime = regime
        _scen[key] = (J, rev)
    return _scen[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--us2014", action="store_true", help="include the US2014-R results (instances must be rebuilt)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ARP, "results_v2", "*__*.json")) + glob.glob(os.path.join(ARP, "results_cg_v2", "*__*.json")))
    if a.us2014:
        files += sorted(glob.glob(os.path.join(ARP, "results_us2014_v2", "*__*.json")))
    n = bad = skipped = 0
    hashes = {}
    for f in files:
        r = json.load(open(f))
        if "plan" not in r or "eval" not in r:
            continue
        if r["instance"].startswith("US") and not a.us2014:
            skipped += 1
            continue
        want = r.get("certificate", {}).get("problem", {}).get("instance_sha256")
        if want:
            got = hashes.setdefault(r["instance"], instance_sha256(r["instance"], DATA))
            if got != want:
                print("INSTANCE HASH MISMATCH", os.path.basename(f))
                bad += 1
                continue
        I, rev = setup(r)
        plan = {int(k): (tuple(v) if v != "C" else "C") for k, v in r["plan"].items()}
        e = evaluate(I, plan, revelation=rev)
        ok = abs(e["total"] - r["eval"]["total"]) <= 1e-6 * max(1.0, abs(r["eval"]["total"])) and e["feasible"] == r["eval"]["feasible"]
        n += 1
        if not ok:
            bad += 1
            print("MISMATCH", os.path.basename(f), r["eval"]["total"], "->", e["total"], r["eval"]["feasible"], "->", e["feasible"])
        if n % 100 == 0:
            print(n, "plans re-evaluated", flush=True)
    print(f"{n} plans re-evaluated, {bad} mismatches" + (f", {skipped} US2014-R results skipped (use --us2014)" if skipped else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
