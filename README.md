# Rolling-horizon aircraft recovery under hourly airport slot quotas

Code, instance builders, experiment definitions and results for the article *"Rolling-horizon aircraft recovery under
hourly airport slot quotas: stage-boundary positioning targets and the price of myopia"* (S. S. Mousavinia and
A. Seyfi; submitted). The solver is pure Python on top of the open-source MIP solver HiGHS; no commercial solver is needed.

## Contents

| Path | Content |
|---|---|
| `arp_rh/arp/` | Solver package: ROADEF 2009 parser, cost regimes, flight-copy time-space MIP with stage-boundary positioning targets, full-horizon / rolling-horizon / staging / OCC policies, dynamic scenario generator (`scenarios.py`), independent plan evaluator, optimality certificates, path-based column generation (`cg/`) and the US2014-R instance builder (`us2014/`). See `arp_rh/README.md` |
| `arp_rh/experiments.py`, `batch.py`, `run_cg.py` | Runners: one run per call, a job list in parallel, one column-generation run |
| `arp_rh/analysis*.py` | Build every table and figure of the paper from the result folders |
| `arp_rh/results_v2/` | 984 result files: ROADEF one-day and two-day A sets, ablation, compensation regimes, B/XB/X sets, 40 dynamic scenarios |
| `arp_rh/results_us2014_v2/` | 70 US2014-R runs, US2014-R column generation, summaries |
| `arp_rh/results_cg_v2/` | Column-generation runs on A01, A02 and B01, summaries |
| `audit/stage_2/jobs/rerun_<block>.txt`, `run_block.sh`, `batch_cg.py` | Exact command lines of every reported run, and their runners (paths as in Appendix E of the paper) |
| `us2014/` | US2014-R documentation (every substitute for the unavailable tools), pinned source manifest, aircraft models, instance checksums, build statistics and validation report |
| `scripts/fetch_roadef2009.py` | Downloads the ROADEF 2009 instances and checks them against the data used for the paper |
| `scripts/check_numbers.py` | Recomputes the numbers quoted in the paper from the result files (208 checks) |
| `scripts/validate_release.py` | Re-evaluates every stored plan with the independent evaluator and compares cost and feasibility |
| `requirements.txt`, `requirements-exact.txt` | Dependencies (minimum versions; exact versions used for the paper) |

Each result file holds the run settings, solver statistics, the evaluation by the independent checker (cost components,
cancellations, delays, broken itineraries, violated requirements), the per-stage log, the complete recovery plan and a
certificate (command line, time limit, LP method, HiGHS version, code commit, LP and MIP status, bound and its source,
claim). The code commit in a certificate refers to the authors' development repository; this repository contains the same
solver code. Absolute local paths in the stored command lines were replaced by paths relative to the repository root.

## Data that is not included

| Data | Why | How to obtain it |
|---|---|---|
| ROADEF/EURO 2009 challenge instances (A, B, XA, XB, X) | published by the challenge organisers without a redistribution licence | `python scripts/fetch_roadef2009.py` downloads them from https://challenge.roadef.org/2009/en/ and checks the SHA-256 of each zip and of each instance |
| BTS On-Time Performance 2014, DB1B Coupon 2014, T-100 Segment 2014 (about 1.1 GB) | size (public-domain US government data) | `python -m arp.us2014.download` (pinned by SHA-256 in `us2014/raw_manifest.json`) |
| `combinedPlaneData.csv`, `airportCoordinates.csv`, F41 inventory zip, and the wiki of github.com/tomersagi/us2014 | the upstream repository has no licence | the builder fetches the three files from the upstream repository at commit `4802d7f` and checks their SHA-256 |
| US2014-R instance files (16 instances and the `US02_A320_NQ` twin) | they combine BTS data with fleet data from the unlicensed upstream files | rebuilt deterministically with `python -m arp.us2014.build` and checked byte for byte against `us2014/instances/CHECKSUMS.json` |

## Install

Python 3.13 (other recent versions should work).
```bash
git clone https://github.com/kiagram/aircraft-recovery-rolling-horizon.git
cd aircraft-recovery-rolling-horizon
python -m pip install -r requirements-exact.txt      # versions used for the paper (or requirements.txt)
```

## Reproduce

All commands are run from the repository root unless a `cd` is shown.

**1. Check the numbers of the paper (seconds, no data needed).**
```bash
python scripts/check_numbers.py
```

**2. Get the ROADEF instances and re-evaluate every stored plan (about 1 min).**
```bash
python scripts/fetch_roadef2009.py
python scripts/validate_release.py
```

**3. Rebuild US2014-R (download of about 1.1 GB, then a few minutes).**
```bash
cd arp_rh
python -m arp.us2014.download
python -m arp.us2014.build --instance US03 US08_EMB120 US02_CRJ700 US08_CRJ700 US02_A320 US08_A320 US02_B737 US02_CRJ200 US02_A320_NQ US02 US04 US05 US06 US07 US08 US09 US10
python -m arp.us2014.build --verify
cd ..
python scripts/validate_release.py --us2014
```

**4. Regenerate the tables and figures** (ROADEF instances and the rebuilt US2014-R instances are needed for the
instance statistics).
```bash
cd arp_rh
mkdir -p ../out/tables ../out/figs
python analysis.py results_v2 ../out
python analysis_us2014.py results_us2014_v2 ../out
python analysis_cg.py ../out results_cg_v2 results_us2014_v2 results_v2
python analysis_synth.py ../out results_v2 results_us2014_v2
```

**5. Re-run experiments.** One run, for example the rolling horizon with stage-boundary targets on A01 (a few seconds):
```bash
cd arp_rh
python experiments.py --inst A01 --policy rh --step 60 --look 60 --terminal boundary --bnd kms --tl 300 --results-dir ../rerun
```
A whole block of the paper, into a new folder (resumable; at most four full-horizon runs in parallel, because contention
degrades them):
```bash
bash audit/stage_2/run_block.sh audit/stage_2/jobs/rerun_A1.txt arp_rh/rerun_v2 8
python audit/stage_2/batch_cg.py audit/stage_2/jobs/rerun_cg.txt 2      # column generation
```
In the column-generation job files the first column is the output folder under `arp_rh/`; existing result files are
skipped, so change it to a new folder before re-running.
Blocks: `A1` (one-day A set and ablation), `A2` (two-day A set), `B` and `B_fh4h` (B/XB/X), `dyn` (dynamic scenarios),
`regime` (compensation regimes), `us2014` (US2014-R), `cg` (column generation); see `audit/stage_2/README.md`. All
blocks together took about three days on one PC. Rolling-horizon runs whose stages finish within their limits are
deterministic (single thread); runs that stop at a time limit, in particular the full-horizon runs on the large
instances, depend on the machine and on how many jobs share it.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The result files are released under the same terms. The data
listed above remain under the terms of their providers. If you use the code or the results, please cite the article
(see `CITATION.cff`).
