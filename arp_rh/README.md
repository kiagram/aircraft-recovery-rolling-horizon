# Rolling-horizon aircraft recovery with hourly airport slot quotas

Code, experiment definitions and results for the article *"Rolling-horizon aircraft recovery under hourly
airport slot quotas: stage-boundary positioning targets and the price of myopia"* (S. S. Mousavinia and
A. Seyfi, Amirkabir University of Technology). It rebuilds and extends the MSc thesis code
(https://github.com/SHMN52/GradThesis).

The solver is pure Python on top of the open-source MIP solver HiGHS; no commercial solver is needed.

## Contents
| Path | Purpose |
|---|---|
| `arp/instance.py` | Parser for the official ROADEF 2009 instances (sets A, B, XA, XB, X) |
| `arp/costs.py` | ROADEF cost structure (EU261-style meal/hotel/compensation, passenger disutility, seat spill); `regime` = `roadef` \| `eu261` \| `eu2025` |
| `arp/model.py` | Flight-copy fleet time-space MIP (HiGHS), LP-guided restriction, warm starts, stage-boundary targets |
| `arp/policies.py` | Full-horizon (FH), rolling horizon (RH, thesis-style staging, day-anchored look-ahead), rule-based OCC baseline, information revelation |
| `arp/scenarios.py` | Dynamic scenarios: weather capacity drops (with extensions), operational delays, AOG breakdowns, announcement times |
| `arp/evaluate.py` | Independent feasibility checker and cost evaluator (does not reuse the MIP) |
| `arp/certificate.py`, `arp/reporting.py` | Optimality certificates stored with every run; grouping and reference selection used by the analysis |
| `arp/cg/` | Path-based column generation (resource-constrained shortest-path pricing) |
| `arp/us2014/` | US2014-R builder: download of the public BTS inputs (pinned by SHA-256), rebuild of the instances, validation |
| `experiments.py` | One run -> `<results dir>/<instance>__<policy>.json` |
| `batch.py` | Runs a job list in parallel (resumable: finished runs are skipped) |
| `run_cg.py` | One column-generation run |
| `analysis.py`, `analysis_us2014.py`, `analysis_cg.py`, `analysis_synth.py` | Build the paper's tables and figures from the result folders |
| `results_v2/`, `results_us2014_v2/`, `results_cg_v2/` | Results reported in the paper: one JSON file per run (see below). They are produced by the job files `../audit/stage_2/jobs/rerun_<block>.txt` |

## Install
```
python -m pip install -r requirements.txt
```
Tested with Python 3.13, highspy 1.12, numpy 2.4 and matplotlib 3.11 on Windows 11.

## Data
The ROADEF/EURO 2009 challenge instances are published by the challenge organisers at https://challenge.roadef.org/2009/en/
and are not redistributed here. `python ../scripts/fetch_roadef2009.py` downloads the five zips, checks their SHA-256
and unpacks them next to this folder, in `../roadef2009/`, with this layout:
```
roadef2009/
  A01_6088570/ ... A05_6088570/      (A1_6088570.zip)
  A06_6088590/ ... A10_6088590/      (SA2_6088590.zip)
  SetB1/B_01/ ... SetB1/B_05/        (SetB1_5.zip)
  SetB2/B_06/ ... SetB2/B_10/        (SetB2_5.zip)
  Copy of Instance_X/XA01_DATA/ ...  (Instances_X.zip: XA, XB and X instances)
```
The mapping from instance names to folders is `INSTANCE_DIRS` in `arp/instance.py`. The US2014-R instances are rebuilt
from public BTS data with `python -m arp.us2014.build` (see `../us2014/README.md`).

## Run
```
python experiments.py --inst A01 --policy fh                                    # full-horizon optimum
python experiments.py --inst A01 --policy rh --step 60 --look 60 --terminal boundary --bnd kms    # RH with targets
python experiments.py --inst A01 --policy rh --step 60 --look 60 --commit stage --terminal none   # thesis-style staging
python experiments.py --inst A01 --policy occ                                   # rule-based OCC baseline
python experiments.py --inst A01 --scenario 3 --cap_ratio 0.3 --info progressive --policy rh --step 60 --look day --terminal boundary
```
Main options: `--step` and `--look` are the step and look-ahead in minutes (`--look inf` = periodic full
re-optimization, `--look day` = until the next night break); `--commit exec|stage` is execution-based or
thesis-style commitment; `--terminal boundary` switches on the stage-boundary targets, whose components
are chosen with `--bnd`: `k` kind-level hierarchical targets, `m` maintenance tail targets, `s` slot
reservation (the method of the paper is `kms`), and, for the ablation, `v` rotation-value pricing, `n`
time-indexed needs and `c` connection protection. `--regime` selects the passenger-compensation rules,
`--tl` the time limit (whole FH run or each RH stage) and `--grid fine|coarse` the delay-copy grid.

## Reproduce the paper
After the Stage-2 audit repairs (`../audit/stage_2/`), every reported experiment has one job file
`../audit/stage_2/jobs/rerun_<block>.txt` (blocks A1, A2, B, dyn, regime, us2014, cg). From the repository root:
```
bash audit/stage_2/run_block.sh audit/stage_2/jobs/rerun_A1.txt arp_rh/results_v2 8     # <= 4 FH runs in parallel
bash audit/stage_2/run_block.sh audit/stage_2/jobs/rerun_us2014.txt arp_rh/results_us2014_v2 6
python audit/stage_2/batch_cg.py audit/stage_2/jobs/rerun_cg.txt 2
cd arp_rh && python analysis.py results_v2 ../paper     # tables and figures (see paper Appendix E)
```
Each result file stores a `certificate` (command, time limit, HiGHS version, code commit, LP/MIP status, bound and its
source, claim).
Results under a time limit depend on how many jobs share the machine: on the large instances, full-horizon
runs that share the CPU with many other jobs find clearly worse solutions within the same limit.

## Result files
Each `results_v2/<name>__<policy>.json` (and likewise in the other result folders) holds the run settings (`instance`, `scenario`, `policy`, `step`,
`look`, `commit`, `terminal`, `bnd`, `regime`, `info`, `grid`), the solver statistics (`runtime`, `stages`,
`solve_time`, `max_stage_time`, `fallbacks`, and for FH the MIP `obj`, `bound` and `gap`), the evaluation
by the independent checker (`eval`: total cost, cost components, numbers of cancellations and delayed
flights, broken itineraries, missing end-of-recovery positions and any violated requirements), the
per-stage `log`, and the complete recovery `plan` (aircraft and departure time, or cancellation, for each
flight).

## Licence and citation
MIT licence (see `LICENSE`). If you use the code or the results, please cite the article (see
`CITATION.cff`).
