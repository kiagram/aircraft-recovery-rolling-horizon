# Experiment definitions of the paper

The results reported in the paper were produced after a correctness audit of the code (Stage 1) and the repairs that
followed it (Stage 2); the folder keeps that name so that the commands in Appendix E of the paper work as printed.

| File | Content |
|---|---|
| `jobs/rerun_<block>.txt` | one line per run (arguments of `arp_rh/experiments.py`): `A1` one-day A set incl. the ablation, `A1_A04A09` and `fix_A` repeated A runs after the D5 fix, `A2` two-day A set, `B` B/XB/X sets, `B_fh4h` four-hour full-horizon runs on B01-B03, `dyn` dynamic scenarios incl. the perfect-information reference, `regime` compensation regimes, `us2014`, `fix_us` and `us2014_fh_uncontended` US2014-R, `cg` and `cg_A` column generation (`<results folder> \| run_cg.py arguments`) |
| `run_block.sh` | runs one job file: full-horizon lines with at most 4 workers, all other lines with the given number of workers; resumable (finished runs are skipped) |
| `batch_cg.py` | runs a column-generation job file |

Some runs appear in more than one job file because they were repeated after a fix. `experiments.py` never overwrites an
existing result file, so a repeat ran only after the earlier file had been removed, and each stored result comes from the
last run of its line.
