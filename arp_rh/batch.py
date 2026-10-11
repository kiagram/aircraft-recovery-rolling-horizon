"""Run a list of experiments.py argument lines in parallel.  Usage: python batch.py jobs.txt [n_workers [results_dir]]
Jobs whose result file already exists are skipped (resumable)."""
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


RESULTS = os.path.join(HERE, "results")


def run(line):
    t = time.time()
    log = os.path.join(RESULTS, "logs")
    os.makedirs(log, exist_ok=True)
    key = "_".join(x.strip("-") for x in shlex.split(line))[:180]
    with open(os.path.join(log, key + ".log"), "w") as fh:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "experiments.py")] + shlex.split(line)
                             + (["--results-dir", RESULTS] if RESULTS != os.path.join(HERE, "results") else []),
                             stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    return line, rc, time.time() - t


if __name__ == "__main__":
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if len(sys.argv) > 3:
        RESULTS = os.path.abspath(sys.argv[3])
    with ThreadPoolExecutor(n) as ex:
        for line, rc, dt in ex.map(run, jobs):
            print(f"[{'ok' if rc == 0 else 'FAIL'}] {dt:7.1f}s  {line}", flush=True)
