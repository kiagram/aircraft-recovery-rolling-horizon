"""Run audit/stage_2/jobs/rerun_cg*.txt lines (`<results folder> | run_cg.py args`) with n workers.
    python audit/stage_2/batch_cg.py audit/stage_2/jobs/rerun_cg.txt 2"""
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ARP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "arp_rh")


def run(line):
    folder, args = (x.strip() for x in line.split("|", 1))
    t = time.time()
    out = os.path.join(ARP, folder)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(x.strip("-") for x in shlex.split(args)))[:150]
    with open(os.path.join(out, "logs", "cg_" + key + ".out"), "w") as fh:
        rc = subprocess.call([sys.executable, "run_cg.py"] + shlex.split(args) + ["--results-dir", out],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=ARP)
    return line, rc, time.time() - t


if __name__ == "__main__":
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as ex:
        for line, rc, dt in ex.map(run, jobs):
            print(f"[{'ok' if rc == 0 else 'FAIL'}] {dt:7.1f}s  {line}", flush=True)
