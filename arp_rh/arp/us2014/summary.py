"""SUMMARY.md and the README table (flights, tails, airports, itineraries, pax, disruptions, maintenance, drop rate)."""
from __future__ import annotations

import itertools
import json
import os
import re

import pandas as pd

from . import CACHE_DIR, INST_DIR, US_DIR

WIKI_COUNT = 1_164_155
S, E_ = "<!-- SUMMARY:START -->", "<!-- SUMMARY:END -->"


def fullyear() -> str:
    b = pd.read_parquet(os.path.join(CACHE_DIR, "base_flights.parquet"), columns=["carrier", "in_fleet", "cancelled", "tailnum"])
    st = json.load(open(os.path.join(CACHE_DIR, "base_stats.json")))
    rows = []
    for cs in itertools.chain.from_iterable(itertools.combinations(["UA", "EV", "OO"], r) for r in (1, 2, 3)):
        x = b[b.carrier.isin(cs)]
        rows.append(("+".join(cs), "any tail", len(x)))
        rows.append(("+".join(cs), "tail in United fleet file", int(x.in_fleet.sum())))
    near = sorted(rows, key=lambda r: abs(r[2] - WIKI_COUNT))[:4]
    lines = ["| filter on 2014 OTP rows | flights | vs wiki 1,164,155 |", "|---|---:|---:|"]
    for c, f, n in near:
        lines.append(f"| reporting carrier {c}, {f} | {n:,} | {n - WIKI_COUNT:+,} ({(n - WIKI_COUNT) / WIKI_COUNT:+.1%}) |")
    n = int(b.in_fleet.sum())
    lines.append(f"| **our base filter**: carriers UA+EV+OO and tail in `combinedPlaneData` | **{n:,}** | {n - WIKI_COUNT:+,} ({(n - WIKI_COUNT) / WIKI_COUNT:+.1%}) |")
    return "\n".join(lines) + f"\n\nOTP 2014 rows for UA/EV/OO: {st['otp_rows_ua_ev_oo']:,} (UA {st['otp_by_carrier']['UA']:,}, EV {st['otp_by_carrier']['EV']:,}, OO {st['otp_by_carrier']['OO']:,}); rows without tail number: {st['base_flights_no_tail']:,}."


def table(res: list) -> str:
    h = ("| instance | flights | tails | airports | itineraries | pax | delay disr. | cancel disr. | aircraft outages | "
         "airport reductions | maintenance | fixed flights | repair drop rate |")
    out = [h, "|" + "---|" * 13]
    for r in res:
        s = r["stats"]
        out.append(f"| {r['name']} | {s['flights']:,} | {s['tails']} | {s['airports']} | {s['itineraries']:,} | {s['pax']:,} | "
                   f"{s['disr_delays']} | {s['disr_cancels']} | {s['aircraft_outages']} | {s['airport_reductions']} | "
                   f"{s['maintenance']} | {s['fixed_flights']} | {s['repair_drop_rate']:.1%} |")
    return "\n".join(out)


def checks(res: list) -> str:
    out = ["| instance | loader round-trip | twin n_violations | twin pos_missing | twin seat spill | bad connections | itineraries referencing dropped flights | repair rerun drops | capacity-fix rerun updates |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res:
        s = r["stats"]
        out.append(f"| {r['name']} | {'OK' if r['roundtrip_ok'] else 'FAIL'} | {r['twin_n_violations']} | {r['twin_pos_missing']} | "
                   f"{r['twin_spilled']} | {r['itin_bad_connections']} | {r['itin_refs_to_dropped']} | {s['repair_rerun_dropped']} | {s['capacity_fix_updates_rerun']} |")
    return "\n".join(out)


def fhtable(res: list) -> str:
    out = ["| instance | flights | status | obj | bound | gap | wall s | solver n_violations | gate |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in res:
        f = r.get("fh")
        if f:
            out.append(f"| {r['name']} | {r['stats']['flights']} | {f['status']} | {f['obj']:,.0f} | {f['bound']:,.0f} | {f['gap']:.1e} | {f['wall_s']} | {f['n_violations']} | {f['gate']} |")
    return "\n".join(out)


def write(res: list, root: str = INST_DIR):
    body = "## Summary table\n\n" + table(res) + "\n\n## Validation\n\n" + checks(res) + \
           "\n\n## FH gate (scale-down instances, 10 min limit, HiGHS, LP-restricted)\n\n" + fhtable(res) + "\n\n## Full-year flight count vs the wiki\n\n" + fullyear() + "\n"
    open(os.path.join(root, "SUMMARY.md"), "w").write(body)
    rp = os.path.join(US_DIR, "README.md")
    if os.path.exists(rp):
        t = open(rp).read()
        if S in t:
            t = re.sub(re.escape(S) + ".*?" + re.escape(E_), S + "\n" + body + E_, t, flags=re.S)
            open(rp, "w", newline="\n").write(t)
