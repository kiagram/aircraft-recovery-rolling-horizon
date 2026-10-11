"""Fetch the public BTS inputs into us2014/raw/ (gitignored) and pin them by SHA-256 in raw_manifest.json.

    python -m arp.us2014.download            # download missing files, verify/record checksums
    python -m arp.us2014.download --check    # verify checksums only (no network)

T-100 Segment (All Carriers) 2014 is obtained by replaying the BTS "field selection" form (the numbered
PREZIP snapshots only cover May 2014 or lack passengers/seats).  If that fails, drop the CSV manually at
us2014/raw/T100_SEGMENT_2014.csv (columns UNIQUE_CARRIER, ORIGIN, DEST, MONTH, DEPARTURES_PERFORMED, SEATS,
PASSENGERS, CLASS); it is then registered with its checksum.  If it is absent the build falls back to a x10
DB1B scaling and says so.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import MANIFEST, RAW_DIR, UPSTREAM_DIR

PRE = "https://transtats.bts.gov/PREZIP/"
OTP = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2014_{m}.zip"
COUPON = "Origin_and_Destination_Survey_DB1BCoupon_2014_{q}.zip"
T100_NAME = "T100_SEGMENT_2014.csv"
T100_FORM = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45"
T100_FIELDS = ("DEPARTURES_SCHEDULED DEPARTURES_PERFORMED SEATS PASSENGERS DISTANCE UNIQUE_CARRIER ORIGIN DEST "
               "AIRCRAFT_TYPE YEAR MONTH CLASS").split()
LOCAL = ["airportCoordinates.csv", "combinedPlaneData.csv", "356548273_T_F41SCHEDULE_B43_2014.zip"]
# The upstream repository has no licence, so these three files are not redistributed: they are fetched from it at a
# pinned commit into us2014/upstream_sources/ and checked against raw_manifest.json.
UPSTREAM_COMMIT = "4802d7fcfd6627bfea0c6caca9c80dd9623aa162"
UPSTREAM_URL = f"https://raw.githubusercontent.com/tomersagi/us2014/{UPSTREAM_COMMIT}/data/sources/"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, dest: str, tries: int = 4):
    tmp = dest + ".part"
    for k in range(tries):
        try:
            with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  retry {k + 1} {url}: {e}", file=sys.stderr)
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed: {url}")


def _t100(dest: str):
    c = httpx.Client(follow_redirects=True, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
    h = c.get(T100_FORM).text
    data = {n: re.search(r'name="%s"[^>]*value="([^"]*)"' % n, h).group(1)
            for n in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION")}
    data.update(cboGeography="All", cboYear="2014", cboPeriod="All", chkDownloadZip="on", btnDownload="Download")
    data.update({f: "on" for f in T100_FIELDS})
    r = c.post(T100_FORM, data=data)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if n.upper().startswith("T_T100")][0]
    with z.open(name) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def targets():
    t = [(OTP.format(m=m), PRE + OTP.format(m=m)) for m in range(1, 13)]
    t += [(COUPON.format(q=q), PRE + COUPON.format(q=q)) for q in range(1, 5)]
    return t


def ensure(check: bool = False) -> int:
    a = argparse.Namespace(check=check)
    os.makedirs(RAW_DIR, exist_ok=True)
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    files = {n: os.path.join(RAW_DIR, n) for n, _ in targets()}
    files[T100_NAME] = os.path.join(RAW_DIR, T100_NAME)
    if not a.check:
        todo = [(n, u) for n, u in targets() if not os.path.exists(files[n])]
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda nu: (print("get", nu[0], flush=True), _get(nu[1], files[nu[0]])), todo))
        if not os.path.exists(files[T100_NAME]):
            try:
                print("get T-100 via BTS form", flush=True)
                _t100(files[T100_NAME])
            except Exception as e:  # noqa: BLE001
                print(f"T-100 unavailable ({e}); build will fall back to DB1B x10", file=sys.stderr)
    for n in LOCAL:
        files[n] = os.path.join(UPSTREAM_DIR, n)
        if not a.check and not os.path.exists(files[n]):
            os.makedirs(UPSTREAM_DIR, exist_ok=True)
            print("get", n, "(github.com/tomersagi/us2014 @", UPSTREAM_COMMIT[:7] + ")", flush=True)
            _get(UPSTREAM_URL + n, files[n])
    bad = 0
    for n, p in files.items():
        if not os.path.exists(p):
            print("MISSING", n)
            bad += n != T100_NAME
            continue
        s = sha256(p)
        if n not in man:
            if a.check:
                print("UNPINNED", n)
                bad += 1
                continue
            man[n] = {"sha256": s, "bytes": os.path.getsize(p),
                      "source": (PRE + n) if n != T100_NAME and n not in LOCAL else
                      ("BTS Air Carrier Statistics, T-100 Segment (All Carriers), 2014, form replay" if n == T100_NAME
                       else UPSTREAM_URL + n)}
        elif man[n]["sha256"] != s:
            print("CHECKSUM MISMATCH", n)
            bad += 1
    if not a.check:
        json.dump(man, open(MANIFEST, "w", newline="\n"), indent=1, sort_keys=True)
    print("OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    sys.exit(ensure(ap.parse_args().check))


if __name__ == "__main__":
    main()
