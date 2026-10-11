"""Download the ROADEF/EURO 2009 challenge instances into roadef2009/ and check them.

    python scripts/fetch_roadef2009.py            # download missing zips, verify SHA-256, unpack
    python scripts/fetch_roadef2009.py --check    # verify zips and the unpacked instances only (no network)

The instances are published by the challenge organisers at https://challenge.roadef.org/2009/en/ without a
licence for redistribution, so this repository does not contain them.  The zips are pinned by SHA-256; after
unpacking, the per-instance hashes are compared with the `instance_sha256` stored in the certificate of every
result file of the paper (arp_rh/results_v2/), so a mismatch with the data used for the paper is detected.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEST = os.path.join(ROOT, "roadef2009")
URL = "https://challenge.roadef.org/2009/files/"
ZIPS = {
    "A1_6088570.zip": "4eedefb03ef60dbaeda730f1c0e3be97fe21aae32affc52b1c8a205a0b2bec7f",      # A01-A05
    "SA2_6088590.zip": "5c6a9db7d6224223a799bb4f4850c66ca9db07221fc92c3a6a15a2d449fec9d6",     # A06-A10
    "SetB1_5.zip": "ac4a6482ebd4a72d2e5bb4248ae6fedbcbaef108ab683c24f037bd202cfa7cde",         # B01-B05
    "SetB2_5.zip": "3c32bb862eb68d9dd7f4ec6cdf42dea7ed3ccf6df88f2a75282ebdbde245d557",         # B06-B10
    "Instances_X.zip": "250b20e7007fb5ae4390d918ca85740984a703926c06e464edf489f5edb3cb40",     # XA, XB, X
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_hashes():
    """instance name -> instance_sha256 recorded in the certificates of the paper's ROADEF results."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "arp_rh", "results_v2", "*__*.json")):
        c = json.load(open(f)).get("certificate", {}).get("problem", {})
        if c.get("instance") and c.get("instance_sha256"):
            out[c["instance"]] = c["instance_sha256"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="no download, verify only")
    a = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    bad = 0
    for name, want in ZIPS.items():
        p = os.path.join(DEST, name)
        if not os.path.exists(p) and not a.check:
            print("get", URL + name, flush=True)
            req = urllib.request.Request(URL + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(p + ".part", "wb") as out:
                out.write(r.read())
            os.replace(p + ".part", p)
        if not os.path.exists(p):
            print("MISSING", name)
            bad += 1
            continue
        if sha256(p) != want:
            print("CHECKSUM MISMATCH", name)
            bad += 1
            continue
        if not a.check:
            with zipfile.ZipFile(p) as z:
                z.extractall(DEST)
    sys.path.insert(0, os.path.join(ROOT, "arp_rh"))
    from arp.certificate import instance_sha256
    ref = stored_hashes()
    ok = 0
    for inst, want in sorted(ref.items()):
        if inst.startswith("US"):
            continue
        got = instance_sha256(inst, DEST)
        if got != want:
            print("INSTANCE MISMATCH" if got else "INSTANCE MISSING", inst)
            bad += 1
        else:
            ok += 1
    print(f"{ok} instances identical to the data used for the paper" + ("" if not bad else f"; {bad} problem(s)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
