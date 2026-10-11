"""Stage 1: OTP -> base_flights.parquet (UTC timestamps, tail/model join); DB1B/T-100 -> demand tables.

All outputs go to us2014/cache/ (gitignored)."""
from __future__ import annotations

import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

from . import CACHE_DIR, RAW_DIR, UPSTREAM_DIR
from .download import COUPON, OTP, T100_NAME

CARRIERS = ("UA", "EV", "OO")
TAIL_FIX = {"N27523": "N13979"}          # wiki: UPDATE SampleFlights SET TailNum='N13979' WHERE TailNum='N27523'
OTP_COLS = ["FlightDate", "Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline", "Origin", "Dest",
            "CRSDepTime", "CRSElapsedTime", "DepDelayMinutes", "ArrDelayMinutes", "Cancelled", "CancellationCode",
            "Diverted", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]


def _p(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def airports() -> pd.DataFrame:
    a = pd.read_csv(os.path.join(UPSTREAM_DIR, "airportCoordinates.csv"))
    a = a[a.iata.notna() & (a.iata != "\\N")].copy()
    a["us"] = (a.country == "United States") | a.icao.fillna("").str.match(r"^[KPT]")   # K*, P* (AK/HI), TJ/TI territories
    a = a.sort_values(["iata", "us"], ascending=[True, False]).drop_duplicates("iata")
    return a.set_index("iata")[["country", "latitude", "longitude", "tz"]]


def fleet() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(UPSTREAM_DIR, "combinedPlaneData.csv"))
    p["TailNum"] = p.TailNum.str.strip()
    cfg = p.Config.fillna("")
    for k in "FCWY":
        p[k] = cfg.str.extract(r"%s(\d+)" % k)[0].fillna(0).astype(int)
    p["has_cfg"] = cfg != ""
    p["cabF"], p["cabB"], p["cabE"] = p.F, p.C, p.W + p.Y     # premium economy (W) merged into economy
    return p.set_index("TailNum")


def build_otp(force: bool = False) -> pd.DataFrame:
    out = _p("base_flights.parquet")
    if os.path.exists(out) and not force:
        return pd.read_parquet(out)
    ap = airports()
    fl = fleet()
    parts, stats = [], {"otp_rows_ua_ev_oo": 0}
    for m in range(1, 13):
        z = zipfile.ZipFile(os.path.join(RAW_DIR, OTP.format(m=m)))
        df = pd.read_csv(z.open(z.namelist()[0]), usecols=OTP_COLS, dtype={"Tail_Number": str, "CRSDepTime": "Int64", "CancellationCode": str})
        df = df[df.Reporting_Airline.isin(CARRIERS)]
        stats["otp_rows_ua_ev_oo"] += len(df)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    stats["otp_by_carrier"] = df.Reporting_Airline.value_counts().to_dict()
    df["tailnum"] = df.Tail_Number.fillna("").str.strip().replace(TAIL_FIX)
    df["in_fleet"] = df.tailnum.isin(fl.index)
    # --- UTC timestamps: local CRSDepTime at origin on FlightDate -> UTC (zoneinfo via pandas), arrival = dep + CRSElapsedTime
    df = df[df.CRSElapsedTime.notna() & df.CRSDepTime.notna()].copy()
    stats["otp_rows_with_schedule"] = len(df)
    df = df[df.Origin.isin(ap.index) & df.Dest.isin(ap.index)].copy()
    stats["rows_with_known_airports"] = len(df)
    hhmm = df.CRSDepTime.astype(int).clip(0, 2359)
    local = pd.to_datetime(df.FlightDate) + pd.to_timedelta((hhmm // 100) % 24 * 60 + hhmm % 100, unit="m")
    dep = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    tzs = df.Origin.map(ap.tz)
    for tz, idx in df.groupby(tzs).groups.items():
        loc = local.loc[idx].dt.tz_localize(tz, ambiguous=np.ones(len(idx), bool), nonexistent="shift_forward")
        dep.loc[idx] = loc.dt.tz_convert("UTC")
    df["dep"] = dep.dt.tz_localize(None)
    df["dur"] = df.CRSElapsedTime.astype(int)
    spd = df.Distance / (df.dur / 60)
    ok = (df.dur >= 15) & (df.dur <= 720) & (spd >= 30) & (spd <= 700) & (df.Origin != df.Dest)
    stats["dropped_implausible_duration"] = int((~ok).sum())
    df = df[ok].copy()
    df["arr"] = df.dep + pd.to_timedelta(df.dur, unit="m")
    for c in ["DepDelayMinutes", "ArrDelayMinutes", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
              "LateAircraftDelay"]:
        df[c] = df[c].astype("float64")
    df = df.rename(columns={"Reporting_Airline": "carrier", "Origin": "orig", "Dest": "dest", "Distance": "dist",
                            "Flight_Number_Reporting_Airline": "fno", "Cancelled": "cancelled",
                            "CancellationCode": "ccode", "Diverted": "diverted"})
    df["cancelled"] = df.cancelled.astype(int)
    df["fno"] = df.fno.astype(str)
    df = df.drop(columns=["Tail_Number", "CRSDepTime", "CRSElapsedTime", "FlightDate"]).sort_values(
        ["dep", "carrier", "tailnum", "orig"], kind="stable").reset_index(drop=True)
    df["fid"] = np.arange(len(df))
    stats["base_flights_all_carriers_ua_ev_oo"] = len(df)
    stats["base_flights_tail_in_fleet"] = int(df.in_fleet.sum())
    stats["base_flights_no_tail"] = int((df.tailnum == "").sum())
    df.to_parquet(out)
    json.dump(stats, open(_p("base_stats.json"), "w"), indent=1)
    return df


def build_demand(force: bool = False):
    """DB1B coupons -> nonstop / one-stop quarterly route demand; T-100 monthly segment table."""
    f_ns, f_os, f_t = _p("db1b_nonstop.parquet"), _p("db1b_onestop.parquet"), _p("t100.parquet")
    if all(os.path.exists(f) for f in (f_ns, f_os, f_t)) and not force:
        return pd.read_parquet(f_ns), pd.read_parquet(f_os), (pd.read_parquet(f_t) if os.path.getsize(f_t) else None)
    ns_parts, os_parts, seg_parts = [], [], []
    cols = ["ItinID", "SeqNum", "Origin", "Dest", "Break", "OpCarrier", "Passengers", "Quarter"]
    for q in range(1, 5):
        z = zipfile.ZipFile(os.path.join(RAW_DIR, COUPON.format(q=q)))
        keep = []
        for ch in pd.read_csv(z.open(z.namelist()[0]), usecols=cols, chunksize=2_000_000,
                              dtype={"Break": str, "OpCarrier": str}):
            keep.append(ch[ch.OpCarrier.isin(CARRIERS)])
        c = pd.concat(keep, ignore_index=True)
        c["brk"] = c.Break.fillna("").str.strip() == "X"      # trip break after this coupon
        c = c.sort_values(["ItinID", "SeqNum"], kind="stable")
        # segment pax (all coupons) per carrier/OD -> scaling factor denominator (wiki DB1B_CARRIER_SEGMENT_DEMANDS)
        seg_parts.append(c.groupby(["Quarter", "OpCarrier", "Origin", "Dest"]).Passengers.sum().reset_index())
        # nonstop trips: coupon with a break after it and no open trip before it
        prev = c[["ItinID", "SeqNum", "brk"]].copy()
        prev["SeqNum"] += 1
        c = c.merge(prev.rename(columns={"brk": "prev_brk"}), on=["ItinID", "SeqNum"], how="left")
        # a first coupon (SeqNum==1) or one following a break starts a trip; missing prev row is ambiguous -> require SeqNum==1 or prev_brk
        starts = (c.SeqNum == 1) | (c.prev_brk == True)  # noqa: E712
        ns = c[c.brk & starts]
        ns_parts.append(ns.groupby(["Quarter", "OpCarrier", "Origin", "Dest"]).Passengers.sum().reset_index())
        # one-stop trips: coupon k without break followed by coupon k+1 with break (wiki: first.Break != 'X')
        nxt = c[["ItinID", "SeqNum", "Dest", "OpCarrier", "brk"]].copy()
        nxt["SeqNum"] -= 1
        j = c[~c.brk & starts].merge(nxt, on=["ItinID", "SeqNum"], suffixes=("", "2"))
        j = j[j.brk2 & (j.Origin != j.Dest2)]
        os_parts.append(j.groupby(["Quarter", "Origin", "Dest", "Dest2", "OpCarrier", "OpCarrier2"]).Passengers.sum()
                        .reset_index().rename(columns={"Dest": "Conn", "Dest2": "Dest", "OpCarrier": "C1", "OpCarrier2": "C2"}))
        print("coupons Q%d: rows %d, nonstop trips %d, one-stop rows %d" % (q, len(c), len(ns), len(j)), flush=True)
    ns = pd.concat(ns_parts, ignore_index=True)
    ns.columns = ["q", "carrier", "orig", "dest", "pax"]
    one = pd.concat(os_parts, ignore_index=True)
    one.columns = ["q", "orig", "conn", "dest", "c1", "c2", "pax"]
    seg = pd.concat(seg_parts, ignore_index=True)
    seg.columns = ["q", "carrier", "orig", "dest", "pax"]
    ns.to_parquet(f_ns)
    one.to_parquet(f_os)
    seg.to_parquet(_p("db1b_segments.parquet"))
    tpath = os.path.join(RAW_DIR, T100_NAME)
    if os.path.exists(tpath):
        t = pd.read_csv(tpath, index_col=False)
        t = t[t.UNIQUE_CARRIER.isin(CARRIERS) & (t.CLASS == "F")]
        t = t.groupby(["MONTH", "UNIQUE_CARRIER", "ORIGIN", "DEST"])[["DEPARTURES_PERFORMED", "SEATS", "PASSENGERS"]].sum().reset_index()
        t.columns = ["month", "carrier", "orig", "dest", "deps", "seats", "pax"]
        t.to_parquet(f_t)
    else:
        open(f_t, "wb").close()
    return ns, one, (pd.read_parquet(f_t) if os.path.exists(tpath) else None)


if __name__ == "__main__":
    b = build_otp(force=True)
    print(json.load(open(_p("base_stats.json"))))
