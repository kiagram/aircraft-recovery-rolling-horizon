#!/usr/bin/env bash
# Stage 2 rerun of one job block: FH/FH-PI lines with at most 4 workers (CLAUDE.md), all other lines in parallel with
# $3 workers.  Resumable (existing result files are skipped by experiments.py).
#   bash audit/stage_2/run_block.sh <job file> <results dir> <rh workers> [fh workers, <= 4]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ARP="$HERE/../../arp_rh"
JOBS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
mkdir -p "$2"
OUT="$(cd "$2" && pwd)"          # absolute: batch.py runs from arp_rh/
W="${3:-8}"
FW="${4:-4}"
TMP="$HERE/jobs/.tmp_$(basename "$1" .txt)"
mkdir -p "$TMP"
grep -E -- "--policy (fh|fhpi)( |$)" "$JOBS" > "$TMP/fh.txt" || true
grep -v -E -- "--policy (fh|fhpi)( |$)" "$JOBS" > "$TMP/rest.txt" || true
cd "$ARP"
echo "block $(basename "$1"): $(wc -l < "$TMP/fh.txt") FH, $(wc -l < "$TMP/rest.txt") other jobs -> $OUT"
[ -s "$TMP/fh.txt" ] && python batch.py "$TMP/fh.txt" "$FW" "$OUT" &
[ -s "$TMP/rest.txt" ] && python batch.py "$TMP/rest.txt" "$W" "$OUT" &
wait
echo "block $(basename "$1") done"
