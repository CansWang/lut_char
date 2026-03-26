#!/usr/bin/env bash
# Sequential GF180 LUT characterization launcher
# Runs one device at a time so all cores go to each job.
#
# Usage:
#   ./run_gf180_sequential.sh [PID] [--workers N]
#
#   PID        (optional) PID of an already-running nfet_03v3 job to wait for.
#   --workers N  cap the number of parallel ngspice processes per device
#                (default: all CPUs).  Passed through to run_lut_char_all.py.

set -euo pipefail
cd "$(dirname "$0")"

LOG=/tmp/gf180_sequential.log
exec > >(tee -a "$LOG") 2>&1

# ── Parse arguments ────────────────────────────────────────────────────────────
WORKERS_ARG=""   # empty → use run_lut_char_all.py default
EXISTING_PID=""

for arg in "$@"; do
    case "$arg" in
        --workers=*) WORKERS_ARG="--workers ${arg#*=}" ;;
        --workers)   : ;;   # value consumed in next iteration below
        [0-9]*)      EXISTING_PID="$arg" ;;
    esac
done
# Handle "--workers N" (two separate tokens)
prev=""
for arg in "$@"; do
    [[ "$prev" == "--workers" ]] && WORKERS_ARG="--workers $arg"
    prev="$arg"
done

echo "========================================"
echo "GF180 Sequential LUT Characterization"
echo "Started: $(date)"
[[ -n "$WORKERS_ARG" ]] && echo "Workers cap: ${WORKERS_ARG#--workers }"
echo "========================================"

STOP_FILE="$(dirname "$0")/STOP"

check_stop() {
    if [[ -f "$STOP_FILE" ]]; then
        echo "STOP file found ($STOP_FILE) — aborting pipeline at $(date)"
        exit 0
    fi
}

run_device() {
    local dev="$1"
    local log="/tmp/gf180_${dev//\//_}.log"
    check_stop
    echo ""
    echo "--- Starting $dev at $(date) ---"
    # shellcheck disable=SC2086
    python run_lut_char_all.py --device "gf180:${dev}" $WORKERS_ARG > "$log" 2>&1
    local rc=$?
    echo "--- Finished $dev at $(date) (exit $rc) ---"
    return $rc
}

# nfet_03v3 is already running; wait for it
[[ -z "$EXISTING_PID" ]] && EXISTING_PID=$(pgrep -f "run_lut_char_all.py.*nfet_03v3" | head -1 || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "Waiting for nfet_03v3 (PID $EXISTING_PID) to finish..."
    while kill -0 "$EXISTING_PID" 2>/dev/null; do sleep 30; done
    echo "nfet_03v3 done at $(date)"
else
    run_device nfet_03v3
fi

run_device pfet_03v3
run_device nfet_05v0
run_device pfet_05v0

echo ""
echo "========================================"
echo "All GF180 devices complete: $(date)"
echo "========================================"
