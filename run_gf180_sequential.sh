#!/usr/bin/env bash
# Sequential GF180 LUT characterization launcher
# Runs one device at a time so all 8 cores go to each job.
# Start: nfet_03v3 is already running (PID given as arg, or auto-detect).

set -euo pipefail
cd "$(dirname "$0")"

LOG=/tmp/gf180_sequential.log
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "GF180 Sequential LUT Characterization"
echo "Started: $(date)"
echo "========================================"

run_device() {
    local dev="$1"
    local log="/tmp/gf180_${dev//\//_}.log"
    echo ""
    echo "--- Starting $dev at $(date) ---"
    python run_lut_char_all.py --device "gf180:${dev}" > "$log" 2>&1
    local rc=$?
    echo "--- Finished $dev at $(date) (exit $rc) ---"
    return $rc
}

# nfet_03v3 is already running; wait for it
EXISTING_PID=${1:-$(pgrep -f "run_lut_char_all.py.*nfet_03v3" | head -1)}
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
