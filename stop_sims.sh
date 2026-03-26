#!/usr/bin/env bash
# Touch the sentinel file that tells run_lut_char_all.py and
# run_gf180_sequential.sh to stop after the current job finishes.
#
# Usage:
#   ./stop_sims.sh          — request graceful stop
#   ./stop_sims.sh --now    — also kill all running ngspice processes immediately
#   ./stop_sims.sh --clear  — remove the STOP file to re-enable future runs

STOP_FILE="$(dirname "$0")/STOP"

case "${1:-}" in
  --clear)
    if [[ -f "$STOP_FILE" ]]; then
        rm "$STOP_FILE"
        echo "STOP file removed. Simulations can run again."
    else
        echo "No STOP file present."
    fi
    ;;
  --now)
    touch "$STOP_FILE"
    echo "STOP file created: $STOP_FILE"
    echo "Killing running ngspice processes..."
    pkill -TERM -u "$USER" ngspice && echo "Sent SIGTERM to ngspice." || echo "No ngspice processes found."
    ;;
  *)
    touch "$STOP_FILE"
    echo "STOP file created: $STOP_FILE"
    echo "Simulations will halt after the current job finishes."
    echo "Run './stop_sims.sh --clear' to re-enable future runs."
    echo "Run './stop_sims.sh --now' to also kill running ngspice immediately."
    ;;
esac
