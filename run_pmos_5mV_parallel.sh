#!/usr/bin/env bash
# Launch 6 parallel L-slice runs for sg13_lv_pmos TT 27 5 mV grid.
# Each invocation spawns one ngspice subprocess; peak = 6 ngspice.
set -u
cd "$(dirname "$0")"

COMMON=(--device ihp:sg13_lv_pmos --corners TT --temps 27
        --uniform-grid --vgs-step 0.005 --vds-step 0.005 --vsb-points 15)

pids=()
for r in 0:3 3:6 6:9 9:12 12:15 15:18; do
    tag=${r/:/to}
    python3 run_lut_char_all.py "${COMMON[@]}" --l-range "$r" \
        > "/tmp/pmos_5mV_L${tag}.log" 2>&1 &
    pids+=($!)
    echo "launched L-range $r as pid $!"
done

echo "waiting for ${#pids[@]} jobs..."
fail=0
for p in "${pids[@]}"; do
    if ! wait "$p"; then
        echo "pid $p FAILED"
        fail=1
    fi
done

if (( fail )); then
    echo "ONE OR MORE L-SLICES FAILED — see /tmp/pmos_5mV_L*.log"
    exit 1
fi
echo "all 6 L-slices done OK"
