#!/usr/bin/env bash
# Generate sg13_lv_pmos TT 27 5 mV grid one L at a time (18 sequential
# ngspice runs). Each single-L slice has ~871k sims and fits comfortably
# in run_lut_char_all.py's 2h subprocess timeout, while letting the solo
# ngspice grab as many BSIM4 threads as the box can give it.
set -u
cd "$(dirname "$0")"

COMMON=(--device ihp:sg13_lv_pmos --corners TT --temps 27
        --uniform-grid --vgs-step 0.005 --vds-step 0.005 --vsb-points 15)

# L vector in nm — must match run_lut_char_all.py:232-234 (_IHP_LV_L).
L_NM=(130 140 150 160 170 180 190 200 300 400 500 600 700 800 900 1000 2000 3000)
OUT_DIR=output/uniform

t_start=$(date +%s)
for i in "${!L_NM[@]}"; do
    n=$((i + 1))
    nm=${L_NM[$i]}
    expected="${OUT_DIR}/sg13_lv_pmos_TT_Tp27_uvgs5mV_uvds5mV_vsb15_L${nm}to${nm}nm.mat"

    if [[ -f "$expected" ]]; then
        echo "[skip] L=${nm}nm already exists ($expected)"
        continue
    fi

    echo "=== [$(date +%H:%M:%S)] L-index $i (L=${nm}nm)  $i/18 done ==="
    python3 run_lut_char_all.py "${COMMON[@]}" --l-range "${i}:${n}" \
        > "/tmp/pmos_5mV_seq_L${nm}.log" 2>&1
    rc=$?

    # The python wrapper swallows ngspice TimeoutExpired and exits 0, so
    # also check that the .mat actually landed.
    if [[ $rc -ne 0 ]] || [[ ! -f "$expected" ]]; then
        echo "FAIL: L=${nm}nm  rc=$rc  expected=$expected"
        echo "Tail of log:"
        tail -10 "/tmp/pmos_5mV_seq_L${nm}.log"
        exit 1
    fi
    echo "  done in $(( $(date +%s) - t_start ))s cumulative"
done

echo "all 18 L slices done in $(( $(date +%s) - t_start ))s total"
