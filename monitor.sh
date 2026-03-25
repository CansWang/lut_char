#!/bin/bash
# Quick progress check for GF180 LUT sweep
EXPECTED_3V3=612612   # 12L × 187VGS × 91VDS × 3VSB (per PVT job)
EXPECTED_5V0_N=524412 # 7L × 231VGS × 108VDS × 3VSB
EXPECTED_5V0_P=599328 # 8L × 231VGS × 108VDS × 3VSB

declare -A EXPECTED
EXPECTED[nfet_03v3]=$EXPECTED_3V3
EXPECTED[pfet_03v3]=$EXPECTED_3V3
EXPECTED[nfet_05v0]=$EXPECTED_5V0_N
EXPECTED[pfet_05v0]=$EXPECTED_5V0_P

echo "$(date '+%H:%M:%S') — Active ngspice: $(ps aux | grep ngspice | grep -cv grep) processes"
echo ""

# Sequential pipeline status
seq_log=/tmp/gf180_sequential.log
if [[ -f "$seq_log" ]]; then
    echo "Pipeline: $(tail -1 $seq_log)"
fi
echo ""

for device in nfet_03v3 pfet_03v3 nfet_05v0 pfet_05v0; do
    exp=${EXPECTED[$device]}
    files=($(ls /home/cwang/lut_char/sim/techsweep_${device}_*.txt 2>/dev/null))
    total_rows=0; n_files=${#files[@]}
    max_rows=0
    for f in "${files[@]}"; do
        rows=$(wc -l < "$f" 2>/dev/null || echo 0)
        total_rows=$((total_rows + rows))
        [[ $rows -gt $max_rows ]] && max_rows=$rows
    done
    mats=$(ls /home/cwang/lut_char/output/${device}_*.mat 2>/dev/null | wc -l)
    if [[ $n_files -gt 0 && $exp -gt 0 ]]; then
        pct=$(awk "BEGIN{printf \"%.1f\", $max_rows*100/$exp}")
        echo "  $device: $n_files active, $mats/15 .mat done, leading job ${pct}% (${max_rows}/${exp} rows)"
    else
        echo "  $device: waiting — $mats/15 .mat done"
    fi
done

echo ""
echo "Output .mat files ($(ls /home/cwang/lut_char/output/*.mat 2>/dev/null | wc -l) total):"
ls -lh /home/cwang/lut_char/output/*.mat 2>/dev/null | awk '{print "  "$9, $5}'
