#!/bin/bash
# Quick progress check for GF180 LUT sweep
# Shows per-corner status for each device.

# Base rows per job (nL × nVGS × nVDS × 1VSB) — VSB factor detected from netlist
declare -A BASE
BASE[nfet_03v3]=204204   # 12L × 187VGS × 91VDS
BASE[pfet_03v3]=204204
BASE[nfet_05v0]=174804   # 7L  × 231VGS × 108VDS
BASE[pfet_05v0]=199776   # 8L  × 231VGS × 108VDS

CORNERS=(TT FF SS SF FS)
TEMPS=(Tm40 Tp27 Tp125)
N_TEMPS=${#TEMPS[@]}

SIM_DIR=/home/canswang/lut_char/sim
OUT_DIR=/home/canswang/lut_char/output

# Parse nVSB from a device's netlist (handles both values and start/stop/step forms)
get_nvsb() {
    local device=$1
    local spice
    spice=$(ls "$SIM_DIR"/techsweep_${device}_*.spice 2>/dev/null | head -1)
    [[ -z "$spice" ]] && echo 3 && return
    grep -m1 'compose vsb_vec' "$spice" | awk '{
        for (i=1; i<=NF; i++) {
            if ($i == "values") { print NF-i; exit }
            if ($i ~ /^start=/) { sub("start=","",$i); start = $i+0 }
            if ($i ~ /^stop=/)  { sub("stop=","",$i);  stop  = $i+0 }
            if ($i ~ /^step=/)  { sub("step=","",$i);  step  = $i+0 }
        }
        if (step != 0) { n = int((stop-start)/step + 0.5) + 1; if (n<1) n=1; print n }
        else { print 1 }
    }'
}

echo "$(date '+%H:%M:%S') — Active ngspice: $(ps aux | grep ngspice | grep -cv grep) processes"
echo ""

seq_log=/tmp/gf180_sequential.log
if [[ -f "$seq_log" ]]; then
    echo "Pipeline: $(tail -1 $seq_log)"
    echo ""
fi

for device in nfet_03v3 pfet_03v3 nfet_05v0 pfet_05v0; do
    base=${BASE[$device]}
    nvsb=$(get_nvsb "$device")
    exp=$((base * nvsb))
    total_mats=$(ls "$OUT_DIR"/${device}_*.mat 2>/dev/null | wc -l)
    total_jobs=$((${#CORNERS[@]} * N_TEMPS))

    echo "  ${device}  [${total_mats}/${total_jobs} done, nVSB=${nvsb}]"

    for corner in "${CORNERS[@]}"; do
        corner_mats=$(ls "$OUT_DIR"/${device}_${corner}_*.mat 2>/dev/null | wc -l)
        corner_txts=($(ls "$SIM_DIR"/techsweep_${device}_${corner}_*.txt 2>/dev/null))

        if [[ $corner_mats -eq $N_TEMPS ]]; then
            printf "    %-4s ✓ done\n" "$corner"
        elif [[ ${#corner_txts[@]} -gt 0 ]]; then
            # Build per-temp progress string
            temp_parts=""
            for temp in "${TEMPS[@]}"; do
                txt="$SIM_DIR/techsweep_${device}_${corner}_${temp}.txt"
                mat="$OUT_DIR/${device}_${corner}_${temp}.mat"
                if [[ -f "$mat" ]]; then
                    temp_parts+=" ${temp}:✓"
                elif [[ -f "$txt" ]]; then
                    rows=$(wc -l < "$txt" 2>/dev/null || echo 0)
                    pct=$(awk "BEGIN{printf \"%d\", $rows*100/$exp}")
                    temp_parts+=" ${temp}:${pct}%"
                fi
            done
            printf "    %-4s ▶ active —%s\n" "$corner" "$temp_parts"
        else
            printf "    %-4s   waiting\n" "$corner"
        fi
    done
    echo ""
done

echo "Output .mat files ($(ls "$OUT_DIR"/*.mat 2>/dev/null | wc -l) total):"
ls -lh "$OUT_DIR"/*.mat 2>/dev/null | awk '{print "  "$9, $5}'
