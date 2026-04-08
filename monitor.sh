#!/bin/bash
# Progress monitor for all PDKs (sky130, IHP, GF180) — non-uniform and uniform grids.

CORNERS=(TT FF SS SF FS)
TEMPS=(Tm40 Tp27 Tp125)
N_TEMPS=${#TEMPS[@]}

SIM_DIR=/home/canswang/lut_char/sim
OUT_DIR=/home/canswang/lut_char/output

# ---------------------------------------------------------------------------
# get_nvsb <device> <sim_dir>
#   Returns the number of VSB points from the first available .spice file.
#   Falls back to 3 if no netlist exists yet.
# ---------------------------------------------------------------------------
get_nvsb() {
    local device=$1 sim_dir=$2
    local spice
    spice=$(ls "$sim_dir"/techsweep_${device}_*.spice 2>/dev/null | head -1)
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

# ---------------------------------------------------------------------------
# get_base <device> <sim_dir>
#   Returns nL × nVGS × nVDS parsed from the first available .spice file.
#   Works for both non-uniform (fine+coarse) and uniform (single compose) netlists.
# ---------------------------------------------------------------------------
get_base() {
    local device=$1 sim_dir=$2
    local spice
    spice=$(ls "$sim_dir"/techsweep_${device}_*.spice 2>/dev/null | head -1)
    [[ -z "$spice" ]] && echo 0 && return

    awk '
    /compose l_vec/ {
        for (i=1;i<=NF;i++) if ($i=="values") { nL=NF-i; break }
    }
    # Uniform VGS/VDS (single compose vg_vec / vd_vec)
    /compose vg_vec / && !/vg_fine/ && !/vg_coarse/ {
        s=0; p=0; t=0
        for (i=1;i<=NF;i++) {
            if ($i~/^start=/) { sub("start=","",$i); s=$i+0 }
            if ($i~/^stop=/)  { sub("stop=","",$i);  p=$i+0 }
            if ($i~/^step=/)  { sub("step=","",$i);  t=$i+0 }
        }
        if (t>0) nvgs = int((p-s)/t + 0.5) + 1
    }
    /compose vd_vec / && !/vd_fine/ && !/vd_coarse/ {
        s=0; p=0; t=0
        for (i=1;i<=NF;i++) {
            if ($i~/^start=/) { sub("start=","",$i); s=$i+0 }
            if ($i~/^stop=/)  { sub("stop=","",$i);  p=$i+0 }
            if ($i~/^step=/)  { sub("step=","",$i);  t=$i+0 }
        }
        if (t>0) nvds = int((p-s)/t + 0.5) + 1
    }
    # Non-uniform: accumulate fine + coarse points
    /compose vg_fine_vec/ {
        s=0; p=0; t=0
        for (i=1;i<=NF;i++) {
            if ($i~/^start=/) { sub("start=","",$i); s=$i+0 }
            if ($i~/^stop=/)  { sub("stop=","",$i);  p=$i+0 }
            if ($i~/^step=/)  { sub("step=","",$i);  t=$i+0 }
        }
        if (t>0) nvgs += int((p-s)/t + 0.5) + 1
    }
    /compose vg_coarse_vec/ {
        s=0; p=0; t=0
        for (i=1;i<=NF;i++) {
            if ($i~/^start=/) { sub("start=","",$i); s=$i+0 }
            if ($i~/^stop=/)  { sub("stop=","",$i);  p=$i+0 }
            if ($i~/^step=/)  { sub("step=","",$i);  t=$i+0 }
        }
        if (t>0) nvgs += int((p-s)/t + 0.5) + 1
    }
    /compose vd_fine_vec/   { nvds += 60 }
    /compose vd_coarse_vec/ {
        s=0; p=0; t=0
        for (i=1;i<=NF;i++) {
            if ($i~/^start=/) { sub("start=","",$i); s=$i+0 }
            if ($i~/^stop=/)  { sub("stop=","",$i);  p=$i+0 }
            if ($i~/^step=/)  { sub("step=","",$i);  t=$i+0 }
        }
        if (t>0) nvds += int((p-s)/t + 0.5) + 1
    }
    END { print nL * nvgs * nvds }
    ' "$spice"
}

# ---------------------------------------------------------------------------
# show_pdk <label> <sim_dir> <out_dir> <device> [<device> ...]
#   Prints per-corner progress for each device in the PDK group.
# ---------------------------------------------------------------------------
show_pdk() {
    local label=$1 sim_dir=$2 out_dir=$3
    shift 3
    local devices=("$@")

    echo "  ── $label ──────────────────────────────────────"

    for device in "${devices[@]}"; do
        local base nvsb exp total_mats total_jobs
        base=$(get_base "$device" "$sim_dir")
        nvsb=$(get_nvsb  "$device" "$sim_dir")
        exp=$((base * nvsb))
        total_mats=$(ls "$out_dir"/${device}_*.mat 2>/dev/null | wc -l)
        total_jobs=$((${#CORNERS[@]} * N_TEMPS))

        printf "  %-22s  [%2d/%d done, nVSB=%s]\n" \
               "$device" "$total_mats" "$total_jobs" "$nvsb"

        for corner in "${CORNERS[@]}"; do
            local corner_mats corner_txts
            corner_mats=$(ls "$out_dir"/${device}_${corner}_*.mat 2>/dev/null | wc -l)
            corner_txts=($(ls "$sim_dir"/techsweep_${device}_${corner}_*.txt 2>/dev/null))

            if [[ $corner_mats -eq $N_TEMPS ]]; then
                printf "    %-4s ✓ done\n" "$corner"
            elif [[ ${#corner_txts[@]} -gt 0 ]]; then
                local temp_parts=""
                for temp in "${TEMPS[@]}"; do
                    local txt="$sim_dir/techsweep_${device}_${corner}_${temp}.txt"
                    local mat="$out_dir/${device}_${corner}_${temp}.mat"
                    if [[ -f "$mat" ]]; then
                        temp_parts+=" ${temp}:✓"
                    elif [[ -f "$txt" ]]; then
                        local rows pct
                        rows=$(wc -l < "$txt" 2>/dev/null || echo 0)
                        if [[ $exp -gt 0 ]]; then
                            pct=$(awk "BEGIN{printf \"%d\", $rows*100/$exp}")
                        else
                            pct="?"
                        fi
                        temp_parts+=" ${temp}:${pct}%"
                    fi
                done
                printf "    %-4s ▶ active —%s\n" "$corner" "$temp_parts"
            else
                printf "    %-4s   waiting\n" "$corner"
            fi
        done
    done
    echo ""
}

# ---------------------------------------------------------------------------
# show_grid_section <heading> <sim_dir> <out_dir>
#   Skips the section silently if sim_dir has no techsweep_*.spice files.
# ---------------------------------------------------------------------------
show_grid_section() {
    local heading=$1 sim_dir=$2 out_dir=$3

    # Skip entire section if no netlists exist yet
    ls "$sim_dir"/techsweep_*.spice 2>/dev/null | grep -q . || return

    echo "$heading"
    echo ""

    show_pdk "sky130" "$sim_dir" "$out_dir" \
        nfet_01v8 pfet_01v8 nfet_01v8_lvt pfet_01v8_lvt

    show_pdk "IHP" "$sim_dir" "$out_dir" \
        sg13_lv_nmos sg13_lv_pmos sg13_hv_nmos sg13_hv_pmos

    show_pdk "GF180" "$sim_dir" "$out_dir" \
        nfet_03v3 pfet_03v3 nfet_05v0 pfet_05v0

    echo "  Output .mat files ($(ls "$out_dir"/*.mat 2>/dev/null | wc -l) total):"
    ls -lh "$out_dir"/*.mat 2>/dev/null | awk '{print "    "$9, $5}'
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "$(date '+%H:%M:%S') — Active ngspice: $(ps aux | grep ngspice | grep -cv grep) processes"
echo ""

seq_log=/tmp/gf180_sequential.log
if [[ -f "$seq_log" ]]; then
    echo "Pipeline: $(tail -1 $seq_log)"
    echo ""
fi

show_grid_section \
    "═══ Non-uniform grid (sim/ → output/) ═══" \
    "$SIM_DIR" "$OUT_DIR"

show_grid_section \
    "═══ Uniform grid (sim/uniform/ → output/uniform/) ═══" \
    "$SIM_DIR/uniform" "$OUT_DIR/uniform"
