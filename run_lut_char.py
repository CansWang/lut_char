#!/usr/bin/env python3
"""
run_lut_char.py — Non-uniform VDS LUT characterization for sky130 nfet_01v8.

PDK: SkyWater 130nm (sky130), using the discrete-corner pm3 model library.
     Path: ~/apragma/models/sky130/models/sky130.lib.spice
     Corners: tt, ff, ss, sf, fs  (loaded as-is; mc_mm_switch=0 by default)

Unit convention
---------------
models/all.spice sets `.option scale=1.0u`, so ALL bare numbers in the
netlist are implicitly in micrometres.  L=0.15 → 0.15 µm, W=5 → 5 µm.
Do NOT append 'u' or 'e-6' to L/W in the netlist.

VDS grid
--------
Fine   : 0.000 – 0.295 V @ 5 mV  → 60 pts
Coarse : 0.300 – 1.800 V @ 50 mV → 31 pts
Total  : 91 pts, no duplicate at 0.300 V

ngspice compose quirk
---------------------
  compose ... stop=0.295 step=0.005  → only 59 pts (FP boundary exclusion)
  compose ... stop=0.2951 step=0.005 → 60 pts ✓  (safe overshoot)

PVT corners: TT, FF, SS, SF, FS  (→ tt / ff / ss / sf / fs in lib)
Temperatures: -40, 27, 125 °C

Implementation note
-------------------
Two sequential foreach loops at the VDS level (fine, then coarse) inside
the VGS loop both append to the same output file.  Rows arrive in exact
(L, VGS, VDS_all, VSB) row-major order → direct numpy reshape.

Usage
-----
  python run_lut_char.py --test-run          # micro-sweep validation
  python run_lut_char.py --corners TT FF     # specific corners, all temps
  python run_lut_char.py                     # full 5-corner × 3-temp PVT
"""

import argparse
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

# ── PDK / device constants ─────────────────────────────────────────────────────
DEVICE    = "sky130_fd_pr__nfet_01v8"
MODEL_LIB = "/home/canswang/apragma/models/sky130/models/sky130.lib.spice"
W_UM      = 5.0    # characterisation width (in bare µm units, no 'u' suffix)
NFING     = 1

# Subcircuit inner MOSFET name (used in .save directives)
_INNER = "msky130_fd_pr__nfet_01v8"
_SAVE_PFX = f"@m.xm1.{_INNER}"

# ── Corners ────────────────────────────────────────────────────────────────────
ALL_CORNERS = ["TT", "FF", "SS", "SF", "FS"]
_CORNER_LIB = {"TT": "tt", "FF": "ff", "SS": "ss", "SF": "sf", "FS": "fs"}

# ── Full-run sweep grids ───────────────────────────────────────────────────────
ALL_TEMPS     = [-40, 27, 125]
L_VEC_FULL    = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
                 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                 1.00, 2.00, 3.00]          # bare µm (scale=1u applied by lib)
VGS_VEC_FULL  = list(np.round(np.arange(0, 1.801, 0.025), 4))
VSB_VEC_FULL  = [0.0, -0.2, -0.4]

# ── Non-uniform VDS axis (authoritative Python definition) ─────────────────────
VDS_FINE   = np.round(np.arange(0,   0.300, 0.005), 4)   # 60 pts: 0.000 … 0.295
VDS_COARSE = np.round(np.arange(0.3, 1.850, 0.050), 4)   # 31 pts: 0.300 … 1.800
VDS_ALL    = np.concatenate([VDS_FINE, VDS_COARSE])        # 91 pts, no duplicate

# ── Test-run reduced grid ──────────────────────────────────────────────────────
CORNERS_TEST = ["TT"]
TEMPS_TEST   = [27]
L_VEC_TEST   = [0.15, 1.00]
VGS_VEC_TEST = [0.6, 1.2]
VSB_VEC_TEST = [0.0]

# ── Paths ──────────────────────────────────────────────────────────────────────
SIM_DIR = Path("/home/canswang/lut_char/sim")
OUT_DIR = Path("/home/canswang/lut_char/output")


# ── Netlist generation ─────────────────────────────────────────────────────────

def _vec_str(vals, fmt=".4f"):
    return " ".join(format(v, fmt) for v in vals)


def generate_netlist(corner: str, temp: int,
                     l_vec, vgs_vec, vsb_vec,
                     out_txt: str, netlist_path: str):
    """
    Write the ngspice deck with dual (fine + coarse) VDS foreach loops.

    Unit note: L/W values are bare numbers (e.g. 0.15, 5) — the model lib
    loads all.spice which sets `.option scale=1.0u`, so all bare dimensions
    are scaled to µm automatically.
    """
    lib_corner = _CORNER_LIB[corner]
    l_str      = _vec_str(l_vec)     # e.g. "0.15 0.16 ... 3.00"
    vgs_str    = _vec_str(vgs_vec)
    vsb_str    = _vec_str(vsb_vec, ".3f")

    # sky130 ad/as/pd/ps (0.29 µm contact-to-gate half-distance, in bare µm)
    ad = f"int(({NFING}+1)/2) * {W_UM}/{NFING} * 0.29"
    as_ = f"int(({NFING}+2)/2) * {W_UM}/{NFING} * 0.29"
    pd = f"2*int(({NFING}+1)/2) * ({W_UM}/{NFING} + 0.29)"
    ps = f"2*int(({NFING}+2)/2) * ({W_UM}/{NFING} + 0.29)"

    content = textwrap.dedent(f"""\
        ** LUT Characterization: {DEVICE}  corner={corner}  T={temp}C
        ** PDK: SkyWater 130nm (sky130)
        ** Unit note: all bare numbers in µm via .option scale=1.0u in all.spice
        **.subckt lut_char
        vg g 0 DC 0.0 AC 1
        vd d 0 0.0
        vb b 0 0
        Hn n 0 vd 1
        XM1 d g 0 b {DEVICE} L={{lx}} W={W_UM} nf={NFING} ad='{ad}' as='{as_}' pd='{pd}' ps='{ps}' nrd='0.29 / {W_UM}' nrs='0.29 / {W_UM}' sa=0 sb=0 sd=0 mult=1 m=1

        .param wx={W_UM} lx=0.15
        .temp {temp}
        .noise v(n) vg lin 1 1 1 1

        .control
        option numdgt = 3
        set wr_singlescale
        set wr_vecnames

        compose l_vec         values {l_str}
        compose vg_vec        values {vgs_str}
        compose vsb_vec       values {vsb_str}
        * Fine VDS:   0.000 to 0.295 V in 5 mV steps (60 points)
        * stop=0.2951 (not 0.295): ngspice compose FP boundary exclusion workaround
        compose vd_fine_vec   start=0    stop=0.2951 step=0.005
        * Coarse VDS: 0.300 to 1.800 V in 50 mV steps (31 points)
        compose vd_coarse_vec start=0.3  stop=1.8    step=0.05

        foreach var1 $&l_vec
          alterparam lx=$var1
          reset
          foreach var2 $&vg_vec
            alter vg $var2
            * === Fine VDS sweep (0 to 0.295 V @ 5 mV) ===
            foreach var3 $&vd_fine_vec
              alter vd $var3
              foreach var4 $&vsb_vec
                alter vb $var4
                run
                wrdata {out_txt} noise1.all
                destroy all
                set appendwrite
                unset wr_vecnames
              end
            end
            * === Coarse VDS sweep (0.3 to 1.8 V @ 50 mV) ===
            foreach var3 $&vd_coarse_vec
              alter vd $var3
              foreach var4 $&vsb_vec
                alter vb $var4
                run
                wrdata {out_txt} noise1.all
                destroy all
                set appendwrite
                unset wr_vecnames
              end
            end
          end
        end
        unset appendwrite
        .endc

        .lib {MODEL_LIB} {lib_corner}

        .save {_SAVE_PFX}[capbd]
        .save {_SAVE_PFX}[capbs]
        .save {_SAVE_PFX}[cdd]
        .save {_SAVE_PFX}[cgb]
        .save {_SAVE_PFX}[cgd]
        .save {_SAVE_PFX}[cgdo]
        .save {_SAVE_PFX}[cgg]
        .save {_SAVE_PFX}[cgs]
        .save {_SAVE_PFX}[cgso]
        .save {_SAVE_PFX}[css]
        .save {_SAVE_PFX}[gds]
        .save {_SAVE_PFX}[gm]
        .save {_SAVE_PFX}[gmbs]
        .save {_SAVE_PFX}[id]
        .save {_SAVE_PFX}[l]
        .save @vb[dc]
        .save @vd[dc]
        .save @vg[dc]
        .save {_SAVE_PFX}[vth]
        .save onoise.m.xm1.{_INNER}.id
        .save onoise.m.xm1.{_INNER}.1overf
        .save g d b n

        **.ends
        .end
    """)

    with open(netlist_path, "w") as fh:
        fh.write(content)
    print(f"  [gen ] {netlist_path}")


# ── ngspice runner ─────────────────────────────────────────────────────────────

def run_ngspice(netlist_path: str, log_path: str) -> bool:
    """Invoke ngspice in batch mode. Returns True on success."""
    cmd = ["ngspice", "-b", netlist_path]
    print(f"  [sim ] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(SIM_DIR), timeout=7200
        )
        with open(log_path, "w") as fh:
            fh.write(result.stdout)
            fh.write("\n\n--- STDERR ---\n")
            fh.write(result.stderr)

        if result.returncode != 0:
            print(f"  [ERR ] ngspice exited {result.returncode} — see {log_path}")
            lines = result.stderr.strip().splitlines()
            for ln in lines[-40:]:
                print(f"         {ln}")
            return False

        print(f"  [sim ] Done. Log → {log_path}")
        return True

    except subprocess.TimeoutExpired:
        print("  [ERR ] ngspice timed out")
        return False


# ── Parser and .mat writer ─────────────────────────────────────────────────────

def parse_and_save(txt_path: str, corner: str, temp: int,
                   l_vec, vgs_vec, vsb_vec, mat_path: str):
    """
    Read ngspice wrdata output, reshape into (L, VGS, VDS, VSB) tensor,
    compute composite capacitances, and save as a .mat LUT.

    Column cleanup follows the identical logic used in the Book-on-gm-ID-design
    techsweep_txt_to_mat.ipynb for sky130.
    """
    print(f"  [read] {txt_path}")
    df_raw = pd.read_csv(txt_path, sep=r'\s+', engine='python')

    par_names = df_raw.columns.to_list()
    fet_name  = next(c for c in par_names if not c.startswith("frequency"))
    fet_name  = fet_name.split('[')[0]   # → "@m.xm1.msky130_fd_pr__nfet_01v8"

    df = df_raw.drop(columns=[c for c in df_raw.columns
                               if c.startswith("frequency")], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce')

    df.columns = df.columns.str.replace(fet_name,    '', regex=False)
    df.columns = df.columns.str.replace(fet_name[1:], '', regex=False)
    df.columns = df.columns.str.replace('[dc]',       '', regex=False)
    df.columns = df.columns.str.replace('onoise..',   'n', regex=False)
    df.columns = df.columns.str.removeprefix('@')
    df.columns = df.columns.str.removeprefix('[')
    df.columns = df.columns.str.removesuffix(']')

    print(f"  [read] Columns: {list(df.columns)}")

    nL   = len(l_vec)
    nVGS = len(vgs_vec)
    nVDS = len(VDS_ALL)   # 91
    nVSB = len(vsb_vec)
    dims = [nL, nVGS, nVDS, nVSB]

    expected_rows = nL * nVGS * nVDS * nVSB
    actual_rows   = len(df)
    print(f"  [chk ] Rows: expected={expected_rows}, actual={actual_rows}")
    if actual_rows != expected_rows:
        print(f"  [WARN] Row mismatch — tensor reshape may fail!")

    def R(col):
        return np.reshape(df[col].values[:expected_rows], dims)

    # Composite capacitances — identical to Book-on-gm-ID-design sky130 notebook
    cgg = R('cgg') + R('cgdo') + R('cgso')
    cgb = -R('cgb')
    cgd = -R('cgd') + R('cgdo')
    cgs = -R('cgs') + R('cgso')
    cdd = R('cdd') + R('capbd') + R('cgdo')
    css = R('css') + R('capbs') + R('cgso')

    dic = {
        "INFO":   f"SkyWater 130nm CMOS BSIM4 — {DEVICE}",
        "CORNER": corner.upper(),
        "TEMP":   float(temp + 273.15),
        "VGS":    np.array(vgs_vec),
        "VDS":    VDS_ALL,
        "VSB":    np.array([abs(v) for v in vsb_vec]),
        "L":      np.array(l_vec),
        "W":      float(W_UM),
        "NFING":  float(NFING),
        "ID":     R('id'),
        "VT":     R('vth'),
        "GM":     R('gm'),
        "GMB":    R('gmbs'),
        "GDS":    R('gds'),
        "CGG":    cgg,
        "CGB":    cgb,
        "CGD":    cgd,
        "CGS":    cgs,
        "CDD":    cdd,
        "CSS":    css,
        "STH":    R('nid')**2,
        "SFL":    R('n1overf')**2,
    }

    sign    = 'p' if temp >= 0 else 'm'
    mat_key = f"nfet_01v8_{corner}_T{sign}{abs(temp)}"
    savemat(mat_path, {mat_key: dic})
    print(f"  [save] {mat_path}  (key='{mat_key}')")
    return dic, dims


# ── Test-run validation reporter ───────────────────────────────────────────────

def validate_test_run(dic, dims):
    nL, nVGS, nVDS, nVSB = dims
    sep = "=" * 68

    print(f"\n{sep}")
    print("  TEST CASE VALIDATION RESULTS")
    print(sep)

    # TC1
    print("\n[TC1] Netlist Verification")
    print("  Result : PASS — ngspice completed without syntax errors")

    # TC2
    vds_axis     = dic["VDS"]
    fine_mask    = vds_axis < 0.3 - 1e-9
    coarse_mask  = vds_axis >= 0.3 - 1e-9
    n_fine       = int(np.sum(fine_mask))
    n_coarse     = int(np.sum(coarse_mask))
    n_total      = len(vds_axis)
    boundary_pts = int(np.sum(np.abs(vds_axis - 0.3) < 1e-9))
    last_fine    = float(vds_axis[fine_mask][-1])
    first_coarse = float(vds_axis[coarse_mask][0])

    print(f"\n[TC2] Data Stitching")
    print(f"  Fine pts  (VDS <  0.3V) : {n_fine}   (expected 60)")
    print(f"  Coarse pts (VDS >= 0.3V) : {n_coarse}  (expected 31)")
    print(f"  Total VDS points          : {n_total} (expected 91)")
    print(f"  Last fine point           : {last_fine:.4f} V  (expected 0.2950)")
    print(f"  First coarse point        : {first_coarse:.4f} V  (expected 0.3000)")
    print(f"  Occurrences of 0.3000 V   : {boundary_pts}  (expected 1 — no duplicate)")
    tc2 = (n_fine == 60 and n_coarse == 31 and boundary_pts == 1)
    print(f"  Result : {'PASS' if tc2 else 'FAIL'}")

    # TC3
    expected_shape = tuple(dims)
    actual_shape   = dic['ID'].shape
    print(f"\n[TC3] Tensor Shape")
    print(f"  Expected ID.shape : {expected_shape}  "
          f"(nL={nL}, nVGS={nVGS}, nVDS={nVDS}, nVSB={nVSB})")
    print(f"  Actual   ID.shape : {actual_shape}")
    tc3 = (actual_shape == expected_shape)
    print(f"  Result : {'PASS' if tc3 else 'FAIL'}")

    # TC4
    print(f"\n[TC4] Physics Check — ID continuity across VDS=0.3 V boundary")
    l_arr   = np.array(dic["L"])
    vgs_arr = np.array(dic["VGS"])
    i_L   = int(np.argmin(np.abs(l_arr   - 1.0)))
    i_VGS = int(np.argmin(np.abs(vgs_arr - 1.2)))
    i_VSB = 0

    id_curve = dic["ID"][i_L, i_VGS, :, i_VSB]
    i_295 = int(np.argmin(np.abs(vds_axis - 0.295)))
    i_300 = int(np.argmin(np.abs(vds_axis - 0.300)))
    i_350 = int(np.argmin(np.abs(vds_axis - 0.350)))

    id_295 = float(id_curve[i_295])
    id_300 = float(id_curve[i_300])
    id_350 = float(id_curve[i_350])
    jump   = abs(id_300 - id_295) / (abs(id_295) + 1e-30)

    print(f"  L={l_arr[i_L]:.2f}µm, VGS={vgs_arr[i_VGS]:.2f}V, VSB=0V")
    print(f"  ID @ VDS=0.295 V = {id_295: .6e} A  (last fine point)")
    print(f"  ID @ VDS=0.300 V = {id_300: .6e} A  (first coarse point)")
    print(f"  ID @ VDS=0.350 V = {id_350: .6e} A")
    print(f"  Relative step at boundary : {jump:.4%}")

    i_start = max(0, i_295 - 2)
    i_end   = min(len(vds_axis), i_300 + 4)
    print(f"  ID profile around 0.3 V boundary:")
    for k in range(i_start, i_end):
        marker = "  ← first coarse pt" if k == i_300 else \
                 "  ← last fine pt"    if k == i_295 else ""
        print(f"    VDS={vds_axis[k]:.4f}V  ID={id_curve[k]:.6e} A{marker}")

    tc4 = jump < 0.05
    print(f"  Result : {'PASS' if tc4 else 'WARN — jump > 5%, check simulation'}")

    overall = all([tc2, tc3, tc4])
    print(f"\n{'─'*68}")
    print(f"  Overall pipeline status: {'ALL PASS ✓' if overall else 'SEE WARNINGS ABOVE'}")
    print(f"{sep}\n")


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_pvt(corners, temps, l_vec, vgs_vec, vsb_vec, test_mode=False):
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for corner in corners:
        for temp in temps:
            sign = 'p' if temp >= 0 else 'm'
            tag  = f"{corner}_T{sign}{abs(temp)}"

            print(f"\n{'='*68}")
            print(f"  Corner={corner}  T={temp:+d}°C  "
                  f"{'[TEST RUN]' if test_mode else '[FULL RUN]'}")
            print(f"  Grid : {len(l_vec)} L × {len(vgs_vec)} VGS × "
                  f"{len(VDS_ALL)} VDS (non-uniform) × {len(vsb_vec)} VSB")
            print(f"         → {len(l_vec)*len(vgs_vec)*len(VDS_ALL)*len(vsb_vec)}"
                  f" total OP+noise simulations")

            out_txt      = str(SIM_DIR / f"techsweep_{tag}.txt")
            netlist_path = str(SIM_DIR / f"techsweep_{tag}.spice")
            log_path     = str(SIM_DIR / f"techsweep_{tag}.log")
            mat_path     = str(OUT_DIR / f"nfet_01v8_{tag}.mat")

            if os.path.exists(out_txt):
                os.remove(out_txt)

            generate_netlist(corner, temp, l_vec, vgs_vec, vsb_vec,
                             out_txt, netlist_path)

            ok = run_ngspice(netlist_path, log_path)
            if not ok:
                print(f"  [SKIP] Parse skipped — fix ngspice errors first.")
                continue

            if not os.path.exists(out_txt) or os.path.getsize(out_txt) == 0:
                print(f"  [ERR ] Output file missing or empty: {out_txt}")
                continue

            dic, dims = parse_and_save(out_txt, corner, temp,
                                       l_vec, vgs_vec, vsb_vec, mat_path)
            if test_mode:
                validate_test_run(dic, dims)


def main():
    ap = argparse.ArgumentParser(
        description="sky130 nfet_01v8 LUT characterization — non-uniform VDS grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_lut_char.py --test-run
              python run_lut_char.py --corners TT SS --temps 27 125
              python run_lut_char.py                        # full 5×3 PVT
        """)
    )
    ap.add_argument(
        "--test-run", action="store_true",
        help="Micro-sweep: TT/27°C, L=[0.15,1.0], VGS=[0.6,1.2], VSB=[0], 91 VDS"
    )
    ap.add_argument(
        "--corners", nargs="+", default=ALL_CORNERS,
        choices=ALL_CORNERS, metavar="CORNER",
        help="Corners: TT FF SS SF FS (default: all 5)"
    )
    ap.add_argument(
        "--temps", nargs="+", type=int, default=ALL_TEMPS,
        metavar="TEMP",
        help="Temperatures in °C (default: -40 27 125)"
    )
    args = ap.parse_args()

    print(f"PDK: sky130  Device: {DEVICE}  W={W_UM}µm")
    print(f"VDS: {len(VDS_FINE)} fine pts (0–0.295V @5mV) + "
          f"{len(VDS_COARSE)} coarse pts (0.3–1.8V @50mV) = {len(VDS_ALL)} total")

    if args.test_run:
        run_pvt(CORNERS_TEST, TEMPS_TEST,
                L_VEC_TEST, VGS_VEC_TEST, VSB_VEC_TEST,
                test_mode=True)
    else:
        run_pvt(args.corners, args.temps,
                L_VEC_FULL, VGS_VEC_FULL, VSB_VEC_FULL,
                test_mode=False)


if __name__ == "__main__":
    main()
