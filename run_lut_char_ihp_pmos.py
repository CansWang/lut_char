#!/usr/bin/env python3
"""
run_lut_char_ihp_pmos.py — Non-uniform VDS LUT characterization for IHP sg13_lv_pmos.

PDK: IHP SG13G2 (ihp-sg13g2), PSP 103.6 model via OSDI.
     Lib:  ~/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib
     Corners: mos_tt / mos_ff / mos_ss / mos_sf / mos_fs

OSDI prerequisite
-----------------
The .spiceinit in the sim/ directory loads the PSP OSDI shared objects.
A symlink sim/.spiceinit → .../ihp-sg13g2/simulation/.spiceinit is created
automatically by this script.

Unit convention
---------------
IHP does NOT set scale=1.0u, so L/W values in the netlist must carry an
explicit 'u' suffix: L=0.13u, W=5u.

PMOS biasing convention
-----------------------
Source (node 0) at 0 V.
  vg 0 g <value>  → V(g) = −<value>  (PMOS gate below source)
  vd 0 d <value>  → V(d) = −<value>  (PMOS drain below source)
  vb 0 b <value>  → V(b) = −<value>  (body; VSB = −V(b) stored in mat)
Sweep magnitudes (|VGS|, |VDS|, |VSB|) are passed as positive values to
'alter'; signs are handled by the netlist topology.

VDS grid (|VDS| magnitudes)
---------------------------
Fine   : 0.000 – 0.295 V @ 5 mV   → 60 pts
Coarse : 0.300 – 1.200 V @ 50 mV  → 19 pts
Total  : 79 pts, no duplicate at 0.300 V

ngspice compose quirk: stop=0.2951 (not 0.295) to reliably include 0.295 V.

Noise data
----------
PSP model writes drain-current noise PSDs directly as operating-point
quantities: 'sid' (thermal, A²/Hz) and 'sfl' (flicker, A²/Hz).
No separate .noise analysis is needed; STH/SFL need NO squaring.

Usage
-----
  python run_lut_char_ihp_pmos.py --test-run          # micro-sweep validation
  python run_lut_char_ihp_pmos.py --corners TT FF     # specific corners
  python run_lut_char_ihp_pmos.py                     # full 5-corner × 3-temp PVT
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
DEVICE    = "sg13_lv_pmos"
MODEL_LIB = ("/home/cwang/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/models"
             "/cornerMOSlv.lib")
SPICEINIT_SRC = Path(
    "/home/cwang/Book-on-gm-ID-design/starter_files_open_source_tools"
    "/ihp-sg13g2/simulation/.spiceinit"
)
W_UM      = 5.0    # characterisation width (bare float, 'u' added in netlist)
NFING     = 1

# Inner MOSFET name inside the subcircuit (used in .save directives)
_INNER    = f"n{DEVICE}"           # "nsg13_lv_pmos"
_SAVE_PFX = f"@n.xm1.{_INNER}"    # "@n.xm1.nsg13_lv_pmos"

# ── Corners ────────────────────────────────────────────────────────────────────
ALL_CORNERS = ["TT", "FF", "SS", "SF", "FS"]
_CORNER_LIB = {
    "TT": "mos_tt", "FF": "mos_ff", "SS": "mos_ss",
    "SF": "mos_sf", "FS": "mos_fs",
}

# ── Full-run sweep grids ───────────────────────────────────────────────────────
ALL_TEMPS    = [-40, 27, 125]
L_VEC_FULL   = [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
                0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                1.00, 2.00, 3.00]   # µm (stored as floats; 'u' added in netlist)
VGS_VEC_FULL = list(np.round(np.arange(0, 1.201, 0.025), 4))  # 49 pts, |VGS|
VSB_VEC_FULL = [0.0, -0.2, -0.4]  # vb source values → V(b) = [0, +0.2, +0.4]

# ── Non-uniform VDS axis (|VDS| magnitudes, 1.2 V device) ─────────────────────
VDS_FINE   = np.round(np.arange(0,   0.300, 0.005), 4)   # 60 pts: 0.000 … 0.295
VDS_COARSE = np.round(np.arange(0.3, 1.250, 0.050), 4)   # 19 pts: 0.300 … 1.200
VDS_ALL    = np.concatenate([VDS_FINE, VDS_COARSE])        # 79 pts, no duplicate

# ── Test-run reduced grid ──────────────────────────────────────────────────────
CORNERS_TEST = ["TT"]
TEMPS_TEST   = [27]
L_VEC_TEST   = [0.13, 1.00]
VGS_VEC_TEST = [0.6, 1.2]
VSB_VEC_TEST = [0.0]

# ── Paths ──────────────────────────────────────────────────────────────────────
SIM_DIR = Path("/home/cwang/lut_char/sim")
OUT_DIR = Path("/home/cwang/lut_char/output")


# ── Helper: L vector string with 'u' suffix (IHP has no scale=1u) ─────────────

def _l_vec_str(l_vals):
    """Format L values with explicit 'u' suffix for ngspice compose."""
    return " ".join(f"{v}u" for v in l_vals)


def _vec_str(vals, fmt=".4f"):
    """Format a plain float vector (VGS, VDS, VSB) for ngspice compose."""
    return " ".join(format(v, fmt) for v in vals)


# ── Netlist generation ─────────────────────────────────────────────────────────

def generate_netlist(corner: str, temp: int,
                     l_vec, vgs_vec, vsb_vec,
                     out_txt: str, netlist_path: str):
    """
    Write the ngspice deck with dual (fine + coarse) |VDS| foreach loops.

    PMOS convention:
      vg 0 g <val>  → V(g) = −val  (gate below source at 0 V)
      vd 0 d <val>  → V(d) = −val  (drain below source at 0 V)
      vb 0 b <val>  → V(b) = −val  (body; vsb_vec carries vb source values)

    L values carry explicit 'u' suffix in compose; W is bare float (set below).
    """
    lib_corner = _CORNER_LIB[corner]
    l_str   = _l_vec_str(l_vec)
    vgs_str = _vec_str(vgs_vec)
    vsb_str = _vec_str(vsb_vec, ".3f")

    content = textwrap.dedent(f"""\
        ** LUT Characterization: {DEVICE}  corner={corner}  T={temp}C
        ** PDK: IHP SG13G2 (PSP 103.6 via OSDI)
        ** PMOS: vg/vd/vb source from node 0; positive alter value → negative V(node)
        **.subckt lut_char
        Hn 0 n vd 1
        vg 0 g DC 0.0 AC 1
        vd 0 d 0.0
        vb 0 b 0
        XM1 d g 0 b {DEVICE} w={W_UM}u l={{lx}} ng={NFING} m=1

        .param lx=0.13u
        .temp {temp}
        .op

        .control
        option numdgt = 3
        set wr_singlescale
        set wr_vecnames

        compose l_vec         values {l_str}
        compose vg_vec        values {vgs_str}
        compose vsb_vec       values {vsb_str}
        * Fine |VDS|:   0.000 to 0.295 V in 5 mV steps (60 points)
        * stop=0.2951 (not 0.295): ngspice compose FP boundary exclusion workaround
        compose vd_fine_vec   start=0    stop=0.2951 step=0.005
        * Coarse |VDS|: 0.300 to 1.200 V in 50 mV steps (19 points)
        * stop=1.2001 (not 1.2): ngspice compose FP boundary exclusion workaround
        compose vd_coarse_vec start=0.3  stop=1.2001 step=0.05

        foreach var1 $&l_vec
          alterparam lx=$var1
          reset
          foreach var2 $&vg_vec
            alter vg $var2
            * === Fine |VDS| sweep (0 to 0.295 V @ 5 mV) ===
            foreach var3 $&vd_fine_vec
              alter vd $var3
              foreach var4 $&vsb_vec
                alter vb $var4
                run
                wrdata {out_txt} all
                destroy all
                set appendwrite
                unset wr_vecnames
              end
            end
            * === Coarse |VDS| sweep (0.3 to 1.2 V @ 50 mV) ===
            foreach var3 $&vd_coarse_vec
              alter vd $var3
              foreach var4 $&vsb_vec
                alter vb $var4
                run
                wrdata {out_txt} all
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

        .save {_SAVE_PFX}[cdd]
        .save {_SAVE_PFX}[cgb]
        .save {_SAVE_PFX}[cgd]
        .save {_SAVE_PFX}[cgdol]
        .save {_SAVE_PFX}[cgg]
        .save {_SAVE_PFX}[cgs]
        .save {_SAVE_PFX}[cgsol]
        .save {_SAVE_PFX}[cjd]
        .save {_SAVE_PFX}[cjs]
        .save {_SAVE_PFX}[css]
        .save {_SAVE_PFX}[gds]
        .save {_SAVE_PFX}[gm]
        .save {_SAVE_PFX}[gmb]
        .save {_SAVE_PFX}[ids]
        .save {_SAVE_PFX}[l]
        .save {_SAVE_PFX}[sfl]
        .save {_SAVE_PFX}[sid]
        .save {_SAVE_PFX}[vth]
        .save @vb[dc]
        .save @vd[dc]
        .save @vg[dc]
        .save g d b n

        **.ends
        .end
    """)

    with open(netlist_path, "w") as fh:
        fh.write(content)
    print(f"  [gen ] {netlist_path}")


# ── ngspice runner ─────────────────────────────────────────────────────────────

def run_ngspice(netlist_path: str, log_path: str) -> bool:
    """Invoke ngspice in batch mode from SIM_DIR (so .spiceinit is found)."""
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
            for ln in result.stderr.strip().splitlines()[-40:]:
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
    Read ngspice wrdata output (.op analysis), reshape into (L, VGS, VDS, VSB)
    tensor, compute composite PSP capacitances, and save as a .mat LUT.

    Column cleanup follows the same logic as the Book-on-gm-ID-design
    techsweep_txt_to_mat.ipynb for IHP sg13_lv_pmos.

    PSP composite cap signs (IHP convention — different from BSIM4/sky130):
      cgg = cgg + cgdol + cgsol
      cgb = -cgb
      cgd =  cgd + cgdol    ← positive sign (PSP)
      cgs =  cgs + cgsol    ← positive sign (PSP)
      cdd = cdd + cjd + cgdol
      css = css + cjs + cgsol

    STH = sid, SFL = sfl  (PSP gives PSDs directly in A²/Hz — no squaring)
    """
    print(f"  [read] {txt_path}")
    df_raw = pd.read_csv(txt_path, sep=r'\s+', engine='python')

    # par_names[1] carries the save prefix (no frequency column for .op)
    par_names = df_raw.columns.to_list()
    fet_name  = par_names[1].split('[')[0]   # "@n.xm1.nsg13_lv_pmos"

    # Drop raw node-voltage columns (g, g.1, b, d, n) — not needed for LUT
    drop_cols = [c for c in df_raw.columns
                 if c in ('g', 'g.1', 'b', 'd', 'n')]
    df = df_raw.drop(columns=drop_cols, errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce')

    df.columns = df.columns.str.replace(fet_name,     '', regex=False)
    df.columns = df.columns.str.replace(fet_name[1:], '', regex=False)
    df.columns = df.columns.str.replace('[dc]',        '', regex=False)
    df.columns = df.columns.str.removeprefix('@')
    df.columns = df.columns.str.removeprefix('[')
    df.columns = df.columns.str.removesuffix(']')

    print(f"  [read] Columns: {list(df.columns)}")

    nL   = len(l_vec)
    nVGS = len(vgs_vec)
    nVDS = len(VDS_ALL)   # 79
    nVSB = len(vsb_vec)
    dims = [nL, nVGS, nVDS, nVSB]

    expected_rows = nL * nVGS * nVDS * nVSB
    actual_rows   = len(df)
    print(f"  [chk ] Rows: expected={expected_rows}, actual={actual_rows}")
    if actual_rows != expected_rows:
        print(f"  [WARN] Row mismatch — tensor reshape may fail!")

    def R(col):
        return np.reshape(df[col].values[:expected_rows], dims)

    # Composite capacitances — IHP PSP 103.6 convention (matches notebook)
    cgg = R('cgg') + R('cgdol') + R('cgsol')
    cgb = -R('cgb')
    cgd =  R('cgd') + R('cgdol')   # PSP sign: +cgd (not -cgd like BSIM4)
    cgs =  R('cgs') + R('cgsol')   # PSP sign: +cgs (not -cgs like BSIM4)
    cdd = R('cdd') + R('cjd') + R('cgdol')
    css = R('css') + R('cjs') + R('cgsol')

    # VSB for mat: magnitude = −(vb source value) since V(b) = −vb_source_val
    vsb_mat = -np.array(vsb_vec)    # [0, +0.2, +0.4] for vsb_vec=[0, -0.2, -0.4]

    dic = {
        "INFO":   f"IHP SG13G2 130nm BiCMOS PSP — {DEVICE}",
        "CORNER": corner.upper(),
        "TEMP":   float(temp + 273.15),
        "VGS":    np.array(vgs_vec),    # |VGS| magnitudes
        "VDS":    VDS_ALL,              # |VDS| magnitudes
        "VSB":    vsb_mat,
        "L":      np.array(l_vec),
        "W":      float(W_UM),
        "NFING":  float(NFING),
        "ID":     R('ids'),
        "VT":     R('vth'),
        "GM":     R('gm'),
        "GMB":    R('gmb'),
        "GDS":    R('gds'),
        "CGG":    cgg,
        "CGB":    cgb,
        "CGD":    cgd,
        "CGS":    cgs,
        "CDD":    cdd,
        "CSS":    css,
        "STH":    R('sid'),   # PSP gives PSD directly in A²/Hz — no squaring
        "SFL":    R('sfl'),   # PSP gives PSD directly in A²/Hz — no squaring
    }

    sign    = 'p' if temp >= 0 else 'm'
    mat_key = f"sg13_lv_pmos_{corner}_T{sign}{abs(temp)}"
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
    print(f"  Fine pts   (|VDS| <  0.3 V) : {n_fine}   (expected 60)")
    print(f"  Coarse pts (|VDS| >= 0.3 V) : {n_coarse}  (expected 19)")
    print(f"  Total VDS points             : {n_total} (expected 79)")
    print(f"  Last fine point              : {last_fine:.4f} V  (expected 0.2950)")
    print(f"  First coarse point           : {first_coarse:.4f} V  (expected 0.3000)")
    print(f"  Occurrences of 0.3000 V      : {boundary_pts}  (expected 1 — no duplicate)")
    tc2 = (n_fine == 60 and n_coarse == 19 and boundary_pts == 1)
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
    print(f"\n[TC4] Physics Check — |ID| continuity across |VDS|=0.3 V boundary")
    l_arr   = np.array(dic["L"])
    vgs_arr = np.array(dic["VGS"])
    i_L   = int(np.argmin(np.abs(l_arr   - 1.0)))
    i_VGS = int(np.argmin(np.abs(vgs_arr - 1.2)))
    i_VSB = 0

    id_curve = np.abs(dic["ID"][i_L, i_VGS, :, i_VSB])
    i_295 = int(np.argmin(np.abs(vds_axis - 0.295)))
    i_300 = int(np.argmin(np.abs(vds_axis - 0.300)))
    i_350 = int(np.argmin(np.abs(vds_axis - 0.350)))

    id_295 = float(id_curve[i_295])
    id_300 = float(id_curve[i_300])
    id_350 = float(id_curve[i_350])
    jump   = abs(id_300 - id_295) / (abs(id_295) + 1e-30)

    print(f"  L={l_arr[i_L]:.2f}µm, |VGS|={vgs_arr[i_VGS]:.2f}V, VSB=0V")
    print(f"  |ID| @ |VDS|=0.295 V = {id_295: .6e} A  (last fine point)")
    print(f"  |ID| @ |VDS|=0.300 V = {id_300: .6e} A  (first coarse point)")
    print(f"  |ID| @ |VDS|=0.350 V = {id_350: .6e} A")
    print(f"  Relative step at boundary : {jump:.4%}")

    i_start = max(0, i_295 - 2)
    i_end   = min(len(vds_axis), i_300 + 4)
    print(f"  |ID| profile around 0.3 V boundary:")
    for k in range(i_start, i_end):
        marker = "  ← first coarse pt" if k == i_300 else \
                 "  ← last fine pt"    if k == i_295 else ""
        print(f"    |VDS|={vds_axis[k]:.4f}V  |ID|={id_curve[k]:.6e} A{marker}")

    tc4 = jump < 0.05
    print(f"  Result : {'PASS' if tc4 else 'WARN — jump > 5%, check simulation'}")

    overall = all([tc2, tc3, tc4])
    print(f"\n{'─'*68}")
    print(f"  Overall pipeline status: {'ALL PASS ✓' if overall else 'SEE WARNINGS ABOVE'}")
    print(f"{sep}\n")


# ── Main orchestrator ──────────────────────────────────────────────────────────

def setup_spiceinit():
    """Create sim/.spiceinit symlink so ngspice finds the OSDI loader."""
    link = SIM_DIR / ".spiceinit"
    if link.exists() or link.is_symlink():
        return
    if not SPICEINIT_SRC.exists():
        print(f"  [WARN] .spiceinit source not found: {SPICEINIT_SRC}")
        return
    link.symlink_to(SPICEINIT_SRC)
    print(f"  [init] Created symlink: {link} → {SPICEINIT_SRC}")


def run_pvt(corners, temps, l_vec, vgs_vec, vsb_vec, test_mode=False):
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_spiceinit()

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
                  f" total OP simulations")

            out_txt      = str(SIM_DIR / f"techsweep_pmos_{tag}.txt")
            netlist_path = str(SIM_DIR / f"techsweep_pmos_{tag}.spice")
            log_path     = str(SIM_DIR / f"techsweep_pmos_{tag}.log")
            mat_path     = str(OUT_DIR / f"sg13_lv_pmos_{tag}.mat")

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
        description="IHP sg13_lv_pmos LUT characterization — non-uniform |VDS| grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_lut_char_ihp_pmos.py --test-run
              python run_lut_char_ihp_pmos.py --corners TT SS --temps 27 125
              python run_lut_char_ihp_pmos.py                  # full 5×3 PVT
        """)
    )
    ap.add_argument(
        "--test-run", action="store_true",
        help="Micro-sweep: TT/27°C, L=[0.13,1.0]µm, VGS=[0.6,1.2]V, VSB=[0], 79 VDS"
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

    print(f"PDK: IHP SG13G2  Device: {DEVICE}  W={W_UM}µm")
    print(f"|VDS|: {len(VDS_FINE)} fine pts (0–0.295V @5mV) + "
          f"{len(VDS_COARSE)} coarse pts (0.3–1.2V @50mV) = {len(VDS_ALL)} total")

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
