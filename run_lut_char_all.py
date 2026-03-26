#!/usr/bin/env python3
"""
run_lut_char_all.py — Unified non-uniform-VDS/VGS LUT characterization for all PDKs.

Covers 12 devices across three open-source PDKs:
  sky130  : nfet_01v8, pfet_01v8, nfet_01v8_lvt, pfet_01v8_lvt
  IHP     : sg13_lv_nmos, sg13_lv_pmos, sg13_hv_nmos, sg13_hv_pmos
  GF180   : nfet_03v3, pfet_03v3, nfet_06v0, pfet_06v0

VGS grid (non-uniform) — better resolution in weak/moderate inversion
--------------------------------------------------------------------
Fine   : 0.000 → fine_max  @ 10 mV
Coarse : coarse_start → VGS_max @ 25–100 mV (PDK-dependent)
  1.2 V: 0–0.70V @10mV (71 pts) + 0.725–1.20V @25mV  (20 pts) =  91 total
  1.8 V: 0–0.90V @10mV (91 pts) + 0.925–1.80V @25mV  (35 pts) = 126 total
  3.3 V: 0–1.50V @10mV (151pts) + 1.550–3.30V @50mV  (36 pts) = 187 total
  5.0 V: 0–2.00V @10mV (201pts) + 2.100–5.00V @100mV (30 pts) = 231 total

VDS grid (non-uniform) — high resolution near triode/saturation boundary
--------------------------------------------------------------------
Fine   : 0.000 – 0.295 V  @ 5 mV   → 60 pts
Coarse : 0.300 – VDS_max  @ 50 or 100 mV (PDK-dependent)
  1.2 V @  50 mV →  19 coarse pts →  79 total
  1.8 V @  50 mV →  31 coarse pts →  91 total
  3.3 V @ 100 mV →  31 coarse pts →  91 total
  5.0 V @ 100 mV →  48 coarse pts → 108 total
ngspice compose FP workaround: stop = exact_max + epsilon

Models
------
sky130  : BSIM4, noise analysis (.noise + wrdata noise1.all)
IHP     : PSP 103.6 via OSDI, op analysis (.op + wrdata all)
GF180   : BSIM4 (non-DSS via fets_mm), noise analysis (.noise + wrdata noise1.all)

Capacitance sign conventions
-----------------------------
BSIM4 (sky130/GF180): CGD = -cgd + cgdo, CGS = -cgs + cgso  (BSIM4 sign)
PSP   (IHP)         : CGD = +cgd + cgdol, CGS = +cgs + cgsol (PSP sign)

Parallel execution
------------------
Uses concurrent.futures.ProcessPoolExecutor to run up to N_CPU ngspice jobs
simultaneously — one job per (corner, temp) combination.

Usage
-----
  python run_lut_char_all.py --list
  python run_lut_char_all.py --device sky130:nfet_01v8 --test-run
  python run_lut_char_all.py --device ihp:sg13_lv_pmos
  python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT FF
  python run_lut_char_all.py --device gf180:nfet_06v0  # TT only
"""

import argparse
import concurrent.futures
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import savemat


# ── Paths ──────────────────────────────────────────────────────────────────────
SIM_DIR   = Path("/home/cwang/lut_char/sim")
OUT_DIR   = Path("/home/cwang/lut_char/output")
STOP_FILE = Path("/home/cwang/lut_char/STOP")

_SPICEINIT_SRC = Path(
    "/home/cwang/Book-on-gm-ID-design/starter_files_open_source_tools"
    "/ihp-sg13g2/simulation/.spiceinit"
)

# Default parallel ngspice workers (all logical CPUs; override with --workers)
N_WORKERS_DEFAULT = max(1, (os.cpu_count() or 1))


# ── VDS non-uniform grid ───────────────────────────────────────────────────────
# Fine grid: same for all devices
_VDS_FINE = np.round(np.arange(0, 0.300, 0.005), 4)   # 60 pts: 0.000 … 0.295

# Per-VDS_max: (coarse_step, python_arange_stop, ngspice_compose_stop)
# python_arange_stop includes the endpoint; ngspice stop has FP epsilon workaround
_VDS_COARSE_CFG = {
    1.2: (0.050, 1.250, "1.2001"),   # 19 pts: 0.300 … 1.200
    1.8: (0.050, 1.850, "1.8"),      # 31 pts: 0.300 … 1.800
    3.3: (0.100, 3.350, "3.3001"),   # 31 pts: 0.300 … 3.300
    5.0: (0.100, 5.100, "5.001"),    # 48 pts: 0.300 … 5.000
}


def build_vds_all(vds_max: float) -> np.ndarray:
    step, py_stop, _ = _VDS_COARSE_CFG[vds_max]
    coarse = np.round(np.arange(0.3, py_stop, step), 4)
    return np.concatenate([_VDS_FINE, coarse])


# ── VGS non-uniform grid ───────────────────────────────────────────────────────
# Fine step is 10 mV for all devices
_VGS_FINE_STEP = 0.010

# Per-VGS_max: (fine_py_stop, fine_ng_stop,
#               coarse_start, coarse_step, coarse_py_stop, coarse_ng_stop)
# fine grid: 0 … fine_py_stop-step  (arange stop is exclusive)
# coarse grid: coarse_start … VGS_max
_VGS_COARSE_CFG = {
    # 1.2V: fine 0–0.70V @10mV (71pts) + coarse 0.725–1.20V @25mV (20pts) = 91
    1.2: (0.71,  "0.7001", 0.725, 0.025, 1.225,  "1.2001"),
    # 1.8V: fine 0–0.90V @10mV (91pts) + coarse 0.925–1.80V @25mV (35pts) = 126
    1.8: (0.91,  "0.9001", 0.925, 0.025, 1.825,  "1.8001"),
    # 3.3V: fine 0–1.50V @10mV (151pts) + coarse 1.55–3.30V @50mV (36pts) = 187
    3.3: (1.51,  "1.501",  1.55,  0.050, 3.350,  "3.3001"),
    # 5.0V: fine 0–2.00V @10mV (201pts) + coarse 2.10–5.00V @100mV (30pts) = 231
    5.0: (2.01,  "2.001",  2.10,  0.100, 5.100,  "5.001"),
}


def build_vgs_all(vgs_max: float) -> np.ndarray:
    fine_py, _, cs_start, cs_step, cs_py, _ = _VGS_COARSE_CFG[vgs_max]
    fine   = np.round(np.arange(0.0, fine_py,  _VGS_FINE_STEP), 4)
    coarse = np.round(np.arange(cs_start, cs_py, cs_step), 4)
    return np.concatenate([fine, coarse])


def _n_vgs_fine(vgs_max: float) -> int:
    fine_py, *_ = _VGS_COARSE_CFG[vgs_max]
    return len(np.round(np.arange(0.0, fine_py, _VGS_FINE_STEP), 4))


def _n_vgs_coarse(vgs_max: float) -> int:
    _, _, cs_start, cs_step, cs_py, _ = _VGS_COARSE_CFG[vgs_max]
    return len(np.round(np.arange(cs_start, cs_py, cs_step), 4))


def _vgs_fine_max(vgs_max: float) -> float:
    """Last fine VGS point (= fine_py_stop - step)."""
    fine_py, *_ = _VGS_COARSE_CFG[vgs_max]
    return round(fine_py - _VGS_FINE_STEP, 4)


# ── PVT ───────────────────────────────────────────────────────────────────────
ALL_TEMPS = [-40, 27, 125]


# ── Device configuration ───────────────────────────────────────────────────────
@dataclass
class DevCfg:
    key: str               # registry key, e.g. "sky130:nfet_01v8"
    device: str            # ngspice subcircuit/model name
    pdk: str               # "sky130" | "ihp" | "gf180"
    fet_type: str          # "nfet" | "pfet"

    # Model loading
    model_lib: str         # path to .lib file
    lib_corner_map: dict   # {"TT": "<lib_section>", ...}
    model_include: Optional[str] = None   # extra .include (GF180 only)
    spiceinit_src: Optional[Path] = None  # IHP only

    # Geometry
    w_um: float = 5.0
    nfing: int = 1
    has_explicit_u: bool = False  # False=sky130 (scale=1u), True=ihp/gf180
    contact_dist: Optional[str] = None   # "0.29" sky130, "0.18u" gf180, None ihp

    # Sweep grids
    l_vec: list = field(default_factory=list)
    vgs_max: float = 1.8
    vds_max: float = 1.8
    vsb_vec: list = field(default_factory=lambda: [0.0, -0.2, -0.4])

    # Netlist
    analysis: str = "noise"         # "noise" | "op"
    save_pfx: str = ""               # e.g. "@m.xm1.msky130_fd_pr__nfet_01v8"
    save_nodes: str = "g d b n"
    ihp_nmos_src_first: bool = False  # True → XM1 0 g d b (IHP NMOS)

    # Model-dependent column names
    caps_model: str = "bsim4"   # "bsim4" | "psp"
    gmb_col: str = "gmbs"        # "gmbs" bsim4, "gmb" psp
    id_col: str = "id"           # "id" bsim4, "ids" psp


# ── Model library paths ────────────────────────────────────────────────────────
_SKY130_LIB    = "/home/cwang/apragma/models/sky130/models/sky130.lib.spice"
_IHP_LV_LIB    = ("/home/cwang/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/"
                  "models/cornerMOSlv.lib")
_IHP_HV_LIB    = ("/home/cwang/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/"
                  "models/cornerMOShv.lib")
_GF180_INCLUDE = "/home/cwang/apragma/models/gf180mcu/ngspice/design.ngspice"
_GF180_LIB     = "/home/cwang/apragma/models/gf180mcu/ngspice/sm141064.ngspice"

# ── Corner maps ────────────────────────────────────────────────────────────────
_SKY130_CM  = {"TT": "tt",     "FF": "ff",     "SS": "ss",     "SF": "sf",     "FS": "fs"}
_IHP_CM     = {"TT": "mos_tt", "FF": "mos_ff", "SS": "mos_ss", "SF": "mos_sf", "FS": "mos_fs"}
# GF180: unified corner sections in sm141064.ngspice (.LIB typical / ff / ss / sf / fs).
# Each unified section calls nfet_03v3_t + pfet_03v3_t + nfet_06v0_t + pfet_06v0_t +
# noise_corner + fets_mm in one shot — exactly matching the reference techsweep files.
# 6V devices use TT BSIM4 params across all corners (no process split for 6V).
_GF180_CM = {"TT": "typical", "FF": "ff", "SS": "ss", "SF": "sf", "FS": "fs"}

# ── L vectors (µm) ─────────────────────────────────────────────────────────────
_SKY130_L_STD      = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
                       0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                       1.00, 2.00, 3.00]          # 16 pts
_SKY130_L_PFET_LVT = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
                       0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00]  # 14 pts
_IHP_LV_L          = [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
                       0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                       1.00, 2.00, 3.00]          # 18 pts
_IHP_HV_L          = [0.45, 0.50, 0.55,
                       0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00]        # 10 pts
_GF180_3V3_L       = [0.28, 0.29, 0.30,
                       0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                       1.00, 2.00, 3.00]          # 12 pts
_GF180_NFET_5V_L   = [0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00]        # 7 pts
_GF180_PFET_5V_L   = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 2.00, 3.00]  # 8 pts

# ── Device registry ────────────────────────────────────────────────────────────
DEVICES = {
    # ── sky130 ────────────────────────────────────────────────────────────────
    "sky130:nfet_01v8": DevCfg(
        key="sky130:nfet_01v8", pdk="sky130", fet_type="nfet",
        device="sky130_fd_pr__nfet_01v8",
        model_lib=_SKY130_LIB, lib_corner_map=_SKY130_CM,
        contact_dist="0.29",
        l_vec=_SKY130_L_STD, vgs_max=1.8, vds_max=1.8,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.msky130_fd_pr__nfet_01v8",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    "sky130:pfet_01v8": DevCfg(
        key="sky130:pfet_01v8", pdk="sky130", fet_type="pfet",
        device="sky130_fd_pr__pfet_01v8",
        model_lib=_SKY130_LIB, lib_corner_map=_SKY130_CM,
        contact_dist="0.29",
        l_vec=_SKY130_L_STD, vgs_max=1.8, vds_max=1.8,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.msky130_fd_pr__pfet_01v8",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    "sky130:nfet_01v8_lvt": DevCfg(
        key="sky130:nfet_01v8_lvt", pdk="sky130", fet_type="nfet",
        device="sky130_fd_pr__nfet_01v8_lvt",
        model_lib=_SKY130_LIB, lib_corner_map=_SKY130_CM,
        contact_dist="0.29",
        l_vec=_SKY130_L_STD, vgs_max=1.8, vds_max=1.8,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.msky130_fd_pr__nfet_01v8_lvt",
        save_nodes="d g s n",   # lvt subcircuit exposes source pin 's'
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    "sky130:pfet_01v8_lvt": DevCfg(
        key="sky130:pfet_01v8_lvt", pdk="sky130", fet_type="pfet",
        device="sky130_fd_pr__pfet_01v8_lvt",
        model_lib=_SKY130_LIB, lib_corner_map=_SKY130_CM,
        contact_dist="0.29",
        l_vec=_SKY130_L_PFET_LVT, vgs_max=1.8, vds_max=1.8,  # L_min=0.35!
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.msky130_fd_pr__pfet_01v8_lvt",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    # ── IHP LV ────────────────────────────────────────────────────────────────
    "ihp:sg13_lv_nmos": DevCfg(
        key="ihp:sg13_lv_nmos", pdk="ihp", fet_type="nfet",
        device="sg13_lv_nmos",
        model_lib=_IHP_LV_LIB, lib_corner_map=_IHP_CM,
        spiceinit_src=_SPICEINIT_SRC,
        has_explicit_u=True,
        l_vec=_IHP_LV_L, vgs_max=1.2, vds_max=1.2,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="op",
        save_pfx="@n.xm1.nsg13_lv_nmos",
        save_nodes="g d b n",
        ihp_nmos_src_first=True,   # XM1 0 g d b sg13_lv_nmos
        caps_model="psp", gmb_col="gmb", id_col="ids",
    ),
    "ihp:sg13_lv_pmos": DevCfg(
        key="ihp:sg13_lv_pmos", pdk="ihp", fet_type="pfet",
        device="sg13_lv_pmos",
        model_lib=_IHP_LV_LIB, lib_corner_map=_IHP_CM,
        spiceinit_src=_SPICEINIT_SRC,
        has_explicit_u=True,
        l_vec=_IHP_LV_L, vgs_max=1.2, vds_max=1.2,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="op",
        save_pfx="@n.xm1.nsg13_lv_pmos",
        save_nodes="g d b n",
        caps_model="psp", gmb_col="gmb", id_col="ids",
    ),
    # ── IHP HV ────────────────────────────────────────────────────────────────
    "ihp:sg13_hv_nmos": DevCfg(
        key="ihp:sg13_hv_nmos", pdk="ihp", fet_type="nfet",
        device="sg13_hv_nmos",
        model_lib=_IHP_HV_LIB, lib_corner_map=_IHP_CM,
        spiceinit_src=_SPICEINIT_SRC,
        has_explicit_u=True,
        l_vec=_IHP_HV_L, vgs_max=3.3, vds_max=3.3,
        vsb_vec=[0.0, -0.2, -0.4, -0.6],   # 4 VSB points for HV
        analysis="op",
        save_pfx="@n.xm1.nsg13_hv_nmos",
        save_nodes="g d b n",
        ihp_nmos_src_first=True,   # XM1 0 g d b sg13_hv_nmos
        caps_model="psp", gmb_col="gmb", id_col="ids",
    ),
    "ihp:sg13_hv_pmos": DevCfg(
        key="ihp:sg13_hv_pmos", pdk="ihp", fet_type="pfet",
        device="sg13_hv_pmos",
        model_lib=_IHP_HV_LIB, lib_corner_map=_IHP_CM,
        spiceinit_src=_SPICEINIT_SRC,
        has_explicit_u=True,
        l_vec=_IHP_HV_L, vgs_max=3.3, vds_max=3.3,
        vsb_vec=[0.0, -0.2, -0.4, -0.6],
        analysis="op",
        save_pfx="@n.xm1.nsg13_hv_pmos",
        save_nodes="g d b n",
        caps_model="psp", gmb_col="gmb", id_col="ids",
    ),
    # ── GF180 3.3 V ────────────────────────────────────────────────────────────
    # Non-DSS devices from fets_mm (no drain/source series resistors).
    # Model: .lib sm141064 typical|ff|ss|sf|fs — each unified section bundles
    # nfet_03v3_t + pfet_03v3_t + noise_corner + fets_mm automatically.
    "gf180:nfet_03v3": DevCfg(
        key="gf180:nfet_03v3", pdk="gf180", fet_type="nfet",
        device="nfet_03v3",
        model_lib=_GF180_LIB, lib_corner_map=_GF180_CM,
        model_include=_GF180_INCLUDE,
        has_explicit_u=True, contact_dist="0.18u",
        l_vec=_GF180_3V3_L, vgs_max=3.3, vds_max=3.3,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.m0",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    "gf180:pfet_03v3": DevCfg(
        key="gf180:pfet_03v3", pdk="gf180", fet_type="pfet",
        device="pfet_03v3",
        model_lib=_GF180_LIB, lib_corner_map=_GF180_CM,
        model_include=_GF180_INCLUDE,
        has_explicit_u=True, contact_dist="0.18u",
        l_vec=_GF180_3V3_L, vgs_max=3.3, vds_max=3.3,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.m0",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    # ── GF180 5 V ──────────────────────────────────────────────────────────────
    # nfet_05v0 / pfet_05v0: wrappers around the nfet_06v0/pfet_06v0 BSIM4 core
    # that allow slightly shorter min L (0.60µm / 0.50µm).  Inner element is m0
    # so save_pfx "@m.xm1.m0" is identical to the 3V3 devices.
    "gf180:nfet_05v0": DevCfg(
        key="gf180:nfet_05v0", pdk="gf180", fet_type="nfet",
        device="nfet_05v0",
        model_lib=_GF180_LIB, lib_corner_map=_GF180_CM,
        model_include=_GF180_INCLUDE,
        has_explicit_u=True, contact_dist="0.18u",
        l_vec=_GF180_NFET_5V_L, vgs_max=5.0, vds_max=5.0,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.m0",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
    "gf180:pfet_05v0": DevCfg(
        key="gf180:pfet_05v0", pdk="gf180", fet_type="pfet",
        device="pfet_05v0",
        model_lib=_GF180_LIB, lib_corner_map=_GF180_CM,
        model_include=_GF180_INCLUDE,
        has_explicit_u=True, contact_dist="0.18u",
        l_vec=_GF180_PFET_5V_L, vgs_max=5.0, vds_max=5.0,
        vsb_vec=[0.0, -0.2, -0.4],
        analysis="noise",
        save_pfx="@m.xm1.m0",
        save_nodes="g d b n",
        caps_model="bsim4", gmb_col="gmbs", id_col="id",
    ),
}


# ── Small helpers ──────────────────────────────────────────────────────────────

def _vec_str(vals, fmt=".4f"):
    return " ".join(format(v, fmt) for v in vals)


def _l_str(l_vals, explicit_u: bool):
    return " ".join(f"{v}u" if explicit_u else str(v) for v in l_vals)


def _n_vds_coarse(vds_max: float) -> int:
    step, py_stop, _ = _VDS_COARSE_CFG[vds_max]
    return len(np.round(np.arange(0.3, py_stop, step), 4))


def _build_save_lines(cfg: DevCfg) -> str:
    pfx = cfg.save_pfx
    if cfg.caps_model == "bsim4":
        params = ["capbd", "capbs", "cdd", "cgb", "cgd", "cgdo",
                  "cgg", "cgs", "cgso", "css", "gds", "gm",
                  cfg.gmb_col, cfg.id_col, "l", "vth"]
        noise_pfx = pfx[1:]    # strip leading '@' for onoise path
        noise_saves = (
            f".save onoise.{noise_pfx}.{cfg.id_col}\n"
            f".save onoise.{noise_pfx}.1overf"
        )
    else:  # psp
        params = ["cdd", "cgb", "cgd", "cgdol", "cgg", "cgs", "cgsol",
                  "cjd", "cjs", "css", "gds", "gm", cfg.gmb_col,
                  cfg.id_col, "l", "sfl", "sid", "vth"]
        noise_saves = ""

    lines = "\n".join(f".save {pfx}[{p}]" for p in params)
    lines += "\n.save @vb[dc]\n.save @vd[dc]\n.save @vg[dc]"
    if noise_saves:
        lines += "\n" + noise_saves
    lines += f"\n.save {cfg.save_nodes}"
    return lines


# ── Netlist generator ──────────────────────────────────────────────────────────

def _append_vds_block(L: list, out_txt: str, wrdata_arg: str,
                      coarse_step: float, vds_max: float, indent: str = "    "):
    """Append Fine+Coarse VDS foreach lines into list L at given indent level."""
    i = indent
    n_c = _n_vds_coarse(vds_max)
    L += [
        f"{i}* Fine VDS (0–0.295V @5mV, 60 pts)",
        f"{i}foreach var3 $&vd_fine_vec",
        f"{i}  alter vd $var3",
        f"{i}  foreach var4 $&vsb_vec",
        f"{i}    alter vb $var4",
        f"{i}    run",
        f"{i}    wrdata {out_txt} {wrdata_arg}",
        f"{i}    destroy all",
        f"{i}    set appendwrite",
        f"{i}    unset wr_vecnames",
        f"{i}  end",
        f"{i}end",
        f"{i}* Coarse VDS (0.3–{vds_max}V @{coarse_step}V, {n_c} pts)",
        f"{i}foreach var3 $&vd_coarse_vec",
        f"{i}  alter vd $var3",
        f"{i}  foreach var4 $&vsb_vec",
        f"{i}    alter vb $var4",
        f"{i}    run",
        f"{i}    wrdata {out_txt} {wrdata_arg}",
        f"{i}    destroy all",
        f"{i}    set appendwrite",
        f"{i}    unset wr_vecnames",
        f"{i}  end",
        f"{i}end",
    ]


def generate_netlist(cfg: DevCfg, corner: str, temp: int,
                     l_vec, vsb_vec,
                     out_txt: str, netlist_path: str,
                     vgs_override=None):
    """
    Write an ngspice netlist.

    vgs_override : None  → non-uniform VGS grid (fine 10mV + coarse), used for
                           full production runs.
                 : list  → explicit VGS values (compose values …), used for
                           test-run mode with a small number of points.
    """
    lib_corner   = cfg.lib_corner_map[corner]
    l_str        = _l_str(l_vec, cfg.has_explicit_u)
    vsb_str      = _vec_str(vsb_vec, ".3f")
    _, _, vds_ng_stop  = _VDS_COARSE_CFG[cfg.vds_max]
    vds_cs       = _VDS_COARSE_CFG[cfg.vds_max][0]
    save_lines   = _build_save_lines(cfg)
    init_lx      = f"{l_vec[0]}u" if cfg.has_explicit_u else str(l_vec[0])

    # ── NFET vs PFET biasing topology ─────────────────────────────────────────
    is_pmos = (cfg.fet_type == "pfet")
    if is_pmos:
        hn_line = "Hn 0 n vd 1"
        vg_line = "vg 0 g DC 0.0 AC 1"
        vd_line = "vd 0 d 0.0"
        vb_line = "vb 0 b 0"
    else:
        hn_line = "Hn n 0 vd 1"
        vg_line = "vg g 0 DC 0.0 AC 1"
        vd_line = "vd d 0 0.0"
        vb_line = "vb b 0 0"

    # ── Instance line ──────────────────────────────────────────────────────────
    u = "u" if cfg.has_explicit_u else ""
    if cfg.pdk == "ihp":
        if cfg.ihp_nmos_src_first:   # IHP NMOS: subcircuit port order is s g d b
            inst = (f"XM1 0 g d b {cfg.device} w={cfg.w_um}u l={{lx}}"
                    f" ng={cfg.nfing} m=1")
        else:                         # IHP PMOS: d g 0 b
            inst = (f"XM1 d g 0 b {cfg.device} w={cfg.w_um}u l={{lx}}"
                    f" ng={cfg.nfing} m=1")
    else:  # sky130 or gf180
        cd = cfg.contact_dist
        w_val = f"{cfg.w_um}{u}"
        # sky130 subcircuits accept mult=1; GF180 non-DSS subcircuits do not
        trail = "mult=1 m=1" if cfg.pdk == "sky130" else "m=1"
        inst = (
            f"XM1 d g 0 b {cfg.device} L={{lx}} W={w_val} nf={cfg.nfing}"
            f" ad='int((nf+1)/2) * W/nf * {cd}'"
            f" as='int((nf+2)/2) * W/nf * {cd}'"
            f" pd='2*int((nf+1)/2) * (W/nf + {cd})'"
            f" ps='2*int((nf+2)/2) * (W/nf + {cd})'"
            f" nrd='{cd} / W' nrs='{cd} / W' sa=0 sb=0 sd=0 {trail}"
        )

    # ── Analysis directive ─────────────────────────────────────────────────────
    if cfg.analysis == "noise":
        analysis_line = ".noise v(n) vg lin 1 1 1 1"
        wrdata_arg    = "noise1.all"
    else:
        analysis_line = ".op"
        wrdata_arg    = "all"

    # ── Model loading ──────────────────────────────────────────────────────────
    if cfg.pdk == "gf180":
        # The unified corner section (typical/ff/ss/sf/fs) already includes
        # nfet_03v3_t + pfet_03v3_t + noise_corner + fets_mm in one call.
        # This matches the reference techsweep files exactly.
        lib_block = (f".include {cfg.model_include}\n"
                     f".lib {cfg.model_lib} {lib_corner}")
    elif cfg.model_include:
        lib_block = (f".include {cfg.model_include}\n"
                     f".lib {cfg.model_lib} {lib_corner}")
    else:
        lib_block = f".lib {cfg.model_lib} {lib_corner}"

    # ── Build control block lines ──────────────────────────────────────────────
    ctrl = []
    ctrl += [
        ".control",
        "option numdgt = 3",
        "set wr_singlescale",
        "set wr_vecnames",
        "",
        f"compose l_vec   values {l_str}",
        f"compose vsb_vec values {vsb_str}",
        f"* Fine VDS (0–0.295V @5mV, 60 pts)",
        f"compose vd_fine_vec   start=0   stop=0.2951       step=0.005",
        f"* Coarse VDS (0.3–{cfg.vds_max}V @{vds_cs}V, {_n_vds_coarse(cfg.vds_max)} pts)",
        f"compose vd_coarse_vec start=0.3 stop={vds_ng_stop} step={vds_cs}",
    ]

    if vgs_override is None:
        # Full non-uniform VGS mode
        fine_py, fine_ng, cs_start, cs_step, cs_py, cs_ng = _VGS_COARSE_CFG[cfg.vgs_max]
        n_vf = _n_vgs_fine(cfg.vgs_max)
        n_vc = _n_vgs_coarse(cfg.vgs_max)
        ctrl += [
            f"* Fine VGS (0–{_vgs_fine_max(cfg.vgs_max):.3f}V @{_VGS_FINE_STEP*1000:.0f}mV, {n_vf} pts)",
            f"compose vg_fine_vec   start=0          stop={fine_ng}  step={_VGS_FINE_STEP}",
            f"* Coarse VGS ({cs_start}–{cfg.vgs_max}V @{cs_step}V, {n_vc} pts)",
            f"compose vg_coarse_vec start={cs_start}  stop={cs_ng}   step={cs_step}",
        ]
    else:
        # Test-mode: explicit VGS list
        ctrl.append(f"compose vg_vec values {_vec_str(vgs_override)}")

    ctrl += ["", "foreach var1 $&l_vec", "  alterparam lx=$var1", "  reset"]

    if vgs_override is None:
        ctrl.append(f"  * === Fine VGS sweep ({n_vf} pts) ===")
        ctrl.append("  foreach var2 $&vg_fine_vec")
        ctrl.append("    alter vg $var2")
        _append_vds_block(ctrl, out_txt, wrdata_arg, vds_cs, cfg.vds_max, indent="    ")
        ctrl.append("  end")
        ctrl.append(f"  * === Coarse VGS sweep ({n_vc} pts) ===")
        ctrl.append("  foreach var2 $&vg_coarse_vec")
        ctrl.append("    alter vg $var2")
        _append_vds_block(ctrl, out_txt, wrdata_arg, vds_cs, cfg.vds_max, indent="    ")
        ctrl.append("  end")
    else:
        ctrl.append("  foreach var2 $&vg_vec")
        ctrl.append("    alter vg $var2")
        _append_vds_block(ctrl, out_txt, wrdata_arg, vds_cs, cfg.vds_max, indent="    ")
        ctrl.append("  end")

    ctrl += ["end", "unset appendwrite", ".endc"]

    # ── Assemble full netlist ──────────────────────────────────────────────────
    header = [
        f"** LUT Characterization: {cfg.device}  corner={corner}  T={temp}C",
        f"** PDK: {cfg.pdk}  Analysis: {cfg.analysis}",
        "**.subckt lut_char",
        hn_line, vg_line, vd_line, vb_line,
        inst, "",
        f".param lx={init_lx}",
        f".temp {temp}",
        analysis_line, "",
    ]
    footer = [
        "", lib_block, "",
        save_lines,
        "**.ends",
        ".end",
        "",
    ]

    content = "\n".join(header + ctrl + footer)

    with open(netlist_path, "w") as fh:
        fh.write(content)
    print(f"  [gen ] {netlist_path}")


# ── ngspice runner ─────────────────────────────────────────────────────────────

def run_ngspice(netlist_path: str, log_path: str, label: str = "") -> bool:
    cmd = ["ngspice", "-b", netlist_path]
    tag = f"[{label}] " if label else ""
    print(f"  {tag}[sim ] {' '.join(cmd)}")
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
            print(f"  {tag}[ERR ] ngspice exited {result.returncode} — see {log_path}")
            for ln in result.stderr.strip().splitlines()[-40:]:
                print(f"           {ln}")
            return False
        print(f"  {tag}[sim ] Done → {log_path}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  {tag}[ERR ] ngspice timed out")
        return False


# ── Parser and .mat writer ─────────────────────────────────────────────────────

def parse_and_save(cfg: DevCfg, txt_path: str, corner: str, temp: int,
                   l_vec, vgs_vec, vsb_vec, mat_path: str):
    print(f"  [read] {txt_path}")
    df_raw = pd.read_csv(txt_path, sep=r'\s+', engine='python')

    pfx = cfg.save_pfx

    # Drop non-data columns
    if cfg.analysis == "noise":
        df = df_raw.drop(
            columns=[c for c in df_raw.columns if c.startswith("frequency")],
            errors='ignore')
    else:  # op: drop raw node voltage columns
        drop_v = [c for c in df_raw.columns if c in ('g', 'g.1', 'b', 'd', 'n')]
        df = df_raw.drop(columns=drop_v, errors='ignore')

    df = df.apply(pd.to_numeric, errors='coerce')

    # Column name cleanup
    df.columns = df.columns.str.replace(pfx,       '', regex=False)
    df.columns = df.columns.str.replace(pfx[1:],   '', regex=False)
    df.columns = df.columns.str.replace('[dc]',     '', regex=False)
    df.columns = df.columns.str.replace('onoise..', 'n', regex=False)
    df.columns = df.columns.str.removeprefix('@')
    df.columns = df.columns.str.removeprefix('[')
    df.columns = df.columns.str.removesuffix(']')

    print(f"  [read] Columns: {list(df.columns)}")

    vds_all = build_vds_all(cfg.vds_max)
    nL   = len(l_vec)
    nVGS = len(vgs_vec)
    nVDS = len(vds_all)
    nVSB = len(vsb_vec)
    dims = [nL, nVGS, nVDS, nVSB]

    expected = nL * nVGS * nVDS * nVSB
    actual   = len(df)
    print(f"  [chk ] Rows: expected={expected}, actual={actual}")
    if actual != expected:
        print(f"  [WARN] Row mismatch — tensor reshape may fail!")

    def R(col):
        return np.reshape(df[col].values[:expected], dims)

    # Composite capacitances
    if cfg.caps_model == "bsim4":
        cgg = R('cgg') + R('cgdo') + R('cgso')
        cgb = -R('cgb')
        cgd = -R('cgd') + R('cgdo')
        cgs = -R('cgs') + R('cgso')
        cdd = R('cdd') + R('capbd') + R('cgdo')
        css = R('css') + R('capbs') + R('cgso')
        sth = R('nid')**2
        sfl = R('n1overf')**2
    else:  # psp
        cgg = R('cgg') + R('cgdol') + R('cgsol')
        cgb = -R('cgb')
        cgd =  R('cgd') + R('cgdol')
        cgs =  R('cgs') + R('cgsol')
        cdd = R('cdd') + R('cjd') + R('cgdol')
        css = R('css') + R('cjs') + R('cgsol')
        sth = R('sid')
        sfl = R('sfl')

    vsb_mat = np.abs(np.array(vsb_vec))

    dic = {
        "INFO":   f"{cfg.pdk.upper()} PDK — {cfg.device}",
        "CORNER": corner.upper(),
        "TEMP":   float(temp + 273.15),
        "VGS":    np.array(vgs_vec),
        "VDS":    vds_all,
        "VSB":    vsb_mat,
        "L":      np.array(l_vec),
        "W":      float(cfg.w_um),
        "NFING":  float(cfg.nfing),
        "ID":     R(cfg.id_col),
        "VT":     R('vth'),
        "GM":     R('gm'),
        "GMB":    R(cfg.gmb_col),
        "GDS":    R('gds'),
        "CGG":    cgg,
        "CGB":    cgb,
        "CGD":    cgd,
        "CGS":    cgs,
        "CDD":    cdd,
        "CSS":    css,
        "STH":    sth,
        "SFL":    sfl,
    }

    dev_key  = cfg.device.replace('-', '_').replace(' ', '_')
    sign     = 'p' if temp >= 0 else 'm'
    mat_key  = f"{dev_key}_{corner}_T{sign}{abs(temp)}"
    savemat(mat_path, {mat_key: dic})
    print(f"  [save] {mat_path}  (key='{mat_key}')")
    return dic, dims


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_test_run(dic, dims, cfg: DevCfg):
    nL, nVGS, nVDS, nVSB = dims
    vds_all  = dic["VDS"]
    vgs_arr  = dic["VGS"]
    n_vds_c  = _n_vds_coarse(cfg.vds_max)
    sep = "=" * 68

    print(f"\n{sep}")
    print("  TEST CASE VALIDATION RESULTS")
    print(sep)

    print("\n[TC1] Netlist Verification")
    print("  Result : PASS — ngspice completed without syntax errors")

    # TC2 — VDS stitching
    fine_mask   = vds_all < 0.3 - 1e-9
    coarse_mask = vds_all >= 0.3 - 1e-9
    n_fine      = int(np.sum(fine_mask))
    n_c         = int(np.sum(coarse_mask))
    n_total     = len(vds_all)
    n_dup       = int(np.sum(np.abs(vds_all - 0.3) < 1e-9))
    last_fine   = float(vds_all[fine_mask][-1])
    first_c     = float(vds_all[coarse_mask][0])

    print(f"\n[TC2] VDS Stitching")
    print(f"  Fine pts   (< 0.3 V) : {n_fine}   (expected 60)")
    print(f"  Coarse pts (>= 0.3 V): {n_c}  (expected {n_vds_c})")
    print(f"  Total VDS points     : {n_total} (expected {60+n_vds_c})")
    print(f"  Last fine point      : {last_fine:.4f} V  (expected 0.2950)")
    print(f"  First coarse point   : {first_c:.4f} V  (expected 0.3000)")
    print(f"  Occurrences of 0.300 V: {n_dup}  (expected 1)")
    tc2 = (n_fine == 60 and n_c == n_vds_c and n_dup == 1)
    print(f"  Result : {'PASS' if tc2 else 'FAIL'}")

    # TC3 — tensor shape
    expected_shape = tuple(dims)
    actual_shape   = dic['ID'].shape
    print(f"\n[TC3] Tensor Shape")
    print(f"  Expected : {expected_shape}")
    print(f"  Actual   : {actual_shape}")
    tc3 = (actual_shape == expected_shape)
    print(f"  Result : {'PASS' if tc3 else 'FAIL'}")

    # TC4 — ID continuity at VDS=0.3 V boundary
    l_arr   = np.array(dic["L"])
    i_L  = int(np.argmin(np.abs(l_arr  - 1.0))) if 1.0 in l_arr else 0
    i_VGS = int(np.argmin(np.abs(vgs_arr - cfg.vgs_max * 0.8)))
    i_VSB = 0

    id_curve = np.abs(dic["ID"][i_L, i_VGS, :, i_VSB])
    i_295 = int(np.argmin(np.abs(vds_all - 0.295)))
    i_300 = int(np.argmin(np.abs(vds_all - 0.300)))
    i_350 = int(np.argmin(np.abs(vds_all - 0.350)))

    id_295 = float(id_curve[i_295])
    id_300 = float(id_curve[i_300])
    id_350 = float(id_curve[i_350])
    jump   = abs(id_300 - id_295) / (abs(id_295) + 1e-30)

    print(f"\n[TC4] |ID| Continuity at |VDS|=0.3 V Boundary")
    print(f"  L={l_arr[i_L]:.2f}µm, |VGS|={vgs_arr[i_VGS]:.3f}V, VSB=0V")
    print(f"  |ID| @ 0.295 V = {id_295: .6e} A")
    print(f"  |ID| @ 0.300 V = {id_300: .6e} A")
    print(f"  |ID| @ 0.350 V = {id_350: .6e} A")
    print(f"  Relative step  = {jump:.4%}")
    tc4 = jump < 0.05
    print(f"  Result : {'PASS' if tc4 else 'WARN — jump > 5%'}")

    overall = all([tc2, tc3, tc4])
    print(f"\n{'─'*68}")
    print(f"  Overall: {'ALL PASS ✓' if overall else 'SEE WARNINGS ABOVE'}")
    print(f"{sep}\n")


# ── IHP .spiceinit setup ───────────────────────────────────────────────────────

def setup_spiceinit(cfg: DevCfg):
    """Create SIM_DIR/.spiceinit symlink for IHP OSDI model loading."""
    if cfg.spiceinit_src is None:
        return
    link = SIM_DIR / ".spiceinit"
    if link.exists() or link.is_symlink():
        return
    if not cfg.spiceinit_src.exists():
        print(f"  [WARN] .spiceinit source not found: {cfg.spiceinit_src}")
        return
    link.symlink_to(cfg.spiceinit_src)
    print(f"  [init] Symlink: {link} → {cfg.spiceinit_src}")


# ── Single-job worker (top-level for ProcessPoolExecutor pickling) ─────────────

def _run_one_pvt(args):
    """
    Run one (corner, temp) PVT job: generate netlist → ngspice → parse → save.
    Returns (corner, temp, mat_path) on success, None on failure.
    """
    cfg, corner, temp, l_vec, vsb_vec, vgs_override, test_mode = args

    SIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.pdk == "ihp":
        setup_spiceinit(cfg)

    sign     = 'p' if temp >= 0 else 'm'
    tag      = f"{corner}_T{sign}{abs(temp)}"
    dev_safe = cfg.device.replace(':', '_')
    label    = f"{cfg.device}/{corner}/T{sign}{abs(temp)}"

    # Append L-range suffix when only a subset of L values is being simulated,
    # so partial .mat files don't overwrite a full run on the same machine.
    is_partial_l = (len(l_vec) < len(cfg.l_vec))
    if is_partial_l:
        l_start_nm = int(round(l_vec[0]  * 1e3))
        l_end_nm   = int(round(l_vec[-1] * 1e3))
        l_suffix   = f"_L{l_start_nm}to{l_end_nm}nm"
    else:
        l_suffix = ""

    out_txt      = str(SIM_DIR / f"techsweep_{dev_safe}_{tag}{l_suffix}.txt")
    netlist_path = str(SIM_DIR / f"techsweep_{dev_safe}_{tag}{l_suffix}.spice")
    log_path     = str(SIM_DIR / f"techsweep_{dev_safe}_{tag}{l_suffix}.log")
    mat_path     = str(OUT_DIR / f"{dev_safe}_{tag}{l_suffix}.mat")

    # VGS vector for parse_and_save tensor reshape
    parse_vgs = list(vgs_override) if vgs_override is not None else list(build_vgs_all(cfg.vgs_max))
    vds_all   = build_vds_all(cfg.vds_max)

    n_sim = len(l_vec) * len(parse_vgs) * len(vds_all) * len(vsb_vec)
    print(f"\n{'='*68}")
    print(f"  [{label}]  {'[TEST] ' if test_mode else ''}PDK={cfg.pdk}")
    print(f"  Grid: {len(l_vec)}L × {len(parse_vgs)}VGS × "
          f"{len(vds_all)}VDS × {len(vsb_vec)}VSB → {n_sim:,} simulations")

    if os.path.exists(out_txt):
        os.remove(out_txt)

    generate_netlist(cfg, corner, temp, l_vec, vsb_vec,
                     out_txt, netlist_path, vgs_override=vgs_override)

    ok = run_ngspice(netlist_path, log_path, label=label)
    if not ok:
        print(f"  [{label}] [SKIP] ngspice failed — see {log_path}")
        return None

    if not os.path.exists(out_txt) or os.path.getsize(out_txt) == 0:
        print(f"  [{label}] [ERR ] Output file missing/empty: {out_txt}")
        return None

    dic, dims = parse_and_save(cfg, out_txt, corner, temp,
                               l_vec, parse_vgs, vsb_vec, mat_path)
    if test_mode:
        validate_test_run(dic, dims, cfg)

    return (corner, temp, mat_path)


# ── Test-run grid selection ────────────────────────────────────────────────────

def _test_grids(cfg: DevCfg):
    """Return (l_vec, vgs_override, vsb_vec) for the micro-sweep test run."""
    # L: first L and 1µm (or last if 1µm not in list)
    l0  = cfg.l_vec[0]
    l1  = 1.0 if 1.0 in cfg.l_vec else cfg.l_vec[-1]
    l_t = [l0] if l0 == l1 else [l0, l1]

    # VGS: one point from fine region (~60% of fine_max) and one from coarse
    vgs_all = build_vgs_all(cfg.vgs_max)
    n_fine  = _n_vgs_fine(cfg.vgs_max)
    i_fine  = max(0, int(n_fine * 0.6) - 1)
    i_coarse = n_fine + max(0, _n_vgs_coarse(cfg.vgs_max) // 2)
    vgs_t = sorted({float(vgs_all[i_fine]), float(vgs_all[i_coarse])})

    # VSB: zero only
    vsb_t = [cfg.vsb_vec[0]]

    return l_t, vgs_t, vsb_t


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_pvt(cfg: DevCfg, corners, temps, l_vec=None, test_mode=False, max_workers=None):
    """
    Orchestrate PVT sweep, running (corner, temp) jobs in parallel.

    l_vec     : L values to simulate (default: cfg.l_vec).  Pass a slice to
                distribute L-range work across machines; use merge_mats.py to
                combine the resulting partial .mat files afterward.
    test_mode : run a micro-sweep (2L × 2VGS × 1VSB, TT/27°C) for validation.
    """
    if l_vec is None:
        l_vec = list(cfg.l_vec)

    SIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.pdk == "ihp":
        setup_spiceinit(cfg)

    if test_mode:
        l_t, vgs_t, vsb_t = _test_grids(cfg)
        jobs = [(cfg, "TT", 27, l_t, vsb_t, vgs_t, True)]
    else:
        jobs = [
            (cfg, corner, temp, l_vec, cfg.vsb_vec, None, False)
            for corner in corners
            for temp in temps
        ]

    n_jobs    = len(jobs)
    cap       = max_workers if max_workers is not None else N_WORKERS_DEFAULT
    n_workers = min(n_jobs, cap)

    vgs_vec  = build_vgs_all(cfg.vgs_max)
    vds_all  = build_vds_all(cfg.vds_max)

    print(f"\n{'='*68}")
    print(f"  Device : {cfg.device}  PDK : {cfg.pdk}")
    if not test_mode:
        l_info = (f"{l_vec[0]}–{l_vec[-1]}µm ({len(l_vec)} pts"
                  + (f", partial {len(l_vec)}/{len(cfg.l_vec)})" if len(l_vec) < len(cfg.l_vec) else ")"))
        print(f"  L      : {l_info}")
        print(f"  VGS    : {_n_vgs_fine(cfg.vgs_max)} fine pts "
              f"(0–{_vgs_fine_max(cfg.vgs_max):.3f}V @10mV) + "
              f"{_n_vgs_coarse(cfg.vgs_max)} coarse pts = "
              f"{len(vgs_vec)} total")
        print(f"  VDS    : 60 fine pts (0–0.295V @5mV) + "
              f"{_n_vds_coarse(cfg.vds_max)} coarse pts = "
              f"{len(vds_all)} total")
        print(f"  Jobs   : {n_jobs}  Workers : {n_workers} / {os.cpu_count()} CPUs")
    print(f"{'='*68}")

    if n_workers <= 1:
        for job in jobs:
            if STOP_FILE.exists():
                print("\n  [STOP] STOP file detected — aborting remaining jobs")
                break
            _run_one_pvt(job)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_job = {pool.submit(_run_one_pvt, job): job for job in jobs}
            stopped = False
            for fut in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[fut]
                _, corner, temp, *_ = job
                sign = 'p' if temp >= 0 else 'm'
                try:
                    result = fut.result()
                    if result is not None:
                        c, t, mat_path = result
                        print(f"\n  [done] {c}/T{sign}{abs(t)} → {mat_path}")
                    else:
                        print(f"\n  [FAIL] {corner}/T{sign}{abs(temp)} — see log in {SIM_DIR}")
                except Exception as exc:
                    print(f"\n  [EXC ] {corner}/T{sign}{abs(temp)} raised: {exc}")
                if STOP_FILE.exists() and not stopped:
                    stopped = True
                    print("\n  [STOP] STOP file detected — cancelling pending jobs")
                    for pending in future_to_job:
                        pending.cancel()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Unified LUT characterization — non-uniform VGS+VDS grids, parallel PVT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_lut_char_all.py --list
              python run_lut_char_all.py --device sky130:nfet_01v8 --test-run
              python run_lut_char_all.py --device ihp:sg13_lv_pmos
              python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT SS
              python run_lut_char_all.py --device gf180:nfet_05v0 --corners TT FF
              python run_lut_char_all.py --device gf180:nfet_03v3 --l-range 0:6  # first 6 L values
        """)
    )
    ap.add_argument("--list", action="store_true",
                    help="Print all available device keys and exit")
    ap.add_argument("--device", metavar="PDK:DEVICE",
                    help="Device key, e.g. sky130:nfet_01v8")
    ap.add_argument("--test-run", action="store_true",
                    help="Micro-sweep validation (2 L, 2 VGS, 1 VSB, TT/27°C)")
    ap.add_argument("--corners", nargs="+", metavar="CORNER",
                    help="Override corners (default: all available for device)")
    ap.add_argument("--temps", nargs="+", type=int, default=ALL_TEMPS,
                    metavar="TEMP",
                    help="Temperatures in °C (default: -40 27 125)")
    ap.add_argument("--l-range", metavar="START:STOP",
                    help="Slice of L indices to simulate, e.g. '0:6' for first 6 L values. "
                         "Partial .mat files get a _L{start}to{end}nm suffix and can be "
                         "combined with merge_mats.py. Enables distributing work across machines.")
    ap.add_argument("--workers", type=int, default=None, metavar="N",
                    help=f"Maximum parallel ngspice workers (default: all {N_WORKERS_DEFAULT} CPUs). "
                         "Use a smaller value to leave cores free for other tasks.")
    args = ap.parse_args()

    if STOP_FILE.exists():
        print(f"  [WARN] Removing stale STOP file from previous run: {STOP_FILE}")
        STOP_FILE.unlink()

    if args.list:
        print(f"\nAvailable devices  (CPU count: {os.cpu_count()}):\n")
        for k, c in DEVICES.items():
            vgs_all = build_vgs_all(c.vgs_max)
            vds_all = build_vds_all(c.vds_max)
            print(f"  {k:<35}  PDK={c.pdk:<7} type={c.fet_type}  "
                  f"L={c.l_vec[0]}–{c.l_vec[-1]}µm ({len(c.l_vec)} pts)  "
                  f"VGS={len(vgs_all)} pts ({_n_vgs_fine(c.vgs_max)}f+{_n_vgs_coarse(c.vgs_max)}c)  "
                  f"VDS={len(vds_all)} pts  "
                  f"corners={list(c.lib_corner_map.keys())}")
        return

    if not args.device:
        ap.error("--device is required (or use --list to see options)")

    if args.device not in DEVICES:
        ap.error(f"Unknown device '{args.device}'. Use --list to see available keys.")

    cfg = DEVICES[args.device]

    # Determine corners
    available = list(cfg.lib_corner_map.keys())
    if args.corners:
        bad = [c for c in args.corners if c not in available]
        if bad:
            ap.error(f"Corner(s) {bad} not available for {cfg.key}. "
                     f"Available: {available}")
        corners = args.corners
    else:
        corners = available

    # Determine L vector (full or sliced via --l-range)
    l_vec = list(cfg.l_vec)
    if args.l_range:
        try:
            s, e = (int(x) for x in args.l_range.split(":"))
        except ValueError:
            ap.error("--l-range must be START:STOP integers, e.g. '0:6'")
        l_vec = l_vec[s:e]
        if not l_vec:
            ap.error(f"--l-range {args.l_range} produced an empty L list for {cfg.key} "
                     f"(has {len(cfg.l_vec)} L values)")

    if not args.test_run:
        vgs_all = build_vgs_all(cfg.vgs_max)
        vds_all = build_vds_all(cfg.vds_max)
        n_pvt   = len(corners) * len(args.temps)
        n_sim   = len(l_vec) * len(vgs_all) * len(vds_all) * len(cfg.vsb_vec)
        l_info  = (f"{l_vec[0]}–{l_vec[-1]}µm ({len(l_vec)} pts"
                   + (f", partial {len(l_vec)}/{len(cfg.l_vec)})" if args.l_range else ")"))
        print(f"\nPDK: {cfg.pdk}  Device: {cfg.device}")
        print(f"L:   {l_info}")
        print(f"VGS: {_n_vgs_fine(cfg.vgs_max)} fine (0–{_vgs_fine_max(cfg.vgs_max):.2f}V @10mV) + "
              f"{_n_vgs_coarse(cfg.vgs_max)} coarse = {len(vgs_all)} pts")
        print(f"VDS: 60 fine (0–0.295V @5mV) + "
              f"{_n_vds_coarse(cfg.vds_max)} coarse = {len(vds_all)} pts")
        print(f"Corners: {corners}  Temps: {args.temps}  PVT jobs: {n_pvt}")
        cap = args.workers if args.workers is not None else N_WORKERS_DEFAULT
        print(f"Simulations per job: {n_sim:,}  "
              f"Workers: {min(n_pvt, cap)}/{os.cpu_count()} CPUs")

    run_pvt(cfg, corners, args.temps, l_vec=l_vec, test_mode=args.test_run,
            max_workers=args.workers)


if __name__ == "__main__":
    main()
