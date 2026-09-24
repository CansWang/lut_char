# LUT Characterization Pipeline

Generates transistor operating-point lookup tables (LUTs) for open-source PDKs
by sweeping SPICE operating points over a configurable (VGS, VDS) grid and
saving results as `.mat` and NetCDF files. The LUTs support general design-space
exploration and remain compatible with gm/ID-based analysis after a solution or
design space has been selected; the pipeline does not enforce a gm/ID method.

Two grid modes are supported:
- **Non-uniform** (default): fine 10 mV / 5 mV steps near 0 V to capture weak/moderate
  inversion, coarser 25–100 mV steps at higher voltages.
- **Uniform** (`--uniform-grid`): simple 25 mV step for both VGS and VDS across the full
  range. Outputs go into separate `sim/uniform/` and `output/uniform/` subdirectories so
  both grid types can coexist without overwriting each other.

## Supported PDKs and Devices

| Key | PDK | Type | VGS/VDS max |
|-----|-----|------|-------------|
| `sky130:nfet_01v8` | SkyWater 130nm | NFET | 1.8 V |
| `sky130:pfet_01v8` | SkyWater 130nm | PFET | 1.8 V |
| `sky130:nfet_01v8_lvt` | SkyWater 130nm | NFET LVT | 1.8 V |
| `sky130:pfet_01v8_lvt` | SkyWater 130nm | PFET LVT | 1.8 V |
| `ihp:sg13_lv_nmos` | IHP SG13G2 | NFET 1.5 V | 1.5 V |
| `ihp:sg13_lv_pmos` | IHP SG13G2 | PFET 1.5 V | 1.5 V |
| `ihp:sg13_hv_nmos` | IHP SG13G2 | NFET 3.3 V | 3.3 V |
| `ihp:sg13_hv_pmos` | IHP SG13G2 | PFET 3.3 V | 3.3 V |
| `gf180:nfet_03v3` | GF 180nm MCU | NFET 3.3 V | 3.3 V |
| `gf180:pfet_03v3` | GF 180nm MCU | PFET 3.3 V | 3.3 V |
| `gf180:nfet_05v0` | GF 180nm MCU | NFET 5 V | 5.0 V |
| `gf180:pfet_05v0` | GF 180nm MCU | PFET 5 V | 5.0 V |

## Prerequisites

- **ngspice** ≥ 41 (with BSIM4 noise support)
- **Python** ≥ 3.9
- Python packages: `numpy`, `scipy`, `pandas`
- For NetCDF4 export (`merge_to_nc.py`): `xarray`, `netCDF4`

PDK model paths are configured at the top of `run_lut_char_all.py`.

## Quick Start

```bash
# List all devices and their grid sizes
python run_lut_char_all.py --list

# Ultra-fast smoke test — end-to-end check, no validation (~10–30 s per VSB point)
python run_lut_char_all.py --device gf180:nfet_03v3 --smoke

# Validate a device with a micro-sweep (fast, ~2 min)
python run_lut_char_all.py --device gf180:nfet_03v3 --test-run

# Full PVT characterization — all 5 corners × 3 temperatures
python run_lut_char_all.py --device gf180:nfet_03v3

# Restrict corners or temperatures
python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT FF
python run_lut_char_all.py --device gf180:nfet_03v3 --temps 27

# Run corners sequentially in batches of 1 (caps at 3 ngspice processes)
python run_lut_char_all.py --device gf180:nfet_03v3 --corners-per-batch 1

# Run all devices sequentially (omit --device)
python run_lut_char_all.py --corners-per-batch 1

# Run all IHP devices only
python run_lut_char_all.py --node ihp --corners-per-batch 1

# Monitor progress of running simulations (per-corner, per-temperature)
bash monitor.sh

# Full-range VSB sweep with 8 points (0 → −VDD)
python run_lut_char_all.py --device gf180:nfet_03v3 --vsb-points 8

# Uniform 25 mV grid (outputs go to sim/uniform/ and output/uniform/)
python run_lut_char_all.py --device gf180:nfet_03v3 --uniform-grid
python run_lut_char_all.py --device gf180:nfet_03v3 --uniform-grid --corners TT
python run_lut_char_all.py --device gf180:nfet_03v3 --uniform-grid --smoke

# Save outputs to a different disk (e.g. external drive)
python run_lut_char_all.py --device gf180:nfet_03v3 --output-dir /mnt/data/lut_output
python run_lut_char_all.py --device gf180:nfet_03v3 --output-dir /mnt/data/lut_output --sim-dir /mnt/data/lut_sim

# Merge all per-(corner, temp) .mat files into a single labelled NetCDF4 file
python merge_to_nc.py --input-dir output/ --output-dir output/
```

## Spectre-only BSIM-CMG characterization

The Spectre backend is selected with `"simulator": "spectre"` in an external
device JSON. It requires `--cap-matrix --uniform-grid` and uses standalone
Spectre PSF ASCII output; MATLAB and Ocean are not required. The backend runs
one nested VDS/VGS DC sweep and a separate 1 Hz noise sweep per
`(device, corner, temperature, L, VSB)` slice. It saves the signed total
Matrix9, DC fields, native `VDSSAT`, required `STH`/`SFL`, and native
`CJDT`/`CJST`; additional native parasitic and saturation fields can be mapped
in the device JSON. A partial or unreadable sweep fails the job. Completed
slices are cached under `sim/` for restart.

The [public example config](examples/tsmc16_spectre.example.json) follows the
LVT NFET/PFET geometry and model names in the cited MATLAB starter file. Copy
it to **remote protected storage**, then replace the example model paths,
corner sections, geometry, legal voltage ranges, and signal names using the
working PDK testbench. Do not put NDA paths, netlists, model contents, or
result files in this repository. The config's `vdd` sets the endpoint for
`--vsb-points`; eight points therefore span 0 to `-vdd` internally and are
stored as positive VSB magnitudes. The remote safe-operating-area check must
approve that full body-bias range before production. The referenced MATLAB
example itself uses only 0 to 0.1 V VSB.

Load the site's Spectre module and set the model paths and run locations to
protected storage. For the public example JSON, the required shell variables
are `TSMC16_MODEL_TOP`, `TSMC16_MODEL_USAGE`, `TSMC16_CFG`, and
`LUT_RUN_ROOT`. From the repository root, set them to the site's actual paths
before running the commands below:

```bash
export TSMC16_MODEL_TOP=/protected/pdk/toplevel.scs
export TSMC16_MODEL_USAGE=/protected/pdk/usage.scs
export TSMC16_CFG=/protected/tsmc16_spectre.json
export LUT_RUN_ROOT=/protected/tsmc16_luts
```

Copy the example JSON to `TSMC16_CFG` and edit it for the remote deck. Once the
Spectre module is loaded, run:

```bash
# End-to-end smoke: first mapped corner, 27 C, one L and VGS, full VDS and VSB.
python run_lut_char_all.py \
  --device-config "$TSMC16_CFG" --device tsmc16:nch_lvt pch_lvt \
  --cap-matrix --uniform-grid --vgs-step 0.005 --vds-step 0.005 \
  --vsb-points 8 --smoke --workers 1 \
  --sim-dir "$LUT_RUN_ROOT/sim" --output-dir "$LUT_RUN_ROOT/output"

# Corner/temperature gate: one L, one VGS, full VDS, VSB=0 for both devices.
python run_lut_char_all.py \
  --device-config "$TSMC16_CFG" --device tsmc16:nch_lvt pch_lvt \
  --cap-matrix --uniform-grid --vgs-step 0.005 --vds-step 0.005 \
  --vsb-points 1 --smoke --corners TT FF SS SF FS --temps -40 27 125 \
  --workers 1 --sim-dir "$LUT_RUN_ROOT/corner_gate/sim" \
  --output-dir "$LUT_RUN_ROOT/corner_gate/output"

# Timed production-grid slice: one device, one corner/temp, one L and VSB.
python run_lut_char_all.py \
  --device-config "$TSMC16_CFG" --device tsmc16:nch_lvt \
  --cap-matrix --uniform-grid --vgs-step 0.005 --vds-step 0.005 \
  --vsb-points 1 --corners TT --temps 27 --l-range 0:1 --workers 1 \
  --sim-dir "$LUT_RUN_ROOT/pilot/sim" --output-dir "$LUT_RUN_ROOT/pilot/output"

# Full TSMC16 characterization: 5 mV VGS/VDS, 8 VSB, 5 corners, 3 temperatures.
python run_lut_char_all.py \
  --device-config "$TSMC16_CFG" --device tsmc16:nch_lvt pch_lvt \
  --cap-matrix --uniform-grid --vgs-step 0.005 --vds-step 0.005 \
  --vsb-points 8 --corners TT FF SS SF FS --temps -40 27 125 \
  --corners-per-batch 1 --workers 1 \
  --sim-dir "$LUT_RUN_ROOT/sim" --output-dir "$LUT_RUN_ROOT/output"

# Export only after every requested PVT MAT file is present.
python merge_to_nc.py \
  --input-dir "$LUT_RUN_ROOT/output/uniform" \
  --output-dir "$LUT_RUN_ROOT/output/uniform" \
  --expect-devices nch_lvt_mac pch_lvt_mac \
  --expect-corners TT FF SS SF FS --expect-temps -40 27 125
```

Before production, confirm the Spectre license and model includes, all mapped
corners and temperatures, legal L/NFIN/NF values, and the full VSB range.
Inspect both smoke MAT files for finite Matrix9 and saturation values,
nonnegative 1 Hz noise PSD, and native junction terms. Compare native total
capacitances against an independent low-frequency AC admittance check at
representative biases. Keep `spectre_cap_junction_mode="native_total"` if the
reported CDD/CSS include junction capacitance; select `"add_junction"` only
when that check shows the junction branch is absent. This prevents counting
`CJDT`/`CJST` twice. If the deck lacks any required field, stop and correct
the signal map or scope before the dense run.

The example maps native `VDSSAT`, which the public starter does not save. If
the remote deck also exposes a distinct native `vdsat`, add
`"VDSAT": "m0:vdsat"` to each device's `spectre_sat_signals` before the smoke
run; the saved MAT and NetCDF files will then contain both. Do not map VDSAT
to VDSSAT as an alias. `CJDT`, `CJST`, and `CGE` are saved separately from the
normalized Matrix9. Additional native parasitic fields can be added to
`spectre_parasitic_signals` after confirming their exact Spectre names in the
remote operating-point output. The current NetCDF exporter also carries
`CGDEXT`, `CGSEXT`, `CGBOV`, and `CFGEO` when mapped; other custom fields
remain in the per-PVT MAT files until added to the export key list.

The first remote smoke is the acceptance test for the site's Spectre PSF ASCII
layout and device operating-point names. The backend checks trace counts,
VGS/VDS/VSB order, numeric values, required noise contributions, and the 1 Hz
frequency when present. A mismatch stops before writing a MAT file; inspect
the slice's `techsweep.log` and `techsweep.raw` under the chosen simulation
directory, then update the protected signal map or parser before production.

At the example's 1.0 V limit, four lengths, eight VSB points, five corners,
three temperatures, and two devices, a 5 mV grid has about **38.8 million
bias points**, each with DC and noise data. Use the timed slice to estimate
runtime and raw disk space. The NetCDF exporter writes one PVT file at a time
to avoid allocating the complete collection in memory.

## Signed Capacitance Matrix (opt-in)

The authoritative implementation contract and verification record are in
[`CAPACITANCE_MATRIX_SSOT.md`](CAPACITANCE_MATRIX_SSOT.md).

Use `--cap-matrix` when a downstream small-signal or symbolic-analysis tool needs
the full set of independent terminal transcapacitances:

```bash
# BSIM4 or PSP built-in device
python run_lut_char_all.py --device sky130:nfet_01v8 --cap-matrix --test-run

# External model/device definition, including BSIM-CMG
python run_lut_char_all.py --device-config devices.json \
    --device mypdk:nmos --cap-matrix --test-run
```

This profile stores the signed total-capacitance Matrix9 fields:

```
CGG CGD CGS
CDG CDD CDS
CSG CSD CSS
```

The convention is `Cij = dQi/dVj`, terminal order is `G,D,S,B`, and each value
includes the model's intrinsic and extrinsic overlap/junction capacitances. The
bulk row and column are reconstructed by charge conservation and common-mode
invariance:

```
CiB = -(CiG + CiD + CiS)
CBj = -(CGj + CDj + CSj)
CBB = sum(Cij), i,j in {G,D,S}
```

Matrix9 outputs use a `_cm9` filename suffix and therefore cannot overwrite or
be merged with legacy six-capacitance outputs. Before a sweep starts, a one-point
ngspice capability probe verifies that every required native field is available.

Model normalization is handled as follows:

- BSIM4: signed intrinsic 3x3 matrix plus `cgdo`, `cgso`, `cgbo`, `capbd`, and `capbs`.
- PSP 103: normalized PSP off-diagonal signs plus `cgdol`, `cgsol`, `lp_cgbov`, `cjd`, and `cjs`.
- BSIM-CMG: direct total `CGG...CSS` outputs, with the model's off-diagonal
  `-dQi/dVj` reporting normalized to the signed schema. The model must be
  compiled with its `INFO` operating-point outputs enabled.

### External device JSON

`--device-config` is repeatable. Each JSON file contains a `devices` list; keys
must not duplicate built-in or previously loaded devices. Paths and environment
values expand `~` and environment variables.

```json
{
  "devices": [
    {
      "key": "mypdk:nmos",
      "device": "nmos_model",
      "pdk": "mypdk",
      "fet_type": "nfet",
      "model_family": "bsim-cmg",
      "model_lib": "$PDK_ROOT/models/mos.lib",
      "lib_corner_map": {"TT": "tt", "FF": "ff", "SS": "ss"},
      "model_setup_lines": [".lib {MODEL_LIB} {LIB_CORNER}"],
      "instance_template": "XM1 {D} {G} {S} {B} {DEVICE} L={LX} W={W_UM}u NFIN={NFING}",
      "save_pfx": "@m.xm1.m0",
      "l_vec": [0.02, 0.03, 0.04],
      "vgs_max": 0.8,
      "vds_max": 0.8,
      "vsb_vec": [0.0, -0.2, -0.4],
      "has_explicit_u": true,
      "analysis": "op",
      "id_col": "ids",
      "gmb_col": "gmbs",
      "output_aliases": {"id": "ids", "gmb": "gmbs"},
      "bulk_terminal_alias": "E",
      "simulation_env": {"PDK_ROOT": "$HOME/pdks/mypdk"}
    }
  ]
}
```

`bulk_terminal_alias: "E"` maps BSIM-CMG's electrostatic bulk terminal name to
canonical terminal `B` in the saved schema. `output_aliases` maps canonical
lowercase operating-point names to simulator-specific names when they differ.

## Voltage Grid

### Non-uniform grid (default)

Each device uses a non-uniform VGS and VDS grid to capture weak/moderate/strong
inversion transitions with high resolution at low voltages and coarser steps in
strong inversion:

| Region | VGS step | VDS step |
|--------|----------|----------|
| Fine (0 → ~½·VGS_max) | 10 mV | 5 mV (0–0.295 V) |
| Coarse (½·VGS_max → VGS_max) | 25–100 mV | 50–100 mV |

Total point counts by device supply voltage:

| VGS_max | nVGS (non-uniform) | nVDS (non-uniform) |
|---------|--------------------|--------------------|
| 1.2 V   | 91                 | 79                 |
| 1.8 V   | 126                | 91                 |
| 3.3 V   | 187                | 91                 |
| 5.0 V   | 231                | 108                |

### Uniform grid (`--uniform-grid`)

Replaces both grids with a simple 25 mV uniform step from 0 V to VGS_max / VDS_max.
The VSB sweep is unchanged.

| VGS_max / VDS_max | nVGS = nVDS (uniform @25 mV) |
|-------------------|------------------------------|
| 1.2 V             | 49                           |
| 1.8 V             | 73                           |
| 3.3 V             | 133                          |
| 5.0 V             | 201                          |

Output files are written to **`sim/uniform/`** and **`output/uniform/`** so they never
overwrite a non-uniform run:

```
sim/
  techsweep_nfet_03v3_TT_Tp27.spice     ← non-uniform
  uniform/
    techsweep_nfet_03v3_TT_Tp27.spice   ← uniform

output/
  nfet_03v3_TT_Tp27.mat                 ← non-uniform
  uniform/
    nfet_03v3_TT_Tp27.mat               ← uniform
```

### VSB sweep

By default each device uses a short built-in `vsb_vec` (typically `[0.0, -0.2, -0.4]` V).
Use `--vsb-points N` to override with N evenly spaced points spanning the full 0 → −VDD range:

```bash
# 5 pts on a 1.8 V device → [0.0, -0.45, -0.9, -1.35, -1.8]
python run_lut_char_all.py --device sky130:nfet_01v8 --vsb-points 5

# 8 pts on a 3.3 V device → [0.0, -0.471, -0.943, …, -3.3]
python run_lut_char_all.py --device gf180:nfet_03v3 --vsb-points 8
```

## Test Modes

Three modes are available for validating and profiling the pipeline before committing to a full run:

| Flag | L | VGS | VSB | Temps | Corners | Validates? | Approx time |
|------|---|-----|-----|-------|---------|------------|-------------|
| `--smoke` | 1 | 1 | all (respects `--vsb-points`) | 27°C | TT | No | ~10–30 s × nVSB |
| `--test-run` | 2 | 2 | VSB=0 only | 27°C | TT | Yes (TC1–4) | ~1–5 min |
| *(full run)* | all | all | all | 3 | all 5 | No | hours–days |

- **`--smoke`**: one ngspice call per VSB point; confirms the pipeline runs end-to-end without crashing. Uses the full `vsb_vec` so `--vsb-points` is exercised.
- **`--test-run`**: micro-sweep (VSB=0 only) with TC validation checks (transconductance continuity, noise floor, etc.).

## Concurrency Control

By default all `corners × temps` jobs are submitted to the process pool simultaneously (up to 15 for a full 5-corner × 3-temp run). On memory-limited machines this can cause processes to queue and stall. Use `--corners-per-batch` to cap peak concurrency:

```bash
# 1 corner at a time — 3 ngspice processes peak (default)
python run_lut_char_all.py --device gf180:nfet_03v3 --corners-per-batch 1

# 2 corners at a time — 6 ngspice processes peak
python run_lut_char_all.py --device gf180:nfet_03v3 --corners-per-batch 2

# Also cap total worker threads within each batch
python run_lut_char_all.py --device gf180:nfet_03v3 --corners-per-batch 1 --workers 2
```

Execution flow with `--corners-per-batch 1`:
```
Batch 1/5: [TT] → 3 parallel jobs (TT/-40, TT/27, TT/125)  ← wait
Batch 2/5: [FF] → 3 parallel jobs                           ← wait
Batch 3/5: [SS] → 3 parallel jobs                           ← wait
Batch 4/5: [SF] → 3 parallel jobs                           ← wait
Batch 5/5: [FS] → 3 parallel jobs                           ← wait
```

A STOP file (`touch STOP` in the working directory) halts execution between batches; any batch already in progress finishes cleanly. When `--device` is omitted, all devices run sequentially and the STOP file is also checked between devices.

## Output Location

By default `.mat` files go to `output/` and simulation files (`.spice`, `.txt`, `.log`)
go to `sim/`, both relative to the working directory. Use `--output-dir` and `--sim-dir`
to redirect to a different disk when the main drive is short on space:

```bash
# Send only .mat results to external storage
python run_lut_char_all.py --device ihp:sg13_lv_nmos \
    --output-dir /mnt/data/lut_output

# Send both sim files and .mat results to external storage
python run_lut_char_all.py --node ihp --corners-per-batch 1 \
    --output-dir /mnt/data/lut_output \
    --sim-dir /mnt/data/lut_sim
```

The `uniform/` subdirectory structure is preserved when `--uniform-grid` is used:

```
/mnt/data/lut_output/
  sg13_lv_nmos_TT_Tp27.mat          ← non-uniform
  uniform/
    sg13_lv_nmos_TT_Tp27.mat        ← --uniform-grid
```

## Progress Monitoring

`monitor.sh` shows live per-corner, per-temperature progress for all GF180 devices.
It auto-detects the VGS × VDS point count and the VSB count directly from the running
netlists, so it works correctly for both the non-uniform and uniform grid modes without
any manual configuration:

```
22:58:04 — Active ngspice: 3 processes

  nfet_03v3  [3/15 done, nVSB=8]
    TT   ✓ done
    FF   ▶ active — Tm40:65% Tp27:62% Tp125:71%
    SS     waiting
    SF     waiting
    FS     waiting
```

```bash
# Run once
bash monitor.sh

# Poll every 60 s
watch -n 60 bash monitor.sh
```

## Output Format

### Per-PVT `.mat` files

Each completed PVT job writes one `.mat` file to `output/` (non-uniform grid) or
`output/uniform/` (uniform grid):

```
output/{device}_{corner}_T{p|m}{temp}.mat           ← non-uniform grid
output/uniform/{device}_{corner}_T{p|m}{temp}.mat   ← --uniform-grid
output/{device}_{corner}_T{p|m}{temp}_vsb{N}_cm9.mat ← --cap-matrix
```

Inside each `.mat`, a single struct named after the device contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `ID` | (nL, nVGS, nVDS, nVSB) | Drain current |
| `GM` | (nL, nVGS, nVDS, nVSB) | Transconductance gm |
| `GDS` | (nL, nVGS, nVDS, nVSB) | Output conductance gds |
| `GMB` | (nL, nVGS, nVDS, nVSB) | Body transconductance gmb |
| `VT` | (nL, nVGS, nVDS, nVSB) | Threshold voltage |
| `CGG/CGD/CGS/CGB` | (nL, nVGS, nVDS, nVSB) | Gate capacitances |
| `CDD/CSS` | (nL, nVGS, nVDS, nVSB) | Drain/source capacitances |
| `STH` | (nL, nVGS, nVDS, nVSB) | Thermal noise PSD |
| `SFL` | (nL, nVGS, nVDS, nVSB) | Flicker noise PSD |
| `VDSAT` | (nL, nVGS, nVDS, nVSB) | Saturation voltage (BSIM4 devices only) |
| `VGS/VDS/VSB/L` | vectors | Axis coordinates |

With `--cap-matrix`, the six legacy capacitance fields are replaced by
`CGG, CGD, CGS, CDG, CDD, CDS, CSG, CSD, CSS`. The MAT struct and resulting
NetCDF file also carry `CAPACITANCE_*`, `MODEL_FAMILY`, and
`BULK_TERMINAL_ALIAS` metadata. Noise fields are retained when the selected
model exposes them and omitted otherwise.

### Per-device `.nc` NetCDF4 file (xarray)

`merge_to_nc.py` merges all per-(corner, temp) `.mat` files for a device into a
single labelled NetCDF4 file suitable for analysis with xarray/dask:

```bash
python merge_to_nc.py --input-dir output/ --output-dir output/
# → output/nfet_03v3.nc, output/pfet_03v3.nc, …

# Filter to a single device
python merge_to_nc.py --device nfet_03v3
```

The resulting dataset has CORNER and TEMP as first-class dimensions:

```python
import xarray as xr
ds = xr.open_dataset("output/nfet_03v3.nc")
# <xarray.Dataset>
# Dimensions:  (corner: 5, temp: 3, L: 12, VGS: 187, VDS: 91, VSB: 8)
# Coordinates:
#   * corner   (corner) <U2  'TT' 'FF' 'SS' 'SF' 'FS'
#   * temp     (temp)   int64  -40  27  125
#   * L        (L)      float64  0.28 … 3.0   [µm]
#   * VGS      (VGS)    float64  0.0  … 3.3   [V]
#   * VDS      (VDS)    float64  0.0  … 3.3   [V]
#   * VSB      (VSB)    float64  0.0  … 3.3   [V, abs]

# Select a slice
ids = ds["ID"].sel(corner="TT", temp=27)        # shape (12, 187, 91, 8)
gm_ff_hot = ds["GM"].sel(corner="FF", temp=125) # shape (12, 187, 91, 8)
```

Missing (corner, temp) combinations are filled with `NaN`; variables absent from
all files (e.g. `VDSAT` on IHP devices) are dropped automatically.

## Interpolation Benchmark

`benchmark_interp.py` compares interpolation **speed vs accuracy** across the
uniform-grid `.mat` files for a chosen device/corner/temp. It sweeps four axes:

- **Grid spacing** — every uniform grid available for the device (e.g. 10 / 25 / 50 mV).
- **Method** — `linear` vs `pchip` (`scipy.interpolate.RegularGridInterpolator`, scipy ≥ 1.13).
- **Domain** — linear vs `asinh(x/scale)` log-domain (sign-preserving, round-trips exactly).
- **Variable** — all 14 LUT vars by default (`ID, VT, GM, GMB, GDS, CGG, CGB, CGD, CGS, CDD, CSS, STH, SFL, VDSAT`).

Accuracy of each coarse grid is measured against the **finest available grid** evaluated
with the *same* (method, domain), so the metric isolates the grid-spacing contribution.

```bash
# Default: sg13_lv_pmos, TT, 27°C, L-idx 0, N=10k queries, 3 repeats (~3 min)
python benchmark_interp.py

# Different device / corner / temperature / L
python benchmark_interp.py --device sg13_lv_nmos --corner SS --temp -40 --l-index 2

# Tighter timing medians (~30 min)
python benchmark_interp.py --n-query 50000 --repeats 5

# Skip the plot or skip accuracy comparison
python benchmark_interp.py --no-plot
python benchmark_interp.py --no-accuracy

# Subset of variables
python benchmark_interp.py --vars ID GM GDS
```

**Methodology** — one shared random query set (fixed seed) of N points in the
axis interior `[a+1%·span, b-1%·span]`. For each combo: one warmup call, then K
timed repeats of a single vectorized batched call; `qps = N / median(times)`.
Build time and query throughput are reported separately. Accuracy metric is
`|y_test − y_ref| / (|y_ref| + median(|y_ref|))`, reported as RMS and p99.

**Outputs**
- Markdown-style table to stdout: `var | grid(mV) | method | domain | build_ms | qps | rms_err | p99_err`.
- `benchmark_interp.png` — speed-vs-error scatter, one subplot per variable
  (color = method, marker = domain, hollow = reference grid).

`VDSAT` is automatically skipped on IHP devices (not produced by the model).

## Distributed Computation

Long simulation times (days to weeks per device) can be reduced by distributing
work across multiple machines. There are three levels of splitting — combine them
as needed.

### Level 1 — By Device (no merge required)

Each machine runs one device independently. Output `.mat` files have unique names
so you simply copy them all to the same `output/` directory when done.

```bash
# Machine A
python run_lut_char_all.py --device gf180:nfet_03v3

# Machine B
python run_lut_char_all.py --device gf180:pfet_03v3
```

### Level 2 — By Corner / Temperature (no merge required)

Split the 15 PVT jobs (5 corners × 3 temperatures) across machines.
Each produces separate per-PVT `.mat` files — collect them in one folder.

```bash
# Machine A — 3 corners, all temperatures (9 jobs)
python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT FF SS

# Machine B — 2 corners, all temperatures (6 jobs)
python run_lut_char_all.py --device gf180:nfet_03v3 --corners SF FS

# Or split by temperature (5 corners × 1 temp = 5 jobs each)
python run_lut_char_all.py --device gf180:nfet_03v3 --temps -40
python run_lut_char_all.py --device gf180:nfet_03v3 --temps 27
python run_lut_char_all.py --device gf180:nfet_03v3 --temps 125
```

### Level 3 — By L Range (merge required)

For the finest granularity, split individual (corner, temp) jobs by L index.
Partial `.mat` files get a `_L{start}to{end}nm` suffix.  Use `merge_mats.py`
to combine them after all machines finish.

```bash
# Machine A — first 6 L values
python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT --temps 27 --l-range 0:6

# Machine B — remaining L values
python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT --temps 27 --l-range 6:12

# Merge on any machine after both finish
python merge_mats.py \
    output/nfet_03v3_TT_Tp27_L280to600nm.mat \
    output/nfet_03v3_TT_Tp27_L700to3000nm.mat \
    --out output/nfet_03v3_TT_Tp27.mat
```

Levels 2 and 3 can be combined freely (e.g., each machine handles one corner with
a specific L range).

## Sequential Pipeline (single machine, all GF180 devices)

```bash
nohup bash run_gf180_sequential.sh > /tmp/gf180_sequential.log 2>&1 &
```

This runs `nfet_03v3 → pfet_03v3 → nfet_05v0 → pfet_05v0` in order, using all
available CPU cores for each device.
