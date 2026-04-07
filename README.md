# LUT Characterization Pipeline

Generates gm/ID design lookup tables (LUTs) for open-source PDKs by sweeping
SPICE operating points over a non-uniform (VGS, VDS) grid and saving results as
`.mat` files compatible with the [gm/ID design methodology].

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
- Python packages: `numpy`, `scipy`
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

# Monitor progress of running simulations (per-corner, per-temperature)
bash monitor.sh

# Full-range VSB sweep with 8 points (0 → −VDD)
python run_lut_char_all.py --device gf180:nfet_03v3 --vsb-points 8

# Merge all per-(corner, temp) .mat files into a single labelled NetCDF4 file
python merge_to_nc.py --input-dir output/ --output-dir output/
```

## Voltage Grid

Each device uses a non-uniform VGS and VDS grid to capture weak/moderate/strong
inversion transitions with high resolution at low voltages and coarser steps in
strong inversion:

| Region | VGS step | VDS step |
|--------|----------|----------|
| Fine (0 → ~½·VGS_max) | 10 mV | 5 mV (0–0.295 V) |
| Coarse (½·VGS_max → VGS_max) | 25–100 mV | 50–100 mV |

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

## Progress Monitoring

`monitor.sh` shows live per-corner, per-temperature progress for all GF180 devices.
It auto-detects the VSB point count from the running netlists so percentages are always accurate:

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

Each completed PVT job writes one `.mat` file to `output/`:

```
output/{device}_{corner}_T{p|m}{temp}.mat
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
