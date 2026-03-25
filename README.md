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

PDK model paths are configured at the top of `run_lut_char_all.py`.

## Quick Start

```bash
# List all devices and their grid sizes
python run_lut_char_all.py --list

# Validate a device with a micro-sweep (fast, ~2 min)
python run_lut_char_all.py --device gf180:nfet_03v3 --test-run

# Full PVT characterization — all 5 corners × 3 temperatures
python run_lut_char_all.py --device gf180:nfet_03v3

# Restrict corners or temperatures
python run_lut_char_all.py --device gf180:nfet_03v3 --corners TT FF
python run_lut_char_all.py --device gf180:nfet_03v3 --temps 27

# Monitor progress of running simulations
bash monitor.sh
```

## Voltage Grid

Each device uses a non-uniform VGS and VDS grid to capture weak/moderate/strong
inversion transitions with high resolution at low voltages and coarser steps in
strong inversion:

| Region | VGS step | VDS step |
|--------|----------|----------|
| Fine (0 → ~½·VGS_max) | 10 mV | 5 mV (0–0.295 V) |
| Coarse (½·VGS_max → VGS_max) | 25–100 mV | 50–100 mV |

## Output Format

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
| `VGS/VDS/VSB/L` | vectors | Axis coordinates |

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
