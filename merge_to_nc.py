#!/usr/bin/env python3
"""
merge_to_nc.py — Merge per-(corner, temp) .mat LUT files into one NetCDF4 file per device.

Usage:
    python merge_to_nc.py [--input-dir DIR] [--output-dir DIR] [--device PATTERN]

Scans input-dir for *.mat files (skipping partial-L files like _L280to600nm.mat),
groups by device name, and writes {output-dir}/{device}.nc with dims:
    (corner, temp, L, VGS, VDS, VSB)

Example:
    python merge_to_nc.py --input-dir output/ --output-dir output/
    python merge_to_nc.py --input-dir output/ --device nfet_03v3
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import xarray as xr

# Canonical corner order
CORNER_ORDER = ["TT", "FF", "SS", "SF", "FS"]

DATA_KEYS = [
    "ID", "VT", "GM", "GMB", "GDS",
    "CGG", "CGB", "CGD", "CGS", "CDD", "CSS",
    "STH", "SFL", "VDSAT",
]

# Matches: {device}_{corner}_T{p|m}{abs_temp}.mat
# Partial-L files (e.g. _L280to600nm.mat) are excluded by anchoring at .mat
_FILE_RE = re.compile(
    r'^(?P<device>.+?)_(?P<corner>TT|FF|SS|SF|FS)_T(?P<tsign>[pm])(?P<tval>\d+)\.mat$'
)


def collect_files(input_dir: Path, device_filter=None):
    """
    Scan input_dir for *.mat files and return:
        { device_name: { (corner, temp): Path } }
    Skips partial-L files (those with _L<N>to<M>nm in the name).
    """
    by_device = {}
    for path in sorted(input_dir.glob("*.mat")):
        fname = path.name
        # Skip partial-L files
        if re.search(r'_L\d+to\d+nm', fname):
            continue
        m = _FILE_RE.match(fname)
        if not m:
            continue
        device = m.group("device")
        corner = m.group("corner")
        tsign  = m.group("tsign")
        tval   = int(m.group("tval"))
        temp   = tval if tsign == "p" else -tval

        if device_filter and device_filter not in device:
            continue

        by_device.setdefault(device, {})[corner, temp] = path

    return by_device


def load_mat(path: Path) -> dict:
    """Load a .mat file and return the inner data dict."""
    raw = scipy.io.loadmat(str(path), simplify_cells=True)
    keys = [k for k in raw if not k.startswith("_")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected 1 top-level key, found {keys}")
    return raw[keys[0]]


def build_dataset(files_by_ct: dict, device_name: str) -> xr.Dataset:
    """
    Build an xr.Dataset with dims (corner, temp, L, VGS, VDS, VSB).

    files_by_ct: { (corner, temp): Path }
    """
    # --- 1. Determine canonical corners / sorted temps ---
    present_corners = {c for c, _ in files_by_ct}
    present_temps   = sorted({t for _, t in files_by_ct})

    corners = [c for c in CORNER_ORDER if c in present_corners]
    # Add any corners not in CORNER_ORDER (shouldn't happen, but be safe)
    corners += sorted(present_corners - set(corners))
    temps = present_temps  # already sorted

    nC, nT = len(corners), len(temps)

    # --- 2. Load one file to determine coordinate arrays ---
    first_path = next(iter(files_by_ct.values()))
    d0 = load_mat(first_path)

    L_arr   = np.atleast_1d(np.array(d0["L"],   dtype=float))
    VGS_arr = np.atleast_1d(np.array(d0["VGS"], dtype=float))
    VDS_arr = np.atleast_1d(np.array(d0["VDS"], dtype=float))
    # VSB may be a full tensor column; take unique values
    VSB_arr = np.unique(np.array(d0["VSB"], dtype=float).ravel())

    nL, nVGS, nVDS, nVSB = len(L_arr), len(VGS_arr), len(VDS_arr), len(VSB_arr)

    # --- 3. Pre-allocate NaN arrays for all DATA_KEYS ---
    shape = (nC, nT, nL, nVGS, nVDS, nVSB)
    arrays = {k: np.full(shape, np.nan, dtype=float) for k in DATA_KEYS}

    # Scalar metadata (W, NFING) from first file
    W     = float(np.array(d0.get("W",     np.nan)).flat[0])
    NFING = float(np.array(d0.get("NFING", np.nan)).flat[0])

    # --- 4. Fill from each file ---
    corner_idx = {c: i for i, c in enumerate(corners)}
    temp_idx   = {t: i for i, t in enumerate(temps)}

    for (corner, temp), path in files_by_ct.items():
        ci = corner_idx[corner]
        ti = temp_idx[temp]
        try:
            d = load_mat(path)
        except Exception as exc:
            print(f"  [warn] could not load {path}: {exc}", file=sys.stderr)
            continue

        for k in DATA_KEYS:
            if k not in d:
                continue
            arr = np.array(d[k], dtype=float)
            # Shape from .mat is (nL, nVGS, nVDS, nVSB) or (nVGS, nVDS, nVSB) for single L
            if arr.ndim == 3:
                arr = arr[np.newaxis]   # → (1, nVGS, nVDS, nVSB)
            if arr.shape != (nL, nVGS, nVDS, nVSB):
                print(
                    f"  [warn] {path.name} key={k} shape {arr.shape} "
                    f"!= expected {(nL, nVGS, nVDS, nVSB)}, skipping",
                    file=sys.stderr,
                )
                continue
            arrays[k][ci, ti] = arr

    # --- 5. Build xr.Dataset, drop all-NaN variables ---
    dims = ("corner", "temp", "L", "VGS", "VDS", "VSB")
    coords = {
        "corner": corners,
        "temp":   np.array(temps, dtype=int),
        "L":      L_arr,
        "VGS":    VGS_arr,
        "VDS":    VDS_arr,
        "VSB":    VSB_arr,
    }

    data_vars = {}
    for k in DATA_KEYS:
        arr = arrays[k]
        if not np.all(np.isnan(arr)):
            data_vars[k] = xr.Variable(dims, arr)

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "device":  device_name,
        "W":       W,
        "NFING":   NFING,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    return ds


def main():
    ap = argparse.ArgumentParser(
        description="Merge per-(corner, temp) .mat LUT files into one NetCDF4 file per device.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input-dir",  default="output/", metavar="DIR",
                    help="Directory containing .mat files (default: output/)")
    ap.add_argument("--output-dir", default="output/", metavar="DIR",
                    help="Directory for output .nc files (default: output/)")
    ap.add_argument("--device", default=None, metavar="PATTERN",
                    help="Optional substring filter on device name (e.g. nfet_03v3)")
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_dir} for .mat files...")
    by_device = collect_files(input_dir, device_filter=args.device)

    if not by_device:
        print("No matching .mat files found.")
        return

    for device_name, files_by_ct in sorted(by_device.items()):
        corners_found = sorted({c for c, _ in files_by_ct})
        temps_found   = sorted({t for _, t in files_by_ct})
        print(f"\nDevice: {device_name}")
        print(f"  Files   : {len(files_by_ct)}")
        print(f"  Corners : {corners_found}")
        print(f"  Temps   : {temps_found}")

        try:
            ds = build_dataset(files_by_ct, device_name)
        except Exception as exc:
            print(f"  [ERROR] Failed to build dataset: {exc}", file=sys.stderr)
            continue

        out_path = output_dir / f"{device_name}.nc"
        ds.to_netcdf(str(out_path))
        print(f"  Written : {out_path}")
        print(f"  Shape   : corner={ds.sizes['corner']}  temp={ds.sizes['temp']}  "
              f"L={ds.sizes['L']}  VGS={ds.sizes['VGS']}  "
              f"VDS={ds.sizes['VDS']}  VSB={ds.sizes['VSB']}")
        print(f"  Vars    : {sorted(ds.data_vars)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
