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
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io
import xarray as xr
from netCDF4 import Dataset

from capacitance import CAP_PROFILE_LEGACY, CAP_PROFILE_MATRIX9, LEGACY_CAP_KEYS, MATRIX9_KEYS

# Canonical corner order
CORNER_ORDER = ["TT", "FF", "SS", "SF", "FS"]

CORE_DATA_KEYS = ("ID", "VT", "GM", "GMB", "GDS")
OPTIONAL_DATA_KEYS = (
    "STH", "SFL", "VDSAT", "VDSSAT", "IGD", "IGS", "CGE",
    "CJDT", "CJST", "CGDEXT", "CGSEXT", "CGBOV", "CFGEO",
)
DATA_KEYS = list(CORE_DATA_KEYS + LEGACY_CAP_KEYS + OPTIONAL_DATA_KEYS)
PROFILE_METADATA_KEYS = (
    "CAPACITANCE_PROFILE", "CAPACITANCE_SCHEMA_VERSION",
    "CAPACITANCE_CONVENTION", "CAPACITANCE_TERMINALS",
    "CAPACITANCE_COMPONENTS", "MODEL_FAMILY", "BULK_TERMINAL_ALIAS",
    "CAPACITANCE_JUNCTION_MODE", "NOISE_FREQ_HZ", "SIMULATOR",
)

# Matches: {device}_{corner}_T{p|m}{abs_temp}{grid_suffix}.mat
# grid_suffix is the file-name suffix added by run_lut_char_all.py to
# differentiate sweeps that share the same (device, corner, temp):
#     _uvgs{X}mV_uvds{Y}mV_vsb{N}   (uniform-grid runs)
#     _vsb{N}                       (non-uniform runs)
#     (empty)                       (legacy files predating the suffix)
# Partial-L files (e.g. _L280to600nm.mat) are excluded by the separate
# _L\d+to\d+nm check in collect_files.
_FILE_RE = re.compile(
    r'^(?P<device>.+)_(?P<corner>[A-Za-z0-9]+)_T(?P<tsign>[pm])(?P<tval>\d+)'
    r'(?P<suffix>(?:_uvgs[\d.]+mV_uvds[\d.]+mV)?(?:_vsb\d+)?)'
    r'(?P<profile>_cm9)?'
    r'\.mat$'
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
        suffix = m.group("suffix") or ""
        profile_suffix = m.group("profile") or ""
        corner = m.group("corner")
        tsign  = m.group("tsign")
        tval   = int(m.group("tval"))
        temp   = tval if tsign == "p" else -tval

        if device_filter and device_filter not in device:
            continue

        # Group separately per grid-suffix so different sweeps don't collide
        # in the output .nc filename.
        device_key = f"{device}{suffix}{profile_suffix}"
        group = by_device.setdefault(device_key, {})
        if (corner, temp) in group:
            raise ValueError(f"duplicate PVT files: {group[corner, temp]} and {path}")
        group[corner, temp] = path

    return by_device


def load_mat(path: Path) -> dict:
    """Load a .mat file and return the inner data dict."""
    raw = scipy.io.loadmat(str(path), simplify_cells=True)
    keys = [k for k in raw if not k.startswith("_")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected 1 top-level key, found {keys}")
    return raw[keys[0]]


def _scalar(value):
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.flat[0].item() if arr.size else None


def _profile(data):
    return str(_scalar(data.get("CAPACITANCE_PROFILE", CAP_PROFILE_LEGACY)))


def _data_keys(data):
    profile = _profile(data)
    if profile == CAP_PROFILE_MATRIX9:
        caps = MATRIX9_KEYS
    elif profile == CAP_PROFILE_LEGACY:
        caps = LEGACY_CAP_KEYS
    else:
        raise ValueError(f"unsupported capacitance profile '{profile}'")
    required = CORE_DATA_KEYS + caps
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{profile} data is missing required keys: {', '.join(missing)}")
    return required + tuple(key for key in OPTIONAL_DATA_KEYS if key in data)


def _as_tensor(data, key, expected_shape):
    arr = np.asarray(data[key], dtype=float)
    expected_size = int(np.prod(expected_shape))
    if arr.size != expected_size:
        raise ValueError(
            f"{key} has {arr.size} values; expected {expected_size} for shape {expected_shape}"
        )
    return arr.reshape(expected_shape)


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
    profile = _profile(d0)
    data_keys = _data_keys(d0)

    L_arr   = np.atleast_1d(np.array(d0["L"],   dtype=float))
    VGS_arr = np.atleast_1d(np.array(d0["VGS"], dtype=float))
    VDS_arr = np.atleast_1d(np.array(d0["VDS"], dtype=float))
    # VSB may be a full tensor column; take unique values
    VSB_arr = np.unique(np.array(d0["VSB"], dtype=float).ravel())

    nL, nVGS, nVDS, nVSB = len(L_arr), len(VGS_arr), len(VDS_arr), len(VSB_arr)

    # --- 3. Pre-allocate NaN arrays for this capacitance profile ---
    shape = (nC, nT, nL, nVGS, nVDS, nVSB)
    arrays = {k: np.full(shape, np.nan, dtype=float) for k in data_keys}

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

        if _profile(d) != profile or _data_keys(d) != data_keys:
            raise ValueError(f"{path}: profile/tensor keys do not match {first_path}")
        for key in PROFILE_METADATA_KEYS:
            if _scalar(d.get(key)) != _scalar(d0.get(key)):
                raise ValueError(f"{path}: metadata '{key}' does not match {first_path}")
        for axis, ref_axis in (("L", L_arr), ("VGS", VGS_arr),
                               ("VDS", VDS_arr), ("VSB", VSB_arr)):
            values = np.unique(np.asarray(d[axis], dtype=float).ravel()) if axis == "VSB" \
                else np.atleast_1d(np.asarray(d[axis], dtype=float))
            if not np.allclose(values, ref_axis):
                raise ValueError(f"{path}: {axis} grid does not match {first_path}")

        for k in data_keys:
            arr = _as_tensor(d, k, (nL, nVGS, nVDS, nVSB))
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
    for k in data_keys:
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
    for key in PROFILE_METADATA_KEYS:
        value = _scalar(d0.get(key))
        if value is not None:
            ds.attrs[key] = value
    return ds


def export_group_streaming(files_by_ct: dict, device_name: str, out_path: Path,
                           expected_corners=None, expected_temps=None) -> None:
    """Write one PVT at a time so dense Spectre collections fit in memory."""
    found_corners = {corner for corner, _ in files_by_ct}
    found_temps = {temp for _, temp in files_by_ct}
    corners = ([c for c in CORNER_ORDER if c in found_corners]
               + sorted(found_corners - set(CORNER_ORDER)))
    temps = sorted(found_temps)
    if expected_corners is not None:
        if set(expected_corners) != found_corners:
            raise ValueError(f"corner set {sorted(found_corners)} != expected {expected_corners}")
        corners = list(expected_corners)
    if expected_temps is not None:
        if set(expected_temps) != found_temps:
            raise ValueError(f"temperature set {temps} != expected {expected_temps}")
        temps = list(expected_temps)
    missing = {(c, t) for c in corners for t in temps} - set(files_by_ct)
    if missing:
        raise ValueError(f"missing PVT files: {sorted(missing)}")

    first_path = next(iter(files_by_ct.values()))
    first = load_mat(first_path)
    keys = _data_keys(first)
    axes = {
        axis: np.unique(np.asarray(first[axis], dtype=float).ravel()) if axis == "VSB"
        else np.atleast_1d(np.asarray(first[axis], dtype=float))
        for axis in ("L", "VGS", "VDS", "VSB")
    }
    shape = tuple(len(axes[axis]) for axis in axes)
    dims = ("corner", "temp", "L", "VGS", "VDS", "VSB")
    tmp_path = out_path.with_name("." + out_path.name + ".tmp")
    try:
        with Dataset(tmp_path, "w", format="NETCDF4") as nc:
            nc.createDimension("corner", len(corners))
            nc.createDimension("temp", len(temps))
            for axis in axes:
                nc.createDimension(axis, len(axes[axis]))
            nc.createVariable("corner", str, ("corner",))[:] = np.asarray(corners, dtype=object)
            nc.createVariable("temp", "i4", ("temp",))[:] = temps
            for axis, values in axes.items():
                nc.createVariable(axis, "f8", (axis,))[:] = values
            variables = {
                key: nc.createVariable(key, "f8", dims, zlib=True, complevel=4,
                                       shuffle=True, fill_value=np.nan,
                                       chunksizes=(1, 1, 1, min(shape[1], 64),
                                                   min(shape[2], 64), 1))
                for key in keys
            }
            nc.setncattr("device", device_name)
            nc.setncattr("W", float(_scalar(first.get("W", np.nan))))
            nc.setncattr("NFING", float(_scalar(first.get("NFING", np.nan))))
            if "NF" in first:
                nc.setncattr("NF", float(_scalar(first["NF"])))
            nc.setncattr("created", datetime.now(timezone.utc).isoformat())
            for meta_key in PROFILE_METADATA_KEYS:
                value = _scalar(first.get(meta_key))
                if value is not None:
                    nc.setncattr(meta_key, value)
            for ci, corner in enumerate(corners):
                for ti, temp in enumerate(temps):
                    path = files_by_ct[corner, temp]
                    data = load_mat(path)
                    if _data_keys(data) != keys:
                        raise ValueError(f"{path}: tensor keys differ from {first_path}")
                    for meta_key in PROFILE_METADATA_KEYS:
                        if _scalar(data.get(meta_key)) != _scalar(first.get(meta_key)):
                            raise ValueError(f"{path}: metadata {meta_key} differs from {first_path}")
                    for axis, expected in axes.items():
                        actual = (np.unique(np.asarray(data[axis], dtype=float).ravel())
                                  if axis == "VSB" else np.atleast_1d(np.asarray(data[axis], dtype=float)))
                        if actual.shape != expected.shape or not np.allclose(actual, expected):
                            raise ValueError(f"{path}: {axis} grid differs from {first_path}")
                    for key in keys:
                        variables[key][ci, ti, :, :, :, :] = _as_tensor(data, key, shape)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    ap.add_argument("--expect-corners", nargs="+", metavar="CORNER",
                    help="Require these corners and every corner-temperature combination")
    ap.add_argument("--expect-temps", nargs="+", type=int, metavar="TEMP",
                    help="Require these temperatures and every corner-temperature combination")
    ap.add_argument("--expect-devices", nargs="+", metavar="DEVICE",
                    help="Require exactly these base device names in the input collection")
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_dir} for .mat files...")
    by_device = collect_files(input_dir, device_filter=args.device)

    if not by_device:
        if args.expect_corners or args.expect_temps or args.expect_devices:
            sys.exit("No matching .mat files found for the required collection.")
        print("No matching .mat files found.")
        return
    if args.expect_devices:
        found_devices = {
            _FILE_RE.match(next(iter(files.values())).name).group("device")
            for files in by_device.values()
        }
        if found_devices != set(args.expect_devices):
            sys.exit(f"Device set {sorted(found_devices)} != expected {args.expect_devices}")

    failures = 0
    for device_name, files_by_ct in sorted(by_device.items()):
        corners_found = sorted({c for c, _ in files_by_ct})
        temps_found   = sorted({t for _, t in files_by_ct})
        print(f"\nDevice: {device_name}")
        print(f"  Files   : {len(files_by_ct)}")
        print(f"  Corners : {corners_found}")
        print(f"  Temps   : {temps_found}")

        try:
            out_path = output_dir / f"{device_name}.nc"
            export_group_streaming(files_by_ct, device_name, out_path,
                                   args.expect_corners, args.expect_temps)
        except Exception as exc:
            print(f"  [ERROR] Failed to export dataset: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"  Written : {out_path}")
        print(f"  PVT files: {len(files_by_ct)}")

    print("\nDone.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
