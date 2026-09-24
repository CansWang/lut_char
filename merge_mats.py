#!/usr/bin/env python3
"""Merge partial L-range MAT files produced by run_lut_char_all.py."""

import argparse
import sys

import numpy as np
import scipy.io

from capacitance import CAP_PROFILE_LEGACY, CAP_PROFILE_MATRIX9, LEGACY_CAP_KEYS, MATRIX9_KEYS


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


def _scalar(value):
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.flat[0].item() if arr.size else None


def _profile(data):
    return str(_scalar(data.get("CAPACITANCE_PROFILE", CAP_PROFILE_LEGACY)))


def _profile_keys(data):
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
        raise ValueError(f"{key} has {arr.size} values; expected {expected_size}")
    return arr.reshape(expected_shape)


def _load(path):
    raw = scipy.io.loadmat(path, simplify_cells=True)
    keys = [key for key in raw if not key.startswith("_")]
    if len(keys) != 1:
        raise ValueError(f"expected exactly one top-level key, found: {keys}")
    return keys[0], raw[keys[0]]


def merge_parts(input_paths, output_path):
    loaded = []
    for path in input_paths:
        key, data = _load(path)
        loaded.append((path, key, data))

    ref_path, dev_key, ref = loaded[0]
    tensor_keys = _profile_keys(ref)
    expected_tail = tuple(len(np.atleast_1d(ref[axis])) for axis in ("VGS", "VDS", "VSB"))

    for path, key, data in loaded:
        if key != dev_key:
            raise ValueError(f"{path}: top-level key '{key}' does not match '{dev_key}'")
        if _profile_keys(data) != tensor_keys:
            raise ValueError(f"{path}: tensor keys/profile do not match {ref_path}")
        for axis in ("VGS", "VDS", "VSB"):
            if not np.allclose(np.atleast_1d(data[axis]), np.atleast_1d(ref[axis])):
                raise ValueError(f"{path}: {axis} grid does not match {ref_path}")
        for meta_key in PROFILE_METADATA_KEYS:
            if _scalar(data.get(meta_key)) != _scalar(ref.get(meta_key)):
                raise ValueError(f"{path}: metadata '{meta_key}' does not match {ref_path}")
        n_l = len(np.atleast_1d(data["L"]))
        expected_shape = (n_l,) + expected_tail
        for tensor_key in tensor_keys:
            try:
                _as_tensor(data, tensor_key, expected_shape)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc

    loaded.sort(key=lambda item: float(np.min(np.atleast_1d(item[2]["L"]))))
    merged = dict(ref)
    merged["L"] = np.concatenate([np.atleast_1d(data["L"]) for _, _, data in loaded])
    if np.any(np.diff(merged["L"]) <= 0):
        raise ValueError("merged L coordinates must be strictly increasing and non-overlapping")
    for key in tensor_keys:
        arrays = []
        for _, _, data in loaded:
            shape = (len(np.atleast_1d(data["L"])),) + expected_tail
            arrays.append(_as_tensor(data, key, shape))
        merged[key] = np.concatenate(arrays, axis=0)

    scipy.io.savemat(output_path, {dev_key: merged})
    return dev_key, merged


def main():
    ap = argparse.ArgumentParser(
        description="Merge partial L-range .mat files along the L axis."
    )
    ap.add_argument("inputs", nargs="+", help="Partial .mat files")
    ap.add_argument("--out", required=True, help="Output merged .mat file path")
    args = ap.parse_args()
    if len(args.inputs) < 2:
        ap.error("Need at least 2 input files to merge.")

    print(f"Loading {len(args.inputs)} files...")
    for path in args.inputs:
        print(f"  {path}")
    try:
        _, merged = merge_parts(args.inputs, args.out)
    except Exception as exc:
        sys.exit(f"Merge failed: {exc}")
    print(f"\nMerged -> {args.out}")
    print(f"  L     : {merged['L']}")
    print(f"  shape : {merged['ID'].shape}  (nL, nVGS, nVDS, nVSB)")


if __name__ == "__main__":
    main()
