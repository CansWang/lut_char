#!/usr/bin/env python3
"""
merge_mats.py — Merge partial L-range .mat files into one.

Usage:
    python merge_mats.py PARTIAL1.mat PARTIAL2.mat ... --out MERGED.mat

Each input file must have been produced by run_lut_char_all.py with --l-range.
Files are concatenated along the L axis (axis 0) in the order given;
pass them in ascending L order.

Example:
    python merge_mats.py \\
        output/nfet_03v3_TT_Tp27_L280to600nm.mat \\
        output/nfet_03v3_TT_Tp27_L700to3000nm.mat \\
        --out output/nfet_03v3_TT_Tp27.mat
"""

import argparse
import sys
import numpy as np
import scipy.io

DATA_KEYS = ['ID', 'VT', 'GM', 'GMB', 'GDS',
             'CGG', 'CGB', 'CGD', 'CGS', 'CDD', 'CSS',
             'STH', 'SFL']


def main():
    ap = argparse.ArgumentParser(
        description="Merge partial L-range .mat files along the L axis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("inputs", nargs="+",
                    help="Partial .mat files in ascending L order")
    ap.add_argument("--out", required=True,
                    help="Output merged .mat file path")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("Need at least 2 input files to merge.")

    print(f"Loading {len(args.inputs)} files...")
    parts = []
    for f in args.inputs:
        try:
            parts.append(scipy.io.loadmat(f, simplify_cells=True))
            print(f"  {f}")
        except Exception as exc:
            sys.exit(f"Failed to load {f}: {exc}")

    # Determine the device key (top-level .mat key, e.g. 'nfet_03v3')
    dev_key = [k for k in parts[0] if not k.startswith('_')]
    if len(dev_key) != 1:
        sys.exit(f"Expected exactly one top-level key, found: {dev_key}")
    dev_key = dev_key[0]

    ref = parts[0][dev_key]

    # Validate that VGS, VDS, VSB match across all files
    for i, p in enumerate(parts[1:], start=1):
        d = p[dev_key]
        for axis in ('VGS', 'VDS', 'VSB'):
            if not np.allclose(np.array(d[axis]), np.array(ref[axis])):
                sys.exit(f"File {args.inputs[i]}: {axis} grid does not match "
                         f"{args.inputs[0]}. Cannot merge.")

    # Concatenate L and all data arrays along axis 0
    merged = dict(ref)
    merged['L'] = np.concatenate([np.atleast_1d(p[dev_key]['L']) for p in parts])
    for k in DATA_KEYS:
        arrays = []
        for p in parts:
            arr = np.array(p[dev_key][k])
            if arr.ndim == 3:          # shape (nVGS, nVDS, nVSB) — single L
                arr = arr[np.newaxis]  # → (1, nVGS, nVDS, nVSB)
            arrays.append(arr)
        merged[k] = np.concatenate(arrays, axis=0)

    scipy.io.savemat(args.out, {dev_key: merged})
    print(f"\nMerged → {args.out}")
    print(f"  L     : {merged['L']}")
    print(f"  shape : {merged['ID'].shape}  (nL, nVGS, nVDS, nVSB)")


if __name__ == "__main__":
    main()
