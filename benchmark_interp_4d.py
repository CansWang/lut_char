#!/usr/bin/env python3
"""
benchmark_interp_4d.py — same comparison as benchmark_interp.py, but on the
FULL 4-D LUT (L × VGS × VDS × VSB). No slice_3d; L is the new interpolation
axis instead of a fixed index.

Sweeps:
    grid spacing  (5/10/25/50/100 mV — auto-discovered)
    method        (linear / pchip)
    domain        (linear / log)
    variable      (ID, GM, GDS, CGG, ... — all 14 by default)

Per-axis 1-D sweeps now include L (in addition to VGS, VDS); VSB sweep is
skipped, matching the 3-D file's convention.

Usage:
    python benchmark_interp_4d.py
    python benchmark_interp_4d.py --pchip-impl fast
    python benchmark_interp_4d.py --device sg13_lv_pmos --corner SS --temp -40
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from benchmark_interp import (
    DATA_KEYS, METHODS, DOMAINS, ABS_FLOOR,
    discover_files, load_mat_table, validate_axes,
    _AsinhWrapped, _CombinedInterp,
    relative_error, relative_error_true,
    print_table, make_plot,
    time_query,
)


# ---------------------------------------------------------------------------
# 4-D interpolator factory
# ---------------------------------------------------------------------------

def make_interp_4d(table, var: str, method: str, domain: str):
    """Single-output 4-D interpolator. Retained for parity with the 3-D file;
    the main loop uses `make_combined_interp_4d` for speed."""
    axes = table["axes"]
    grid = (axes["L"], axes["VGS"], axes["VDS"], axes["VSB"])
    values = table["data"][var]
    if domain == "log":
        scale = float(np.median(np.abs(values)))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        transformed = np.arcsinh(values / scale)
    else:
        transformed = values
    rgi = RegularGridInterpolator(
        grid, transformed, method=method,
        bounds_error=False, fill_value=np.nan,
    )
    return _AsinhWrapped(rgi, scale) if domain == "log" else rgi


def make_combined_interp_4d(table, method: str, domain: str):
    """One RGI over values of shape (nL, nVGS, nVDS, nVSB, n_vars). Each call
    returns (N, n_vars) so the locate/weight work is amortised across all 14
    variables — same trick as the 3-D file, one extra axis."""
    axes = table["axes"]
    grid = (axes["L"], axes["VGS"], axes["VDS"], axes["VSB"])
    var_names = [v for v in DATA_KEYS if v in table["data"]]

    stacked = []
    if domain == "log":
        scales = np.empty(len(var_names), dtype=float)
        for i, v in enumerate(var_names):
            vals = table["data"][v]
            scale = float(np.median(np.abs(vals)))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            scales[i] = scale
            stacked.append(np.arcsinh(vals / scale))
    else:
        scales = None
        for v in var_names:
            stacked.append(table["data"][v])
    # np.stack(..., axis=-1) often returns a non-c_contiguous view (numpy
    # gotcha); force a contig copy so downstream code (especially the
    # pchip_numba kernel wrapper) doesn't pay a per-call 100-400 ms memcpy
    # on the larger grids.
    values = np.ascontiguousarray(np.stack(stacked, axis=-1))

    rgi = RegularGridInterpolator(
        grid, values, method=method,
        bounds_error=False, fill_value=np.nan,
    )
    return _CombinedInterp(rgi, var_names, scales)


# ---------------------------------------------------------------------------
# 4-D query workload
# ---------------------------------------------------------------------------

# Voltage pins — same grid-aligned choices as the 3-D file.
_AXIS_PIN_VOLT = {"VGS": 0.6, "VDS": 0.6, "VSB": 0.0}


def make_query_set_4d(axes_4d, n: int, seed: int = 0):
    """Random uniform query in interior [a+1%span, b-1%span] for each of 4 axes."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 4), dtype=float)
    for i, k in enumerate(("L", "VGS", "VDS", "VSB")):
        a, b = float(axes_4d[k].min()), float(axes_4d[k].max())
        margin = 0.01 * (b - a)
        lo, hi = a + margin, b - margin
        if hi <= lo:
            out[:, i] = a
        else:
            out[:, i] = rng.uniform(lo, hi, size=n)
    return out


def make_query_set_axis_4d(axes_4d, n: int, sweep_axis: str, seed: int = 0):
    """N points varying only along `sweep_axis`. Other 3 axes pinned to grid-
    aligned values: VGS/VDS/VSB to `_AXIS_PIN_VOLT`, L to L[0] (Lmin — always
    a sample point)."""
    if sweep_axis not in ("L", "VGS", "VDS", "VSB"):
        raise ValueError(f"sweep_axis must be L/VGS/VDS/VSB, got {sweep_axis!r}")
    rng = np.random.default_rng(seed)
    out = np.empty((n, 4), dtype=float)
    L_pin = float(axes_4d["L"][0])
    for i, k in enumerate(("L", "VGS", "VDS", "VSB")):
        ax = axes_4d[k]
        if k == sweep_axis:
            a, b = float(ax.min()), float(ax.max())
            margin = 0.01 * (b - a)
            lo, hi = a + margin, b - margin
            if hi <= lo:
                out[:, i] = a
            else:
                out[:, i] = rng.uniform(lo, hi, size=n)
        else:
            pin = L_pin if k == "L" else float(np.clip(
                _AXIS_PIN_VOLT[k], ax.min(), ax.max()))
            out[:, i] = pin
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-dir", default="output/uniform/")
    ap.add_argument("--device", default="sg13_lv_pmos",
                    help="Device name (default: sg13_lv_pmos — 5 grid spacings).")
    ap.add_argument("--corner", default="TT")
    ap.add_argument("--temp", type=int, default=27, help="°C")
    ap.add_argument("--n-query", type=int, default=300,
                    help="Query set size (default: 300). Pchip on 4-D is "
                         "~3× slower per query than on 3-D, so keep small.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vars", nargs="+", default=None,
                    help="Variable subset (default: all 14).")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="benchmark_interp_4d.png")
    ap.add_argument("--pchip-impl", choices=("scipy", "fast", "numba"),
                    default="scipy",
                    help="Pchip impl: 'scipy' = stock RGI; 'fast' = "
                         "fast_pchip monkey-patch (skips per-call BPoly "
                         "construction, scalar-pt local stencil); 'numba' = "
                         "pchip_numba JIT 4-D local-stencil kernel (O(4**ndim) "
                         "per query, flattens the trail-dim scaling). All "
                         "three are bit-exact; only the pchip qps changes.")
    args = ap.parse_args()

    if args.pchip_impl == "fast":
        import fast_pchip
        fast_pchip.enable()
    elif args.pchip_impl == "numba":
        import pchip_numba
        pchip_numba.enable()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    files = discover_files(input_dir, args.device, args.corner, args.temp)
    if not files:
        sys.exit(f"No files matching {args.device} {args.corner} {args.temp}°C "
                 f"in {input_dir}")

    print(f"Device : {args.device}  ({args.corner}, {args.temp}°C, 4-D LUT)")
    print(f"Query  : N={args.n_query}, repeats={args.repeats}, seed={args.seed}")
    pchip_note = {
        "scipy": " (stock scipy)",
        "fast":  " (fast_pchip patched — scalar-pt local stencil)",
        "numba": " (pchip_numba patched — JIT 4-D local-stencil reduction)",
    }[args.pchip_impl]
    print(f"Pchip  : {args.pchip_impl}{pchip_note}")
    print(f"Found  : {len(files)} grid(s):")
    for f in files:
        print(f"          {f['vgs_step']*1000:g} mV  →  {f['path'].name}")

    print("\nLoading & validating...")
    tables = [load_mat_table(f["path"]) for f in files]
    for t in tables[1:]:
        validate_axes(tables[0], t)
    print("Axes validated (L, VSB identical; VGS/VDS bounds match).")
    for t in tables:
        t["meta"]["vgs_step_mv"] = t["meta"]["vgs_step"] * 1000.0

    ref = tables[0]
    print(f"Reference grid: {ref['meta']['vgs_step_mv']:g} mV (finest)")
    L_axis = ref["axes"]["L"]
    print(f"L axis        : {len(L_axis)} value(s), "
          f"range [{L_axis.min():g}, {L_axis.max():g}]")

    available = set(ref["data"])
    if args.vars:
        missing = [v for v in args.vars if v not in available]
        if missing:
            print(f"  [warn] requested vars not in data, skipping: {missing}")
        vars_to_run = [v for v in args.vars if v in available]
    else:
        vars_to_run = [v for v in DATA_KEYS if v in available]
    if not vars_to_run:
        sys.exit("No variables to benchmark.")

    sweeps = [
        ("4D",  make_query_set_4d(ref["axes"], args.n_query, seed=args.seed)),
        ("L",   make_query_set_axis_4d(ref["axes"], args.n_query, "L",
                                       seed=args.seed)),
        ("VGS", make_query_set_axis_4d(ref["axes"], args.n_query, "VGS",
                                       seed=args.seed)),
        ("VDS", make_query_set_axis_4d(ref["axes"], args.n_query, "VDS",
                                       seed=args.seed)),
    ]
    L_pin = float(L_axis[0])
    pin_vgs = float(np.clip(_AXIS_PIN_VOLT["VGS"],
                            ref["axes"]["VGS"].min(), ref["axes"]["VGS"].max()))
    pin_vds = float(np.clip(_AXIS_PIN_VOLT["VDS"],
                            ref["axes"]["VDS"].min(), ref["axes"]["VDS"].max()))
    pin_vsb = float(np.clip(_AXIS_PIN_VOLT["VSB"],
                            ref["axes"]["VSB"].min(), ref["axes"]["VSB"].max()))
    print(f"Sweeps : 4D random; "
          f"L @ (VGS={pin_vgs:g},VDS={pin_vds:g},VSB={pin_vsb:g}); "
          f"VGS @ (L={L_pin:g},VDS={pin_vds:g},VSB={pin_vsb:g}); "
          f"VDS @ (L={L_pin:g},VGS={pin_vgs:g},VSB={pin_vsb:g})")

    print("\nRunning benchmark...")
    rows = []
    cache_ref = {}  # (axis, var, method, domain) → y_ref
    for method in METHODS:
        for domain in DOMAINS:
            for tab in tables:
                grid_mv = tab["meta"]["vgs_step_mv"]
                t0 = time.perf_counter()
                interp = make_combined_interp_4d(tab, method, domain)
                build_s = time.perf_counter() - t0
                var_names = interp.var_names
                is_ref = (tab is ref)
                for axis_name, query_pts in sweeps:
                    query_s, y_all = time_query(interp, query_pts, args.repeats)
                    qps = args.n_query / query_s if query_s > 0 else np.inf

                    for i, var in enumerate(var_names):
                        if var not in vars_to_run:
                            continue
                        y_var = y_all[:, i]

                        rms = p99 = mx = np.nan
                        rms_t = p99_t = mx_t = np.nan
                        n_kept = n_total = 0
                        key = (axis_name, var, method, domain)
                        if is_ref:
                            cache_ref[key] = y_var
                        elif not args.no_accuracy:
                            y_ref = cache_ref.get(key)
                            if y_ref is not None:
                                rms, p99, mx = relative_error(y_var, y_ref)
                                floor = ABS_FLOOR.get(var, 0.0)
                                rms_t, p99_t, mx_t, n_kept, n_total = \
                                    relative_error_true(y_var, y_ref, floor)

                        rows.append({
                            "axis":     axis_name,
                            "var":      var,
                            "grid(mV)": f"{grid_mv:g}",
                            "method":   method,
                            "domain":   domain,
                            "build_ms": f"{build_s*1000:.1f}",
                            "qps":      f"{qps:,.0f}",
                            "rms_err":  ("—" if is_ref else
                                         (f"{rms:.2e}" if np.isfinite(rms) else "—")),
                            "p99_err":  ("—" if is_ref else
                                         (f"{p99:.2e}" if np.isfinite(p99) else "—")),
                            "max_err":  ("—" if is_ref else
                                         (f"{mx:.2e}" if np.isfinite(mx) else "—")),
                            "rms_rel":  ("—" if is_ref else
                                         (f"{rms_t:.2e}" if np.isfinite(rms_t) else "—")),
                            "max_rel":  ("—" if is_ref else
                                         (f"{mx_t:.2e}" if np.isfinite(mx_t) else "—")),
                            "kept":     ("—" if is_ref else
                                         f"{n_kept}/{n_total}"),
                            "_qps":     qps,
                            "_rms_err": rms,
                            "_max_err": mx,
                            "_rms_rel": rms_t,
                            "_max_rel": mx_t,
                        })

    print("\n=== Results ===\n")
    print_table([{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in rows])

    if not args.no_plot:
        base_title = (f" — {args.device} {args.corner} {args.temp}°C "
                      f"(4-D L×VGS×VDS×VSB)")
        out_path = Path(args.out)
        stem, ext = out_path.stem, out_path.suffix
        print()

        ref_grid_label = f"{ref['meta']['vgs_step_mv']:g}"
        rows_plot = [r for r in rows if r["grid(mV)"] != ref_grid_label]
        print(f"Plotting (excluding reference grid {ref_grid_label} mV).")

        rows_full = [r for r in rows_plot if r["axis"] == "4D"]
        make_plot(rows_full, out_path.with_name(f"{stem}_rms{ext}"),
                  "_rms_err", "RMS rel. error (scaled)",
                  title_suffix=base_title + ", 4D random")
        make_plot(rows_full, out_path.with_name(f"{stem}_max{ext}"),
                  "_max_err", "max rel. error (scaled)",
                  title_suffix=base_title + ", 4D random")
        make_plot(rows_full, out_path.with_name(f"{stem}_rms_true{ext}"),
                  "_rms_rel", "RMS |Δy/y| (floored)",
                  title_suffix=base_title + ", 4D random")
        make_plot(rows_full, out_path.with_name(f"{stem}_max_true{ext}"),
                  "_max_rel", "max |Δy/y| (floored)",
                  title_suffix=base_title + ", 4D random")

        for axis_name in ("L", "VGS", "VDS"):
            rows_ax = [r for r in rows_plot if r["axis"] == axis_name]
            suffix = base_title + f", sweep {axis_name} only"
            make_plot(rows_ax,
                      out_path.with_name(f"{stem}_rms_true_{axis_name}{ext}"),
                      "_rms_rel", f"RMS |Δy/y| (floored, {axis_name} sweep)",
                      title_suffix=suffix)
            make_plot(rows_ax,
                      out_path.with_name(f"{stem}_max_true_{axis_name}{ext}"),
                      "_max_rel", f"max |Δy/y| (floored, {axis_name} sweep)",
                      title_suffix=suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
