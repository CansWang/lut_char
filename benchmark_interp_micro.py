#!/usr/bin/env python3
"""
benchmark_interp_micro.py — confirm scipy RGI linear-vs-pchip cost model.

Two hypotheses:
    H1: trailing 14-channel axis is NOT a pchip-specific penalty.
        Probe: vary n_vars in {1, 14}; if it were the cost driver,
        pchip wall time would scale ~14x with n_vars.
    H2: pchip's per-query Python loop in scipy.interpolate._rgi.py
        (_evaluate_spline at lines 565-618) is the dominant cost.
        Probe: vary m (query count) in {10, 100, 1000, 10000};
        if H2 holds, pchip per-query us is roughly flat across m,
        while linear's per-query us drops as fixed overhead amortizes.

Grid is synthetic (random values, realistic shape). Times in milliseconds
and microseconds per query. Repeats=7, median reported.
"""

import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPEATS = 7
WARMUP = 2
NVGS, NVDS, NVSB = 40, 40, 15
M_GRID = [10, 100, 1000, 10000]
NVARS_GRID = [1, 14]
METHODS = ["linear", "pchip"]


def make_grid_and_values(n_vars: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    vgs = np.linspace(0.0, 1.5, NVGS)
    vds = np.linspace(0.0, 1.5, NVDS)
    vsb = np.linspace(0.0, 0.6, NVSB)
    values = rng.standard_normal((NVGS, NVDS, NVSB, n_vars))
    return (vgs, vds, vsb), values


def make_query(m: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(0.05, 1.45, m),
        rng.uniform(0.05, 1.45, m),
        rng.uniform(0.05, 0.55, m),
    ])
    return pts


def time_call(f, pts, repeats: int = REPEATS, warmup: int = WARMUP):
    for _ in range(warmup):
        f(pts)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f(pts)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    print(f"scipy RGI micro-benchmark — grid ({NVGS}, {NVDS}, {NVSB}), "
          f"repeats={REPEATS}, warmup={WARMUP}")
    print()

    rows = []
    for n_vars in NVARS_GRID:
        grid, values = make_grid_and_values(n_vars)
        for method in METHODS:
            rgi = RegularGridInterpolator(grid, values, method=method,
                                          bounds_error=False, fill_value=np.nan)
            for m in M_GRID:
                pts = make_query(m)
                wall_s = time_call(rgi, pts)
                rows.append({
                    "method":   method,
                    "n_vars":   n_vars,
                    "m":        m,
                    "wall_ms":  wall_s * 1e3,
                    "us_per_q": wall_s * 1e6 / m,
                })

    cols = ["method", "n_vars", "m", "wall_ms", "us_per_q"]
    widths = {c: max(len(c), 8) for c in cols}
    hdr = " | ".join(c.ljust(widths[c]) for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = " | ".join([
            r["method"].ljust(widths["method"]),
            str(r["n_vars"]).ljust(widths["n_vars"]),
            str(r["m"]).ljust(widths["m"]),
            f"{r['wall_ms']:8.3f}",
            f"{r['us_per_q']:8.3f}",
        ])
        print(line)

    print()
    print("=== H1: pchip wall scaling with n_vars (1 -> 14) ===")
    print(f"{'m':>6} | {'linear ratio':>14} | {'pchip ratio':>14}")
    for m in M_GRID:
        def get(method, n_vars):
            return next(r["wall_ms"] for r in rows
                        if r["method"] == method and r["m"] == m
                        and r["n_vars"] == n_vars)
        lin_r = get("linear", 14) / get("linear", 1)
        pch_r = get("pchip",  14) / get("pchip",  1)
        print(f"{m:>6} | {lin_r:>14.2f} | {pch_r:>14.2f}")
    print("(If H1 holds, pchip ratio << 14; trailing axis is vectorized.)")

    print()
    print("=== H2: per-query us across m (n_vars=14) ===")
    print(f"{'m':>6} | {'linear us/q':>14} | {'pchip us/q':>14} | "
          f"{'pchip/linear':>14}")
    for m in M_GRID:
        def get(method):
            return next(r["us_per_q"] for r in rows
                        if r["method"] == method and r["m"] == m
                        and r["n_vars"] == 14)
        lin = get("linear")
        pch = get("pchip")
        print(f"{m:>6} | {lin:>14.3f} | {pch:>14.3f} | {pch/lin:>14.1f}")
    print("(If H2 holds, pchip us/q is ~flat across m; "
          "linear us/q drops as m grows.)")


if __name__ == "__main__":
    main()
