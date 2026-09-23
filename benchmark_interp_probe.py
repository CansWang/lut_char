#!/usr/bin/env python3
"""
benchmark_interp_probe.py — attribute pchip RGI wall time to its hotspots.

Monkey-patches 5 scipy entry points with cumulative wall-time wrappers so
one `rgi(pts)` call's cost can be split across:

    RGI.__call__
      RGI._evaluate_spline
        RGI._do_pchip
          PchipInterpolator.__init__   ← the slope rebuild (hypothesised dominant)
          PchipInterpolator.__call__   ← the actual evaluation

If `PchipInterpolator.__init__` ≥ 70% of `RGI.__call__` cumulative time
and `_do_pchip` is called m × (ndim − 1) × repeats times, the
benchmark_interp_analysis.md diagnosis is confirmed.

Run:  python3 benchmark_interp_probe.py [--m 300]
"""

import argparse
import functools
import time
from collections import defaultdict

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator


# ---------------------------------------------------------------------------
# Probe accumulator + monkey-patches (installed at import time)
# ---------------------------------------------------------------------------

class ProbeStats:
    """Per-name (calls, cumulative ns). reset() between warmup and measure."""
    def __init__(self):
        self.calls = defaultdict(int)
        self.ns = defaultdict(int)

    def reset(self):
        self.calls.clear()
        self.ns.clear()

    def add(self, name: str, dur_ns: int):
        self.calls[name] += 1
        self.ns[name] += dur_ns


PROBE = ProbeStats()


def _wrap_method(cls, attr: str, label: str):
    """Wrap a regular instance/class method with a perf_counter_ns probe."""
    orig = getattr(cls, attr)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            return orig(*args, **kwargs)
        finally:
            PROBE.add(label, time.perf_counter_ns() - t0)

    setattr(cls, attr, wrapper)


def _wrap_staticmethod(cls, attr: str, label: str):
    """Wrap a @staticmethod, re-decorating so the descriptor still works."""
    orig = getattr(cls, attr)  # already unwrapped (staticmethod __get__)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            return orig(*args, **kwargs)
        finally:
            PROBE.add(label, time.perf_counter_ns() - t0)

    setattr(cls, attr, staticmethod(wrapper))


# Install patches — order doesn't matter, they nest at call time.
_wrap_method(RegularGridInterpolator, "__call__",          "RGI.__call__")
_wrap_method(RegularGridInterpolator, "_evaluate_spline",  "RGI._evaluate_spline")
_wrap_staticmethod(RegularGridInterpolator, "_do_pchip",   "RGI._do_pchip")
_wrap_method(PchipInterpolator,       "__init__",          "PchipInterpolator.__init__")
_wrap_method(PchipInterpolator,       "__call__",          "PchipInterpolator.__call__")


# ---------------------------------------------------------------------------
# Run shape
# ---------------------------------------------------------------------------

NVGS, NVDS, NVSB = 40, 40, 15
N_VARS = 14
REPEATS = 7
WARMUP = 2


def make_grid_and_values(n_vars: int = N_VARS, seed: int = 0):
    rng = np.random.default_rng(seed)
    vgs = np.linspace(0.0, 1.5, NVGS)
    vds = np.linspace(0.0, 1.5, NVDS)
    vsb = np.linspace(0.0, 0.6, NVSB)
    values = rng.standard_normal((NVGS, NVDS, NVSB, n_vars))
    return (vgs, vds, vsb), values


def make_query(m: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0.05, 1.45, m),
        rng.uniform(0.05, 1.45, m),
        rng.uniform(0.05, 0.55, m),
    ])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_calls(n: int) -> str:
    return f"{n:>8d}"


def _fmt_ms(ns: int) -> str:
    return f"{ns / 1e6:>9.3f}"


def _fmt_us_per_call(ns: int, calls: int) -> str:
    if calls == 0:
        return "      —  "
    return f"{ns / calls / 1e3:>9.3f}"


def _fmt_pct(ns: int, total_ns: int) -> str:
    if total_ns == 0:
        return "    —  "
    return f"{100.0 * ns / total_ns:>6.2f}%"


def print_breakdown(m: int, ndim: int, repeats: int, linear_wall_ms: float,
                    pchip_total_ms_median: float):
    total_ns = PROBE.ns["RGI.__call__"]
    # _evaluate_spline does ONE vectorized _do_pchip call across all m points
    # on the last axis (rgi.py:596-599), THEN a per-point loop that calls
    # _do_pchip (ndim-1) times per query point (rgi.py:604-616). So the count
    # per rgi(pts) call is m × (ndim-1) + 1, not m × (ndim-1).
    expected_do_pchip = (m * (ndim - 1) + 1) * repeats

    print()
    print(f"=== pchip RGI breakdown "
          f"(grid {NVGS}×{NVDS}×{NVSB}, n_vars={N_VARS}, "
          f"m={m}, repeats={repeats}) ===")
    print(f"Total wall (median across {repeats} repeats): "
          f"{pchip_total_ms_median:.3f} ms")
    print()

    rows = [
        ("RGI.__call__",                  ""),
        ("RGI._evaluate_spline",          "  "),
        ("RGI._do_pchip",                 "    "),
        ("PchipInterpolator.__init__",    "      "),
        ("PchipInterpolator.__call__",    "      "),
    ]

    hdr = (f"{'Layer':<40} | {'calls*':>8} | "
           f"{'ms total':>9} | {'µs/call':>9} | {'% of total':>10}")
    print(hdr)
    print("-" * len(hdr))
    for label, indent in rows:
        calls = PROBE.calls[label]
        ns = PROBE.ns[label]
        print(f"{indent + label:<40} | {_fmt_calls(calls)} | "
              f"{_fmt_ms(ns)} | {_fmt_us_per_call(ns, calls)} | "
              f"{_fmt_pct(ns, total_ns):>10}")

    inner_sum_ns = (PROBE.ns["RGI._do_pchip"])
    residual_ns = max(0, PROBE.ns["RGI._evaluate_spline"] - inner_sum_ns)
    print(f"{'  (residual / outer-loop Python)':<40} | "
          f"{'       —':>8} | {_fmt_ms(residual_ns)} | "
          f"{'      —  ':>9} | {_fmt_pct(residual_ns, total_ns):>10}")

    print()
    print(f"* calls accumulated across all {repeats} repeats.")
    print(f"Expected RGI._do_pchip count: "
          f"(m × (ndim − 1) + 1) × repeats "
          f"= ({m} × {ndim - 1} + 1) × {repeats} = {expected_do_pchip}.")
    print(f"  (+1 is the vectorized first-axis call at _rgi.py:596-599;")
    print(f"   the +600 are the per-point inner-loop calls at _rgi.py:604-616.)")

    print()
    speedup = pchip_total_ms_median / linear_wall_ms if linear_wall_ms > 0 else float("inf")
    print(f"Linear baseline (same grid/n_vars/m): "
          f"{linear_wall_ms:.3f} ms wall  (~{speedup:.0f}× faster).")

    # Assertions
    print()
    actual_do_pchip = PROBE.calls["RGI._do_pchip"]
    actual_init = PROBE.calls["PchipInterpolator.__init__"]
    ok1 = actual_do_pchip == expected_do_pchip
    ok2 = actual_init == actual_do_pchip
    print(f"Assertion 1 — _do_pchip count == (m × (ndim − 1) + 1) × repeats: "
          f"{actual_do_pchip} vs {expected_do_pchip} → "
          f"{'OK' if ok1 else 'MISMATCH'}")
    print(f"Assertion 2 — PchipInterpolator.__init__ count == _do_pchip count: "
          f"{actual_init} vs {actual_do_pchip} → "
          f"{'OK' if ok2 else 'MISMATCH'}")

    # Verdict
    init_pct = (100.0 * PROBE.ns["PchipInterpolator.__init__"] / total_ns
                if total_ns > 0 else 0.0)
    print()
    if init_pct >= 70 and ok1 and ok2:
        print(f"VERDICT: PASS — PchipInterpolator.__init__ is {init_pct:.1f}% "
              f"of RGI.__call__. Slope-rebuild hypothesis CONFIRMED.")
    elif init_pct < 50:
        print(f"VERDICT: FAIL — PchipInterpolator.__init__ is only "
              f"{init_pct:.1f}% of RGI.__call__. Slope rebuild is NOT "
              f"dominant; investigate PchipInterpolator.__call__ and "
              f"_find_derivatives next.")
    else:
        print(f"VERDICT: WEAK — PchipInterpolator.__init__ is {init_pct:.1f}% "
              f"of RGI.__call__ (between 50% and 70%). Diagnosis partially "
              f"supported; the rest is in the per-point evaluator or numpy "
              f"work.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=300,
                    help="Query count per call (default 300, matches "
                         "benchmark_interp.py --n-query default).")
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help=f"Timed repeats (default {REPEATS}).")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help=f"Untimed warmup calls (default {WARMUP}).")
    args = ap.parse_args()

    grid, values = make_grid_and_values()
    pts = make_query(args.m)
    ndim = len(grid)

    # Linear baseline FIRST, then reset PROBE. RGI.__call__ is patched at the
    # class level so linear queries also fire it; if we measured linear after
    # the reset, its calls would pollute the pchip RGI.__call__ count.
    rgi_lin = RegularGridInterpolator(
        grid, values, method="linear", bounds_error=False, fill_value=np.nan,
    )
    for _ in range(args.warmup):
        rgi_lin(pts)
    lin_walls = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        rgi_lin(pts)
        lin_walls.append((time.perf_counter() - t0) * 1e3)
    linear_median_ms = float(np.median(lin_walls))

    # Build pchip RGI. PchipInterpolator is NOT constructed by RGI's __init__
    # for pchip — slopes are computed lazily inside _do_pchip during query.
    rgi_pchip = RegularGridInterpolator(
        grid, values, method="pchip", bounds_error=False, fill_value=np.nan,
    )

    # Pchip warmup (cold-import, JIT, etc.) then reset stats.
    for _ in range(args.warmup):
        rgi_pchip(pts)
    PROBE.reset()

    # Timed pchip loop — also record total wall per call for median.
    walls_ms = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        rgi_pchip(pts)
        walls_ms.append((time.perf_counter() - t0) * 1e3)
    pchip_median_ms = float(np.median(walls_ms))

    print_breakdown(
        m=args.m, ndim=ndim, repeats=args.repeats,
        linear_wall_ms=linear_median_ms,
        pchip_total_ms_median=pchip_median_ms,
    )


if __name__ == "__main__":
    main()
