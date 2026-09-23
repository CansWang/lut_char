#!/usr/bin/env python3
"""
verify_pchip_numba.py — sanity-check that the Numba 4-D PCHIP kernel agrees
with scipy across a range of point categories and grid fixtures.

Categories tested per fixture:
  - Hypercube corners (2**ndim = 16 extreme points)
  - Random exact grid samples (where interp must return stored y bit-exact)
  - Random points inside boundary cells (first/last cell of each axis)
  - Random interior points (away from any boundary)

Fixtures:
  - Minimum-size (n=4 per axis: every cell is a boundary cell)
  - Realistic device-LUT shape (non-uniform L axis, smaller voltage axes)
  - Real LUT loaded from output/uniform/ if present

Pass condition: max abs diff vs scipy is ≤ 1e-10 for all categories on all
fixtures. Exact grid samples additionally tracked against a tighter 1e-12
threshold since interp at a sample should be a no-op.

Usage:
    python3 verify_pchip_numba.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import pchip_numba


TOL_DEFAULT = 1e-10
TOL_GRID_SAMPLE = 1e-12  # tighter — both impls should reproduce the stored y


# ---------------------------------------------------------------------------
# Point generators
# ---------------------------------------------------------------------------

def gen_hypercube_corners(grid):
    """All 2**ndim combinations of (axis_min, axis_max). 16 points for 4-D."""
    ndim = len(grid)
    out = np.empty((2 ** ndim, ndim), dtype=float)
    for i, combo in enumerate(itertools.product(
            *[(float(ax.min()), float(ax.max())) for ax in grid])):
        out[i] = combo
    return out


def gen_random_grid_samples(grid, n, seed):
    """N points sitting exactly on grid coordinates (random index per axis).
    Both scipy and numba must return the stored y bit-exactly here."""
    rng = np.random.default_rng(seed)
    ndim = len(grid)
    out = np.empty((n, ndim), dtype=float)
    for i, ax in enumerate(grid):
        idx = rng.integers(0, len(ax), size=n)
        out[:, i] = ax[idx]
    return out


def gen_random_interior(grid, n, seed):
    """Random uniform within (axis_min + 1%span, axis_max - 1%span)."""
    rng = np.random.default_rng(seed)
    ndim = len(grid)
    out = np.empty((n, ndim), dtype=float)
    for i, ax in enumerate(grid):
        a, b = float(ax.min()), float(ax.max())
        margin = 0.01 * (b - a)
        out[:, i] = rng.uniform(a + margin, b - margin, n)
    return out


def gen_boundary_cell(grid, n_per_axis, seed):
    """For each axis: n_per_axis points in the first cell + n_per_axis in
    the last cell, with the other 3 axes at random interior positions.
    Stresses the at_left / at_right code paths of the slope helper."""
    rng = np.random.default_rng(seed)
    ndim = len(grid)
    rows_per_axis = 2 * n_per_axis
    total = rows_per_axis * ndim
    out = np.empty((total, ndim), dtype=float)
    # Fill all rows with random interior positions first.
    for i, ax in enumerate(grid):
        a, b = float(ax.min()), float(ax.max())
        margin = 0.01 * (b - a)
        out[:, i] = rng.uniform(a + margin, b - margin, total)
    # Override one axis at a time with first-cell / last-cell positions.
    cursor = 0
    for axis in range(ndim):
        ax = grid[axis]
        # First cell: random in (ax[0], ax[1]).
        eps = 1e-6 * (ax[1] - ax[0])
        out[cursor:cursor + n_per_axis, axis] = rng.uniform(
            ax[0] + eps, ax[1] - eps, n_per_axis)
        cursor += n_per_axis
        # Last cell: random in (ax[-2], ax[-1]).
        eps = 1e-6 * (ax[-1] - ax[-2])
        out[cursor:cursor + n_per_axis, axis] = rng.uniform(
            ax[-2] + eps, ax[-1] - eps, n_per_axis)
        cursor += n_per_axis
    return out


# ---------------------------------------------------------------------------
# Compare scipy vs numba on one fixture
# ---------------------------------------------------------------------------

def _build_and_query(grid, values, pts):
    """Build both interpolators, query both at the same points, return both
    output arrays. Numba is enabled/disabled around the build so each RGI is
    independent of the other's state."""
    pchip_numba.disable()
    rgi_s = RegularGridInterpolator(grid, values, method='pchip',
                                    bounds_error=False, fill_value=np.nan)
    y_s = rgi_s(pts)

    pchip_numba.enable()
    rgi_n = RegularGridInterpolator(grid, values, method='pchip',
                                    bounds_error=False, fill_value=np.nan)
    y_n = rgi_n(pts)
    pchip_numba.disable()
    return y_s, y_n


def _diff_stats(y_s, y_n):
    """Return (max_abs, max_rel, n_kept) computed over finite entries."""
    mask = np.isfinite(y_s) & np.isfinite(y_n)
    n_kept = int(mask.sum())
    if n_kept == 0:
        return np.nan, np.nan, 0
    diff = np.abs(y_s[mask] - y_n[mask])
    abs_max = float(diff.max())
    denom = float(np.max(np.abs(y_s[mask])))
    if denom == 0.0:
        rel_max = float("inf") if abs_max > 0 else 0.0
    else:
        rel_max = abs_max / denom
    return abs_max, rel_max, n_kept


def run_fixture(name, grid, values, n_per_axis_boundary=50,
                n_random_interior=1000, n_grid_samples=500, seed=0):
    """Run all 4 categories on one fixture. Returns True iff every category
    passes its tolerance."""
    print(f"\nFixture: {name}")
    print(f"  axes lengths: {[len(a) for a in grid]}    "
          f"values shape: {values.shape}")

    cases = [
        ("Hypercube corners",      gen_hypercube_corners(grid), TOL_DEFAULT),
        ("Random grid samples",    gen_random_grid_samples(grid, n_grid_samples,
                                                           seed + 1),
                                   TOL_GRID_SAMPLE),
        ("Random interior",        gen_random_interior(grid, n_random_interior,
                                                       seed + 2),
                                   TOL_DEFAULT),
        ("Boundary cells",         gen_boundary_cell(grid, n_per_axis_boundary,
                                                     seed + 3),
                                   TOL_DEFAULT),
    ]

    rows = []
    for label, pts, tol in cases:
        y_s, y_n = _build_and_query(grid, values, pts)
        abs_max, rel_max, n_kept = _diff_stats(y_s, y_n)
        ok = np.isfinite(abs_max) and abs_max <= tol
        rows.append((label, len(pts), n_kept, abs_max, rel_max, tol, ok))

    width = max(len(r[0]) for r in rows)
    print(f"  {'category':<{width}}   {'n':>5}  {'kept':>5}   "
          f"{'max abs diff':>13}   {'max rel diff':>13}   {'tol':>9}   verdict")
    print(f"  {'-' * (width + 70)}")
    for label, n, n_kept, abs_max, rel_max, tol, ok in rows:
        verdict = "PASS" if ok else "*** FAIL ***"
        print(f"  {label:<{width}}   {n:>5d}  {n_kept:>5d}   "
              f"{abs_max:>13.3e}   {rel_max:>13.3e}   {tol:>9.0e}   {verdict}")

    return all(r[6] for r in rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build_synthetic_fixtures():
    rng = np.random.default_rng(42)
    fixtures = []

    # F1: minimum-size grid (n=4 per axis). Two interior cells per axis,
    # all boundary cells exercise the at_left/at_right paths often.
    grid = (np.array([0.0, 0.5, 1.5, 3.0]),       # non-uniform
            np.array([-1.0, -0.2, 0.3, 1.0]),
            np.linspace(0.0, 2.0, 4),               # uniform
            np.array([10.0, 11.0, 14.0, 20.0]))
    values = rng.standard_normal((4, 4, 4, 4, 7))
    fixtures.append(("minimum 4×4×4×4 (every cell touches a boundary)",
                     grid, values))

    # F2: realistic device-LUT shape (non-uniform L like sg13 cells, smaller
    # voltage axes so the test runs fast).
    grid = (np.array([0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.30, 0.50, 1.00, 3.00]),
            np.linspace(0.0, 1.5, 16),
            np.linspace(0.0, 1.5, 16),
            np.linspace(0.0, 0.6, 15))
    values = rng.standard_normal((10, 16, 16, 15, 14))
    fixtures.append(("realistic non-uniform L (10×16×16×15, n_vars=14)",
                     grid, values))

    # F3: skewed shape (large L, tiny voltage axes) — stresses boundary on
    # voltage axes since most of the LUT is L-direction.
    grid = (np.linspace(0.1, 5.0, 20),
            np.linspace(0.0, 1.0, 5),
            np.linspace(0.0, 1.0, 5),
            np.linspace(0.0, 0.5, 4))
    values = rng.standard_normal((20, 5, 5, 4, 3))
    fixtures.append(("skewed 20×5×5×4 (small voltage axes)",
                     grid, values))

    return fixtures


def maybe_load_real_lut():
    """Load the smallest pmos LUT for a real-data sanity check, if present."""
    candidates = [
        "output/uniform/sg13_lv_pmos_TT_Tp27_uvgs100mV_uvds100mV_vsb15.mat",
        "output/uniform/sg13_lv_nmos_TT_Tp27_uvgs50mV_uvds50mV_vsb15.mat",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                import benchmark_interp_4d as b4
            except ImportError:
                return None
            table = b4.load_mat_table(p)
            grid = (table["axes"]["L"], table["axes"]["VGS"],
                    table["axes"]["VDS"], table["axes"]["VSB"])
            var_names = [v for v in b4.DATA_KEYS if v in table["data"]]
            values = np.stack([table["data"][v] for v in var_names], axis=-1)
            return (f"real LUT: {p.name}", grid, values)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("scipy vs numba 4-D PCHIP sanity check")
    print("=" * 75)

    # Trigger JIT compile once up front so the per-fixture wall time is honest.
    print("\nWarming up Numba JIT...")
    pchip_numba._self_test(tol=1e-10)

    fixtures = build_synthetic_fixtures()
    real = maybe_load_real_lut()
    if real is not None:
        fixtures.append(real)
    else:
        print("\n[note] no real LUT found under output/uniform/; "
              "synthetic fixtures only")

    all_ok = True
    for name, grid, values in fixtures:
        all_ok &= run_fixture(name, grid, values)

    print()
    print("=" * 75)
    if all_ok:
        print("ALL CATEGORIES PASSED across all fixtures.")
        return 0
    print("*** SOME CATEGORIES FAILED ***")
    return 1


if __name__ == "__main__":
    sys.exit(main())
