#!/usr/bin/env python3
"""
fast_pchip.py — drop-in replacement for scipy's RGI pchip evaluator.

scipy's `RegularGridInterpolator._do_pchip` (`_rgi.py:626-630`) builds a
full `PchipInterpolator` on every call, which does two expensive things
even when only a single cell is needed:

  1. Computes Hermite slopes on the FULL y array (via `_find_derivatives`).
  2. Wraps everything in a PchipInterpolator → CubicHermiteSpline → PPoly
     object hierarchy (~300 µs of pure Python overhead per call).

For the RGI inner loop (`_evaluate_spline` lines 604-616), `pt` is a
scalar and only one cell's worth of slopes matters. So this module:

  - For SCALAR pt: locates the cell, computes slopes locally from a
    3- or 4-point stencil around that cell. O(trail) per call instead
    of O(n_axis · trail).
  - For ARRAY pt (the vectorized first-axis call at `_rgi.py:596`):
    use scipy's `_find_derivatives` once on the full y (unavoidable —
    different queries hit different cells), then evaluate Hermite cubic
    directly with raw numpy — skipping the PPoly wrapper.

Slope numerics match scipy's `_find_derivatives` formula exactly
(Carlson-Fritsch weighted harmonic mean + Moler's edge case).

Usage:
    import fast_pchip
    fast_pchip.enable()
    rgi = RegularGridInterpolator(grid, values, method="pchip")
    rgi(pts)
    fast_pchip.disable()
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate._cubic import PchipInterpolator


# ---------------------------------------------------------------------------
# Slope kernels — broadcast over trailing dims of y
# ---------------------------------------------------------------------------

def _interior_slope(delta_back, delta_fwd, h_back, h_fwd):
    """
    Carlson-Fritsch pchip slope at an interior grid point.
        delta_back, delta_fwd : forward differences (broadcastable)
        h_back, h_fwd         : grid spacings (scalars)
    Returns: slope, same shape as deltas.
    """
    w1 = 2.0 * h_fwd + h_back
    w2 = h_fwd + 2.0 * h_back
    same_sign = (np.sign(delta_back) * np.sign(delta_fwd)) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        whmean = (w1 / delta_back + w2 / delta_fwd) / (w1 + w2)
    return np.where(same_sign, 1.0 / whmean, 0.0)


def _edge_slope(h0, h1, m0, m1):
    """
    Moler's pchip edge-case slope (matches scipy `_cubic.py:_edge_case`).
        h0, h1 : adjacent cell widths (scalars), h0 is the cell at the edge.
        m0, m1 : forward differences in those cells (broadcastable).
    Returns: slope, same shape as m0.
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    sign_d = np.sign(d)
    sign_m0 = np.sign(m0)
    sign_m1 = np.sign(m1)
    mask_zero = sign_d != sign_m0
    mask_3x = (sign_m0 != sign_m1) & (np.abs(d) > 3.0 * np.abs(m0)) & ~mask_zero
    d = np.where(mask_zero, 0.0, d)
    d = np.where(mask_3x, 3.0 * m0, d)
    return d


def _hermite_eval(y0, y1, d0, d1, h, t):
    """Direct Hermite cubic. All args broadcastable; h, t can be scalars."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1


# ---------------------------------------------------------------------------
# Scalar-pt path: local 3- or 4-point stencil. O(trail) per call.
# ---------------------------------------------------------------------------

def _do_pchip_scalar(x, y, pt):
    n = x.shape[0]
    if n == 2:
        # Degenerate: linear between the only 2 points.
        h = x[1] - x[0]
        t = (pt - x[0]) / h
        return y[0] + t * (y[1] - y[0])

    pt_f = float(pt)
    k = int(np.searchsorted(x, pt_f)) - 1
    if k < 0:
        k = 0
    elif k > n - 2:
        k = n - 2

    h_c = x[k + 1] - x[k]
    y_k = y[k]
    y_kp1 = y[k + 1]
    delta_c = (y_kp1 - y_k) / h_c

    # Slope at k (left endpoint of the active cell).
    if k == 0:
        h_n = x[2] - x[1]
        delta_n = (y[2] - y[1]) / h_n
        d_k = _edge_slope(h_c, h_n, delta_c, delta_n)
    else:
        h_p = x[k] - x[k - 1]
        delta_p = (y_k - y[k - 1]) / h_p
        d_k = _interior_slope(delta_p, delta_c, h_p, h_c)

    # Slope at k+1 (right endpoint).
    if k == n - 2:
        h_p = x[k] - x[k - 1]
        delta_p = (y_k - y[k - 1]) / h_p
        d_k1 = _edge_slope(h_c, h_p, delta_c, delta_p)
    else:
        h_n = x[k + 2] - x[k + 1]
        delta_n = (y[k + 2] - y_kp1) / h_n
        d_k1 = _interior_slope(delta_c, delta_n, h_c, h_n)

    t = (pt_f - x[k]) / h_c
    return _hermite_eval(y_k, y_kp1, d_k, d_k1, h_c, t)


# ---------------------------------------------------------------------------
# Dispatch + install/restore (called by RGI._evaluate_spline)
# ---------------------------------------------------------------------------

# Capture scipy's original _do_pchip ONCE at import time. We use it as the
# vectorized fallback: scipy's PPoly evaluation is a C-level routine that
# beats a naive numpy reimplementation on the large (m, *trail) tensors
# produced by the first-axis call at _rgi.py:596. The big win for fast_pchip
# is on the INNER loop (scalar pt, called m × (ndim-1) times), where the
# local-stencil scalar path avoids both the per-call slope rebuild on the
# full y and the PPoly construction overhead.
_SCIPY_DO_PCHIP = RegularGridInterpolator._do_pchip


def _do_pchip_fast(x, y, pt, k):
    """k (spline degree) is ignored — pchip is always cubic."""
    if np.ndim(pt) == 0:
        return _do_pchip_scalar(x, y, pt)
    return _SCIPY_DO_PCHIP(x, y, pt, k)


_original_do_pchip = None  # cached BARE function (staticmethod unwrapped on getattr)


def enable() -> None:
    """Install the fast _do_pchip. Idempotent."""
    global _original_do_pchip
    if _original_do_pchip is None:
        _original_do_pchip = RegularGridInterpolator._do_pchip
        RegularGridInterpolator._do_pchip = staticmethod(_do_pchip_fast)


def disable() -> None:
    """Restore scipy's original _do_pchip. Re-wraps in staticmethod so that
    later `self._do_pchip(...)` calls do NOT auto-bind `self` as a 5th arg."""
    global _original_do_pchip
    if _original_do_pchip is not None:
        RegularGridInterpolator._do_pchip = staticmethod(_original_do_pchip)
        _original_do_pchip = None


def is_enabled() -> bool:
    return _original_do_pchip is not None


# ---------------------------------------------------------------------------
# Self-test: compare fast vs scipy on a 3-D grid with trailing axis.
# ---------------------------------------------------------------------------

def _self_test(tol: float = 1e-10) -> None:
    rng = np.random.default_rng(0)
    # Cover boundary cells too (first/last interval) with this query range.
    vgs = np.linspace(0.0, 1.5, 40)
    vds = np.linspace(0.0, 1.5, 40)
    vsb = np.linspace(0.0, 0.6, 15)
    grid = (vgs, vds, vsb)
    values = rng.standard_normal((40, 40, 15, 14))
    pts = np.column_stack([
        rng.uniform(0.0001, 1.4999, 1000),
        rng.uniform(0.0001, 1.4999, 1000),
        rng.uniform(0.0001, 0.5999, 1000),
    ])

    disable()
    rgi_scipy = RegularGridInterpolator(grid, values, method="pchip",
                                        bounds_error=False, fill_value=np.nan)
    y_scipy = rgi_scipy(pts)

    enable()
    rgi_fast = RegularGridInterpolator(grid, values, method="pchip",
                                       bounds_error=False, fill_value=np.nan)
    y_fast = rgi_fast(pts)
    disable()

    if y_scipy.shape != y_fast.shape:
        raise AssertionError(f"shape mismatch: {y_scipy.shape} vs {y_fast.shape}")
    diff = np.max(np.abs(y_scipy - y_fast))
    rel = diff / np.max(np.abs(y_scipy))
    print(f"fast_pchip self-test: max abs diff = {diff:.3e}, "
          f"max rel diff = {rel:.3e} (n={pts.shape[0]}, n_vars=14)")
    if diff > tol:
        raise AssertionError(
            f"fast vs scipy disagree by {diff:.3e} > tol {tol:.0e}"
        )
    print("OK — fast pchip matches scipy within tolerance.")


if __name__ == "__main__":
    _self_test()
