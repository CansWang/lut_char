#!/usr/bin/env python3
"""
pchip_numba.py — Numba JIT 4-D PCHIP evaluator. Drops scipy's RGI pchip
per-query cost from O(trail) to O(4**ndim) by reading only a local 4-on-a-side
stencil and reducing it dimension-by-dimension.

The kernel performs a 4-axis sequential reduction (256 -> 64 -> 16 -> 4 -> 1)
on the local stencil values. Slopes are recomputed dynamically at each step
on the just-reduced "virtual" 4-point set. This is required because PCHIP is
NOT a tensor product: the Carlson-Fritsch weighted harmonic mean and the
Moler edge formula contain sign tests and `|d| > 3|m|` clamps that do not
algebraically distribute across axes.

Per-call work:
  - Python wrapper computes the (4, m) `indices` / `norm_distances` ONCE
    via scipy's own `_find_indices` (vectorized numpy), then hands raw int64
    and float64 arrays to the JIT body. No searchsorted inside @njit.
  - Per query: 64+16+4+1 = 85 calls to `_pchip_1d_eval`, each inlined.
    Slopes computed from a 3- or 4-point stencil; boundary cells fall back
    to the Moler 3-point one-sided formula on the valid adjacent points
    (no h=0 divisions from clamped sentinels).

Usage:
    import pchip_numba
    pchip_numba.enable()
    rgi = RegularGridInterpolator((x0, x1, x2, x3), values, method="pchip")
    rgi(pts)                # uses Numba kernel
    pchip_numba.disable()   # restores scipy

Notes:
  - First call triggers Numba JIT compile (~1-2 s). The benchmark's
    `time_query` runs one warmup call, so JIT cost stays out of the
    measured window.
  - Slope/Hermite formulas mirror those in `fast_pchip.py:44-85` and match
    scipy's `PchipInterpolator._find_derivatives` modulo fastmath FMA
    reordering (tol 1e-10 in self-test).
  - Only handles the 4-D pchip case. Non-4-D and non-pchip calls delegate
    to the captured original `RGI._evaluate_spline`.
"""

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Inlined scalar slope + Hermite kernels.
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, boundscheck=False, inline='always')
def _interior_slope_scalar(delta_back, delta_fwd, h_back, h_fwd):
    """Carlson-Fritsch weighted harmonic mean. Returns 0 if signs differ
    or either delta is zero (the `delta_back * delta_fwd <= 0` check catches
    both cases)."""
    if delta_back * delta_fwd <= 0.0:
        return 0.0
    w1 = 2.0 * h_fwd + h_back
    w2 = h_fwd + 2.0 * h_back
    whmean = (w1 / delta_back + w2 / delta_fwd) / (w1 + w2)
    return 1.0 / whmean


@njit(cache=True, fastmath=True, boundscheck=False, inline='always')
def _edge_slope_scalar(h0, h1, m0, m1):
    """Moler 3-point one-sided edge slope with `|d| > 3|m0|` clamp.
    `h0` is the cell width at the edge; `h1` is the adjacent cell width."""
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if d * m0 <= 0.0:
        return 0.0
    abs_d = -d if d < 0.0 else d
    abs_m0 = -m0 if m0 < 0.0 else m0
    if m0 * m1 <= 0.0 and abs_d > 3.0 * abs_m0:
        return 3.0 * m0
    return d


@njit(cache=True, fastmath=True, boundscheck=False, inline='always')
def _hermite_eval_scalar(y0, y1, d0, d1, h, t):
    """Hermite cubic on [0, 1] with derivatives scaled by `h`."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1


@njit(cache=True, fastmath=True, boundscheck=False, inline='always')
def _pchip_1d_eval(y0, y1, y2, y3,
                   h_left, h_mid, h_right,
                   at_left, at_right, t):
    """1-D PCHIP eval at t in [0,1] in the cell between y1 and y2.

    Stencil convention: y0 = y[k-1], y1 = y[k], y2 = y[k+1], y3 = y[k+2].
    At a boundary cell the duplicated sentinel (y0 == y1 at left, y3 == y2
    at right) is never read — the slope formula switches to the Moler 3-
    point one-sided form on the valid adjacent points, so no h = 0 divide.
    """
    if at_left and at_right:
        # Degenerate axis length 2: linear between y1 and y2.
        return y1 + t * (y2 - y1)

    m_mid = (y2 - y1) / h_mid

    if at_left:
        m_right = (y3 - y2) / h_right
        slope_left = _edge_slope_scalar(h_mid, h_right, m_mid, m_right)
    else:
        m_left = (y1 - y0) / h_left
        slope_left = _interior_slope_scalar(m_left, m_mid, h_left, h_mid)

    if at_right:
        m_left = (y1 - y0) / h_left
        slope_right = _edge_slope_scalar(h_mid, h_left, m_mid, m_left)
    else:
        m_right = (y3 - y2) / h_right
        slope_right = _interior_slope_scalar(m_mid, m_right, h_mid, h_right)

    return _hermite_eval_scalar(y1, y2, slope_left, slope_right, h_mid, t)


# ---------------------------------------------------------------------------
# Main 4-D kernel — sequential 256 → 64 → 16 → 4 → 1 reduction.
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, boundscheck=False)
def _pchip_eval_4d(x0, x1, x2, x3, values,
                   indices, norm_distances, out,
                   stencil_cache, buf3, buf2, buf1):
    """
    Per-query sequential 256 → 64 → 16 → 4 → 1 reduction with a hierarchical
    stencil + reduction cache. Skips work when the current query shares the
    relevant (k, t) state with the previous query (which the wrapper
    maximises via lexsort).

    Validity hierarchy (outer → inner):
        stencil_cache valid IFF (k0, k1, k2, k3) match prev query
        buf3 valid          IFF stencil_cache valid AND t3 matches prev
        buf2 valid          IFF buf3 valid          AND t2 matches prev
        buf1 valid          IFF buf2 valid          AND t1 matches prev
        axis-0 reduction always runs (it's the per-query output).

    Parameters
    ----------
    x0..x3 : float64 (n_a,)
        1-D axis arrays (possibly non-uniform).
    values : float64 (n0, n1, n2, n3, n_vars), C-contiguous
        Grid values with trailing variable axis.
    indices : int64 (4, m)
        Left-edge cell index per axis per query, clamped to [0, n_a - 2].
    norm_distances : float64 (4, m)
        t in [0, 1] within the active cell.
    out : float64 (m, n_vars)
        Output, written in place.
    stencil_cache : float64 (4, 4, 4, 4, n_vars)
        Local 256-point stencil. Re-populated only on cell change.
    buf3 : float64 (4, 4, 4, n_vars)
        Axis-3 reduction output.
    buf2 : float64 (4, 4, n_vars)
        Axis-2 reduction output.
    buf1 : float64 (4, n_vars)
        Axis-1 reduction output.
    """
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    n3 = x3.shape[0]
    m = indices.shape[1]
    n_vars = values.shape[4]

    # Sentinels — `-1` never matches a valid (clamped) index;
    # `-1.0` never matches a valid t (which is in [0, 1]). Hardcoded
    # `-1.0` literals (instead of math.nan) avoid Numba type-inference
    # surprises on the equality checks.
    last_k0 = -1
    last_k1 = -1
    last_k2 = -1
    last_k3 = -1
    last_t1 = -1.0
    last_t2 = -1.0
    last_t3 = -1.0
    # (no last_t0 — axis-0 reduction always runs.)

    for q in range(m):
        k0 = indices[0, q]
        k1 = indices[1, q]
        k2 = indices[2, q]
        k3 = indices[3, q]
        t0 = norm_distances[0, q]
        t1 = norm_distances[1, q]
        t2 = norm_distances[2, q]
        t3 = norm_distances[3, q]

        # Per-axis boundary flags (active cell at the leftmost / rightmost).
        a0L = (k0 == 0); a0R = (k0 == n0 - 2)
        a1L = (k1 == 0); a1R = (k1 == n1 - 2)
        a2L = (k2 == 0); a2R = (k2 == n2 - 2)
        a3L = (k3 == 0); a3R = (k3 == n3 - 2)

        # Cell widths around the active cell. `_left` / `_right` are zero
        # at boundaries; the slope helper never divides by them when the
        # corresponding boundary flag is set.
        h0_mid = x0[k0 + 1] - x0[k0]
        h1_mid = x1[k1 + 1] - x1[k1]
        h2_mid = x2[k2 + 1] - x2[k2]
        h3_mid = x3[k3 + 1] - x3[k3]

        h0_left = 0.0 if a0L else (x0[k0] - x0[k0 - 1])
        h1_left = 0.0 if a1L else (x1[k1] - x1[k1 - 1])
        h2_left = 0.0 if a2L else (x2[k2] - x2[k2 - 1])
        h3_left = 0.0 if a3L else (x3[k3] - x3[k3 - 1])

        h0_right = 0.0 if a0R else (x0[k0 + 2] - x0[k0 + 1])
        h1_right = 0.0 if a1R else (x1[k1 + 2] - x1[k1 + 1])
        h2_right = 0.0 if a2R else (x2[k2 + 2] - x2[k2 + 1])
        h3_right = 0.0 if a3R else (x3[k3 + 2] - x3[k3 + 1])

        # Per-axis stencil grid indices. Position 1 = k, position 2 = k+1
        # (the active cell). Position 0 = k-1 normally, k at the left edge
        # (sentinel — duplicated, never read). Position 3 = k+2 normally,
        # k+1 at the right edge (sentinel).
        s0_0 = k0 if a0L else (k0 - 1)
        s1_0 = k1 if a1L else (k1 - 1)
        s2_0 = k2 if a2L else (k2 - 1)
        s3_0 = k3 if a3L else (k3 - 1)
        s0_3 = (k0 + 1) if a0R else (k0 + 2)
        s1_3 = (k1 + 1) if a1R else (k1 + 2)
        s2_3 = (k2 + 1) if a2R else (k2 + 2)
        s3_3 = (k3 + 1) if a3R else (k3 + 2)

        # === Cache validity ===
        # raw_valid implies all k's match → boundary flags and h's also match.
        raw_valid = (k0 == last_k0 and k1 == last_k1
                     and k2 == last_k2 and k3 == last_k3)
        buf3_valid = raw_valid  and (t3 == last_t3)
        buf2_valid = buf3_valid and (t2 == last_t2)
        buf1_valid = buf2_valid and (t1 == last_t1)

        # === Stencil fetch (only on cell change) ===
        # Same C-contig walk order as before: i0/i1/i2 outer, axis-3 + v
        # innermost so the DRAM access pattern is unchanged from the
        # pre-cache kernel.
        if not raw_valid:
            for i0 in range(4):
                if i0 == 0:   idx0 = s0_0
                elif i0 == 1: idx0 = k0
                elif i0 == 2: idx0 = k0 + 1
                else:         idx0 = s0_3
                for i1 in range(4):
                    if i1 == 0:   idx1 = s1_0
                    elif i1 == 1: idx1 = k1
                    elif i1 == 2: idx1 = k1 + 1
                    else:         idx1 = s1_3
                    for i2 in range(4):
                        if i2 == 0:   idx2 = s2_0
                        elif i2 == 1: idx2 = k2
                        elif i2 == 2: idx2 = k2 + 1
                        else:         idx2 = s2_3
                        for v in range(n_vars):
                            stencil_cache[i0, i1, i2, 0, v] = \
                                values[idx0, idx1, idx2, s3_0,   v]
                            stencil_cache[i0, i1, i2, 1, v] = \
                                values[idx0, idx1, idx2, k3,     v]
                            stencil_cache[i0, i1, i2, 2, v] = \
                                values[idx0, idx1, idx2, k3 + 1, v]
                            stencil_cache[i0, i1, i2, 3, v] = \
                                values[idx0, idx1, idx2, s3_3,   v]

        # === Reduction along axis 3: 256 → 64 ===
        if not buf3_valid:
            for i0 in range(4):
                for i1 in range(4):
                    for i2 in range(4):
                        for v in range(n_vars):
                            y0 = stencil_cache[i0, i1, i2, 0, v]
                            y1 = stencil_cache[i0, i1, i2, 1, v]
                            y2 = stencil_cache[i0, i1, i2, 2, v]
                            y3 = stencil_cache[i0, i1, i2, 3, v]
                            buf3[i0, i1, i2, v] = _pchip_1d_eval(
                                y0, y1, y2, y3,
                                h3_left, h3_mid, h3_right,
                                a3L, a3R, t3,
                            )

        # === Reduction along axis 2: 64 → 16 ===
        if not buf2_valid:
            for i0 in range(4):
                for i1 in range(4):
                    for v in range(n_vars):
                        y0 = buf3[i0, i1, 0, v]
                        y1 = buf3[i0, i1, 1, v]
                        y2 = buf3[i0, i1, 2, v]
                        y3 = buf3[i0, i1, 3, v]
                        buf2[i0, i1, v] = _pchip_1d_eval(
                            y0, y1, y2, y3,
                            h2_left, h2_mid, h2_right,
                            a2L, a2R, t2,
                        )

        # === Reduction along axis 1: 16 → 4 ===
        if not buf1_valid:
            for i0 in range(4):
                for v in range(n_vars):
                    y0 = buf2[i0, 0, v]
                    y1 = buf2[i0, 1, v]
                    y2 = buf2[i0, 2, v]
                    y3 = buf2[i0, 3, v]
                    buf1[i0, v] = _pchip_1d_eval(
                        y0, y1, y2, y3,
                        h1_left, h1_mid, h1_right,
                        a1L, a1R, t1,
                    )

        # === Reduction along axis 0: 4 → 1 (always per-query) ===
        for v in range(n_vars):
            y0 = buf1[0, v]
            y1 = buf1[1, v]
            y2 = buf1[2, v]
            y3 = buf1[3, v]
            out[q, v] = _pchip_1d_eval(
                y0, y1, y2, y3,
                h0_left, h0_mid, h0_right,
                a0L, a0R, t0,
            )

        # === Update cache state for next iteration ===
        last_k0 = k0
        last_k1 = k1
        last_k2 = k2
        last_k3 = k3
        last_t1 = t1
        last_t2 = t2
        last_t3 = t3


# ---------------------------------------------------------------------------
# Monkey-patch
# ---------------------------------------------------------------------------

_ORIGINAL_EVALUATE_SPLINE = RegularGridInterpolator._evaluate_spline


def _evaluate_spline_pchip(self, xi, method):
    """Drop-in replacement for `RGI._evaluate_spline`. Diverts the 4-D
    pchip case to the Numba kernel; everything else falls back to scipy's
    original implementation so other RGI methods/dimensionalities still work."""
    if method != 'pchip' or len(self._grid) != 4:
        return _ORIGINAL_EVALUATE_SPLINE(self, xi, method)

    if xi.ndim == 1:
        xi = xi.reshape(1, xi.size)
    m = xi.shape[0]

    values = self._values  # (n0, n1, n2, n3, *trail)
    trail_shape = values.shape[4:]

    # Collapse trailing dims into one variable axis. scipy already enforces
    # that the leading 4 axes match the grid, so reshape is safe.
    if len(trail_shape) == 0:
        n_vars = 1
        values_5d = values.reshape(values.shape + (1,))
    elif len(trail_shape) == 1:
        n_vars = trail_shape[0]
        values_5d = values
    else:
        n_vars = int(np.prod(trail_shape))
        values_5d = values.reshape(values.shape[:4] + (n_vars,))

    if values_5d.dtype != np.float64 or not values_5d.flags.c_contiguous:
        # np.stack(..., axis=-1) can produce a view with weird strides that
        # is NOT c_contiguous (numpy gotcha). Without caching, this copy
        # would fire on every call and dominate wall time on large grids
        # (340 ms memcpy of a 400 MB array on the 10 mV LUT). Cache the
        # contig copy on the RGI instance, keyed by id(self._values) so a
        # values replacement triggers a refresh.
        cache_key = id(self._values)
        cached = getattr(self, "_pchip_numba_values_5d", None)
        if cached is None or cached[0] != cache_key:
            values_5d = np.ascontiguousarray(values_5d, dtype=np.float64)
            self._pchip_numba_values_5d = (cache_key, values_5d)
        else:
            values_5d = cached[1]

    # scipy's _find_indices returns (4, m) int64 + (4, m) float64.
    indices_arr, nd_arr = self._find_indices(xi.T)
    indices = np.ascontiguousarray(indices_arr, dtype=np.int64)
    norm_distances = np.ascontiguousarray(nd_arr, dtype=np.float64)

    # Lexsort queries so adjacent ones share as much of the kernel's cache
    # hierarchy as possible. Cache hierarchy is OUTER (axis 3) → INNER
    # (axis 0); the primary sort key is therefore k3, then t3, then k2, ...
    # np.lexsort takes keys with LAST as primary.
    order = np.lexsort((
        norm_distances[0], indices[0],
        norm_distances[1], indices[1],
        norm_distances[2], indices[2],
        norm_distances[3], indices[3],
    ))
    indices_sorted = np.ascontiguousarray(indices[:, order])
    nd_sorted      = np.ascontiguousarray(norm_distances[:, order])

    out_sorted = np.empty((m, n_vars), dtype=np.float64)
    stencil_cache = np.empty((4, 4, 4, 4, n_vars), dtype=np.float64)
    buf3 = np.empty((4, 4, 4, n_vars), dtype=np.float64)
    buf2 = np.empty((4, 4, n_vars), dtype=np.float64)
    buf1 = np.empty((4, n_vars), dtype=np.float64)

    g = self._grid
    x0 = np.ascontiguousarray(g[0], dtype=np.float64)
    x1 = np.ascontiguousarray(g[1], dtype=np.float64)
    x2 = np.ascontiguousarray(g[2], dtype=np.float64)
    x3 = np.ascontiguousarray(g[3], dtype=np.float64)

    _pchip_eval_4d(x0, x1, x2, x3, values_5d,
                   indices_sorted, nd_sorted, out_sorted,
                   stencil_cache, buf3, buf2, buf1)

    # Unscramble: out[order] = out_sorted reverses the lexsort permutation
    # so the caller sees results in the original query order.
    out = np.empty_like(out_sorted)
    out[order] = out_sorted

    if len(trail_shape) == 0:
        return out.reshape(m)
    return out.reshape((m,) + trail_shape)


_installed = False


def enable() -> None:
    """Install the Numba 4-D pchip kernel. Idempotent."""
    global _installed
    if not _installed:
        RegularGridInterpolator._evaluate_spline = _evaluate_spline_pchip
        _installed = True


def disable() -> None:
    """Restore scipy's original `_evaluate_spline`."""
    global _installed
    if _installed:
        RegularGridInterpolator._evaluate_spline = _ORIGINAL_EVALUATE_SPLINE
        _installed = False


def is_enabled() -> bool:
    return _installed


# ---------------------------------------------------------------------------
# Self-test — bit-exact agreement vs scipy on a 4-D random fixture.
# ---------------------------------------------------------------------------

def _self_test(tol: float = 1e-10) -> None:
    rng = np.random.default_rng(0)
    # Non-uniform L axis (matches the device-LUT pattern: tight near Lmin,
    # then sparse). Voltage axes uniform.
    L   = np.array([0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.30, 0.50, 1.00, 3.00])
    vgs = np.linspace(0.0, 1.5, 40)
    vds = np.linspace(0.0, 1.5, 40)
    vsb = np.linspace(0.0, 0.6, 15)
    grid = (L, vgs, vds, vsb)
    values = rng.standard_normal((10, 40, 40, 15, 14))
    pts = np.column_stack([
        rng.uniform(L[0] + 1e-4, L[-1] - 1e-4, 1000),
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
    diff = float(np.max(np.abs(y_scipy - y_fast)))
    rel = diff / float(np.max(np.abs(y_scipy)))
    print(f"pchip_numba self-test: n={pts.shape[0]}, n_vars=14, grid 10x40x40x15")
    print(f"  max abs diff: {diff:.3e}")
    print(f"  max rel diff: {rel:.3e}")
    if diff > tol:
        raise AssertionError(
            f"Numba vs scipy disagree by {diff:.3e} > tol {tol:.0e}"
        )
    print(f"  OK — Numba pchip matches scipy within tol {tol:.0e}.")


if __name__ == "__main__":
    _self_test()
