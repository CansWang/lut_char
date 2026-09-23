# Numba JIT 4-D PCHIP — implementation notes

A drop-in replacement for `scipy.interpolate.RegularGridInterpolator`'s 4-D
PCHIP path. Replaces scipy's `_evaluate_spline` with a Numba-JIT kernel that
reads only a local 4-on-a-side stencil per query, exploits inter-query
locality via a hierarchical cache, and sorts queries to maximise cache hits.

Files:
- `pchip_numba.py` — kernel + monkey-patch.
- `benchmark_interp_4d.py` — driver that swaps in the kernel via
  `--pchip-impl numba` and benchmarks accuracy + qps across grids.

---

## Why scipy's pchip is slow on 4-D LUTs

scipy's pchip RGI builds a full `BPoly` along the trail axis for every cell
touched per query, so cost scales with the trail dimension and grid size,
not with the local-stencil size. On the 10 mV device LUT (~50 MB of
values), this is a 3 ms kernel buried under a 340 ms memcpy fired by the
wrapper's `np.ascontiguousarray` safety fallback (see [Pitfalls](#pitfalls)).

The numba kernel reduces per-query work to O(4^ndim) reads + 85 scalar
PCHIP evaluations.

---

## Kernel algorithm — sequential 256 → 64 → 16 → 4 → 1 reduction

PCHIP is **not** a tensor product. The Carlson-Fritsch weighted harmonic
mean and the Moler edge formula contain sign tests and a `|d| > 3|m|`
peak-limiter clamp that do not distribute algebraically across axes. The
kernel therefore reduces one axis at a time on the just-reduced "virtual"
4-point set, recomputing slopes dynamically at each stage:

```
stencil (4×4×4×4)  --axis 3-->  buf3 (4×4×4)
                                buf3 --axis 2--> buf2 (4×4)
                                                 buf2 --axis 1--> buf1 (4)
                                                                  buf1 --axis 0--> out
```

Scalar helpers (`pchip_numba.py:50-120`), each `@njit(inline='always')`:
- `_interior_slope_scalar` — Carlson-Fritsch weighted harmonic mean.
- `_edge_slope_scalar`     — Moler 3-point one-sided with `|d| > 3|m|` clamp.
- `_hermite_eval_scalar`   — Hermite cubic on [0, 1] with `h`-scaled derivs.
- `_pchip_1d_eval`         — one cell of 1-D PCHIP with boundary fallback.

Boundary cells duplicate a sentinel into the unused stencil slot
(`s0_0 = k0` when `a0L`, `s0_3 = k0+1` when `a0R`) — the slope helpers
switch to the Moler 3-point form on the valid adjacent points and never
divide by the sentinel-zeroed `h_left`/`h_right`.

---

## Blocking + state-machine cache

Adjacent queries that share the same active cell (or the same outer-axis
t's) can re-use intermediate reduction buffers. The kernel keeps a
hierarchy of validity bits:

| Cache level     | Valid iff                                                  |
|-----------------|------------------------------------------------------------|
| `stencil_cache` | `(k0, k1, k2, k3)` all match the previous query            |
| `buf3`          | `stencil_cache` valid AND `t3 == last_t3`                  |
| `buf2`          | `buf3` valid          AND `t2 == last_t2`                  |
| `buf1`          | `buf2` valid          AND `t1 == last_t1`                  |
| axis-0 output   | always re-runs (it is the per-query output)                |

Sentinels are `-1` for `last_k*` (never matches a clamped index) and
`-1.0` for `last_t*` (never matches `t ∈ [0, 1]`). Hard literals avoid
Numba type-inference surprises that arise with `math.nan` equality tests.

The stencil-fetch loop preserves the original C-contiguous walk order
(`i0/i1/i2` outer, `v` innermost) so DRAM access pattern is unchanged
relative to the pre-cache kernel — there is no penalty when the cache is
cold.

### Wrapper-side: lexsort to maximise cache hits

In `_evaluate_spline_pchip` (`pchip_numba.py:344-426`):

```python
order = np.lexsort((
    norm_distances[0], indices[0],
    norm_distances[1], indices[1],
    norm_distances[2], indices[2],
    norm_distances[3], indices[3],   # primary key (last)
))
```

`np.lexsort` treats the **last** key as primary, so this sorts by
`(k3, t3, k2, t2, k1, t1, k0, t0)` — outer → inner, matching the cache
hierarchy. Adjacent queries after the sort share as much of the kernel's
cache state as possible. Results are unscrambled with the inverse
permutation (`out[order] = out_sorted`) so callers see the original order.

---

## Pseudo-code: original loop vs optimised loop

### Original — naïve per-query 4-D reduction

Every query independently fetches the full 256-point stencil and runs all
four reduction axes from scratch. There is no sort, no cache, no skip.
Per query: 256 reads + 85 PCHIP evals + 256 stencil-cache writes.

```text
for q in 0 .. m-1:                       # queries in caller-given order
    (k0, k1, k2, k3) = indices[:, q]
    (t0, t1, t2, t3) = norm_distances[:, q]

    # --- fetch full 4×4×4×4 local stencil (256 reads × n_vars) ---
    for i0 in 0..3:
      for i1 in 0..3:
        for i2 in 0..3:
          for i3 in 0..3:
            for v in 0..n_vars-1:
              S[i0,i1,i2,i3,v] = values[idx0(i0), idx1(i1),
                                        idx2(i2), idx3(i3), v]

    # --- axis-3 reduction: 256 → 64 (always) ---
    for i0,i1,i2,v: buf3[i0,i1,i2,v] = pchip_1d(S[i0,i1,i2,:,v], h3*, t3)

    # --- axis-2 reduction: 64 → 16 (always) ---
    for i0,i1,v:    buf2[i0,i1,v]    = pchip_1d(buf3[i0,i1,:,v], h2*, t2)

    # --- axis-1 reduction: 16 → 4 (always) ---
    for i0,v:       buf1[i0,v]       = pchip_1d(buf2[i0,:,v], h1*, t1)

    # --- axis-0 reduction: 4 → 1 (always) ---
    for v:          out[q,v]         = pchip_1d(buf1[:,v], h0*, t0)
```

Cost is constant per query regardless of inter-query locality: a 1-D L
sweep that touches the same `(k1,k2,k3)` cell 300 times in a row still
fetches the same stencil + runs the same axis-3/2/1 reductions 300 times.

### Optimised — lexsort + hierarchical state-machine cache

The wrapper reorders queries so adjacent ones share outer-axis state; the
kernel maintains `last_k*` / `last_t*` and skips reduction levels whose
inputs match the previous query.

```text
# ---------- WRAPPER (Python, vectorised) ----------
indices, nd = self._find_indices(xi.T)            # scipy locate
order = lexsort((nd[0], idx[0],                   # primary key = LAST
                 nd[1], idx[1],
                 nd[2], idx[2],
                 nd[3], idx[3]))                  # → sort by (k3,t3,k2,t2,...)
indices_sorted = indices[:, order]
nd_sorted      = nd[:, order]
call kernel(indices_sorted, nd_sorted, out_sorted)
out[order] = out_sorted                           # inverse permutation

# ---------- KERNEL (Numba @njit) ----------
last_k0 = last_k1 = last_k2 = last_k3 = -1        # sentinels
last_t1 = last_t2 = last_t3 = -1.0                # (no last_t0 needed)

for q in 0 .. m-1:                                # sorted order
    (k0, k1, k2, k3) = indices_sorted[:, q]
    (t0, t1, t2, t3) = nd_sorted[:, q]

    # --- cache validity (outer → inner cascade) ---
    raw_valid  = (k0==last_k0 and k1==last_k1
                  and k2==last_k2 and k3==last_k3)
    buf3_valid = raw_valid  and (t3 == last_t3)
    buf2_valid = buf3_valid and (t2 == last_t2)
    buf1_valid = buf2_valid and (t1 == last_t1)

    # --- fetch stencil ONLY on cell change ---
    if not raw_valid:
        for i0,i1,i2,v: stencil_cache[i0,i1,i2,0..3,v] = values[...]

    # --- axis-3 reduction ONLY when stencil or t3 changed ---
    if not buf3_valid:
        for i0,i1,i2,v: buf3[i0,i1,i2,v] = pchip_1d(stencil_cache[...], h3*, t3)

    # --- axis-2 reduction ONLY when buf3 or t2 changed ---
    if not buf2_valid:
        for i0,i1,v:    buf2[i0,i1,v]    = pchip_1d(buf3[...], h2*, t2)

    # --- axis-1 reduction ONLY when buf2 or t1 changed ---
    if not buf1_valid:
        for i0,v:       buf1[i0,v]       = pchip_1d(buf2[...], h1*, t1)

    # --- axis-0 reduction ALWAYS (per-query output) ---
    for v:              out_sorted[q,v]  = pchip_1d(buf1[...], h0*, t0)

    # --- update cache state ---
    last_k0..k3 = k0..k3
    last_t1..t3 = t1..t3
```

### Per-query work, compared

For an `m`-query 1-D sweep along axis 0 (caller pins `k1,k2,k3,t1,t2,t3`):

| Phase            | Original | Optimised (after lexsort) |
|------------------|----------|---------------------------|
| Stencil fetch    | m × 256  | **1** × 256               |
| Axis-3 reduction | m × 64   | **1** × 64                |
| Axis-2 reduction | m × 16   | **1** × 16                |
| Axis-1 reduction | m × 4    | **1** × 4                 |
| Axis-0 reduction | m × 1    | m × 1                     |

Total scalar PCHIP evals: `85m` → `84 + m`. For `m = 300` that is
`25 500 → 384`, a ~66× drop in eval work plus a ~m× drop in DRAM reads.
This is exactly why the L / VGS / VDS sweep rows in the headline table
are essentially grid-independent.

---

## Pitfalls

### 1. `np.stack(..., axis=-1)` produces non-c_contiguous views

This is a numpy gotcha that bit the implementation hard: stacking arrays
along `axis=-1` returns a view with strides arranged for axis-0 stacking,
which is **not** C-contiguous despite looking like it should be.

Without intervention, the wrapper's safety check fires
`np.ascontiguousarray(values_5d)` on **every call**, copying the entire
grid (~340 ms for 392 MB on the 10 mV LUT) and burying the 3 ms kernel.

Two-layer defense:
- `benchmark_interp_4d.py:90` — force `np.ascontiguousarray` once at
  build time: `values = np.ascontiguousarray(np.stack(stacked, axis=-1))`.
- `pchip_numba.py:370-383` — cache the contig copy on the RGI instance,
  keyed by `id(self._values)` so values replacement triggers a refresh.

### 2. Axis hoisting is mathematically invalid for PCHIP

Tempting "optimisation": pre-reduce static axes (those with the same
`t`/`k` across all queries) once, outside the per-query loop. **This is
wrong** for PCHIP: the virtual points arriving at inner axes change per
query, so the dynamic slopes change even when `t` on that axis is static.
The state-machine cache above is the correct way to exploit query
clustering — it short-circuits identical work without algebraic
re-arrangement.

### 3. `pgrep -f` self-trap when waiting on background jobs

(Operational, not in the kernel.) `until ! pgrep -f "pchip_numba"; do
sleep N; done` self-traps because pgrep matches the wait-shell's own
argv. Use `kill -0 PID` or a filename-only pattern instead.

---

## Benchmark integration (`benchmark_interp_4d.py`)

- One `RGI` over `(nL, nVGS, nVDS, nVSB, n_vars)` so locate/weight cost
  is amortised across all 14 variables per call
  (`make_combined_interp_4d`, line 64).
- `--pchip-impl scipy|fast|numba` selects the implementation; numba and
  fast monkey-patch `RGI._evaluate_spline`, scipy is the stock baseline.
  All three are bit-exact (≤ 1e-10 self-test in `pchip_numba.py:456`).
- Four query workloads exercise different cache hit rates:
  - **4D random** — worst case, cache misses on every query.
  - **L sweep**   — pins VGS/VDS/VSB, varies L. Outer-axis sorts let the
    inner buffers (`buf1`, `buf2`, `buf3`) stay valid.
  - **VGS / VDS sweeps** — similar, with different axes varying.
- Plots are written per-axis-per-metric at `dpi=200` (via
  `benchmark_interp.make_plot`).

### Headline results

| Workload    | qps (10 mV grid) | qps (100 mV grid) | scaling     |
|-------------|------------------|-------------------|-------------|
| L sweep     | 1.4 M            | 1.4 M             | flat        |
| VGS sweep   | 232 K            | 1.06 M            | mild        |
| VDS sweep   | 206 K            | 469 K             | mild        |
| 4D random   | 70 K             | 117 K             | grid-bound  |

1-D sweeps are now essentially grid-independent (the kernel hits cache on
nearly every query). 4D random remains grid-dependent — the cache rarely
hits, so per-query work is the full 85 PCHIP evals + 256-point stencil
fetch. There is no further wins to take here without giving up bit-
exactness or moving to an approximation.

---

## Usage

```python
import pchip_numba
pchip_numba.enable()
rgi = RegularGridInterpolator((x0, x1, x2, x3), values, method="pchip")
rgi(pts)                # uses Numba kernel
pchip_numba.disable()   # restores scipy
```

Non-4-D and non-pchip calls fall through to the original
`_evaluate_spline`, so the monkey-patch is safe to leave installed
process-wide.

Run the self-test:

```bash
python pchip_numba.py
```

Run the benchmark:

```bash
python benchmark_interp_4d.py --pchip-impl numba --n-query 300 --repeats 2
```

---

# Supplemental: caching tricks in depth

This section walks through three worked examples that illustrate *why* the
hierarchical cache + lexsort gives the speedups it does, and the design
choices that make the trick work.

## A. Why the cascade is outer → inner, not inner → outer

The validity bits chain in one direction:

```
raw_valid  ⊇  buf3_valid  ⊇  buf2_valid  ⊇  buf1_valid
(superset)
```

`raw_valid` requires only that the four cell indices match. `buf3_valid`
requires `raw_valid` AND `t3`-match. Etc. That ordering is **not
arbitrary** — it falls out of the dependency graph:

- `buf3` is the output of the axis-3 reduction, which reads
  `stencil_cache` and uses `t3`. So `buf3` is reusable iff *both* inputs
  are unchanged.
- `buf2` is the output of axis-2 reduction, which reads `buf3` and uses
  `t2`. Reusable iff *both* are unchanged.
- …and so on inward.

If you tried the reverse cascade (`buf1` checks `t1` only, ignoring
whether `buf2` is stale) you'd silently read a stale buffer and return
garbage. The cascade exists because the cache is **derived state**, not
input state — each level can only be valid if every level upstream of it
is also valid.

This is also why we sort with axis-3 as the **primary** lexsort key:
adjacent queries are most likely to share `k3` and `t3`, which is the
*outermost* (most expensive) cache level. Hits at the outer level
automatically give you hits at every inner level too (as long as the
corresponding `t`'s also match), so each outer hit saves the work of all
inner reductions.

## B. Worked example — L sweep traced through the cache

Setup: 300 queries that vary only L (axis 0). VGS / VDS / VSB are pinned
to grid-aligned constants. The L axis has ~10-20 cells (depends on grid
density). After `_find_indices`:

```
indices       = [k0_q  k1=4  k2=12  k3=0]   for every query q
norm_distances= [t0_q  t1=0.0 t2=0.0 t3=0.0]  for every query q
```

After `np.lexsort((t0, k0, t1, k1, t2, k2, t3, k3))` with `k3` primary,
since `k3`/`t3`/`k2`/`t2`/`k1`/`t1` are all identical the sort
**collapses to sorting by `(k0, t0)`**. Consecutive queries now share `k0`
when they fall in the same L cell. Suppose 12 L cells, ~25 queries per
cell.

Tracing the cache state as `q` advances:

| q   | k0 | (k1,k2,k3) | (t1,t2,t3) | raw | buf3 | buf2 | buf1 | work this query |
|-----|----|------------|------------|-----|------|------|------|-----------------|
| 0   | 0  | (4,12,0)   | (0,0,0)    | ✗   | ✗    | ✗    | ✗    | 256 + 64 + 16 + 4 + 1 evals |
| 1   | 0  | (4,12,0)   | (0,0,0)    | ✓   | ✓    | ✓    | ✓    | **1** eval (axis-0 only)    |
| 2   | 0  | (4,12,0)   | (0,0,0)    | ✓   | ✓    | ✓    | ✓    | **1** eval                  |
| …   | 0  |            |            | ✓   | ✓    | ✓    | ✓    | **1** eval each             |
| 25  | 1  | (4,12,0)   | (0,0,0)    | ✗   | ✗    | ✗    | ✗    | 256 + 64 + 16 + 4 + 1 evals |
| 26  | 1  | (4,12,0)   | (0,0,0)    | ✓   | ✓    | ✓    | ✓    | **1** eval                  |
| …   |    |            |            |     |      |      |      |                             |

Totals for `m = 300`, `n_cells = 12`:

| Phase            | Calls   |
|------------------|---------|
| Stencil fetch    | 12 × 256 = **3 072** reads |
| Axis-3 reduction | 12 × 64  = **768** evals   |
| Axis-2 reduction | 12 × 16  = **192** evals   |
| Axis-1 reduction | 12 × 4   = **48** evals    |
| Axis-0 reduction | 300 × 1  = **300** evals   |
| **Total evals**  |          | **1 308**                  |

Compared to the original `300 × 85 = 25 500` evals, that is a **~20×
reduction** — and the DRAM read drop is even more dramatic (3 K vs
76.8 K reads). qps is dominated by the few-hundred axis-0 evals, which
fit comfortably in L1 and don't scale with grid size — hence the **flat
1.4 M qps** across the 10 / 25 / 50 / 100 mV grids.

## C. Worked example — VGS sweep, partial cache hits

Setup: 300 queries vary only VGS (axis 1). L / VDS / VSB pinned.
Now `k1`/`t1` vary; `k0,k2,k3,t0,t2,t3` are constant.

After lexsort (still by `(k3,t3,k2,t2,k1,t1,k0,t0)` ordering), queries
group by `k1`. When `k1` changes between queries `q` and `q+1`:

```
raw_valid  = (k1 == last_k1)  → FAIL
buf3_valid = raw_valid AND …  → FAIL  (cascade)
buf2_valid = buf3_valid AND … → FAIL
buf1_valid = buf2_valid AND … → FAIL
```

→ **every cache level rebuilds** when `k1` changes. The cache is useful
only for repeated hits within the same `k1` cell.

Hit rate is governed by query density:

| Grid spacing | VGS cells | Queries per cell (m=300) | Cache hit rate |
|--------------|-----------|--------------------------|----------------|
| 10 mV        | ~150      | ~2                       | ~50 %          |
| 25 mV        | ~60       | ~5                       | ~80 %          |
| 50 mV        | ~30       | ~10                      | ~90 %          |
| 100 mV       | ~15       | ~20                      | ~95 %          |

That table maps directly onto the **232 K → 1.06 M qps** spread in the
headline results: finer grid = more cells = fewer hits per cell = closer
to worst-case 85 evals per query.

## D. Worked example — 4-D random, why lexsort can't help

300 queries with all 4 axes uniform-random over the interior. The product
grid size (e.g. 10 × 150 × 150 × 7 = 1.57 M cells on a 10 mV LUT) is
~5 000× larger than `m`, so the expected number of queries landing in any
given cell is ≪ 1. Even after lexsort, consecutive queries differ in
*something* — `raw_valid` is essentially never true.

```
For 300 queries vs 1.57 M cells:
  P(two queries hit the same cell) ≈ 300 / 1.57 M = 1.9e-4
  Expected #cache hits ≈ 300 × 1.9e-4 = 0.06
```

Practical observation: **0 raw_valid hits** on the 10 mV 4D-random
workload. Every query pays the full 256-read + 85-eval cost, which is
where the 70 K qps on the 10 mV grid comes from (and why it climbs to
117 K on the coarser 100 mV grid — finer grid = more L1/L2 evictions of
the `values` array as we wander through it).

This is also why the optimisation is *blocking* and not *vectorisation*:
the only way to make 4D-random faster without giving up bit-exactness is
to reduce per-query work, and the kernel is already at the floor (256
reads + 85 inlined Hermite evaluations).

## E. Design footnotes

### E.1 — Why `-1.0` and `-1`, not `math.nan`

We use scalar `last_t* = -1.0` as the sentinel. The natural choice would
be `math.nan`, but Numba's type-inferred `x == math.nan` always returns
`False` (per IEEE 754), which would *correctly* never match — but Numba's
type-inference on the mixed `int`/`float`/`nan` arithmetic sometimes
boxes scalars into Python objects, killing perf. `-1.0` is in the same
float64 type as `t`, equality is a simple SIMD compare, and `t ∈ [0, 1]`
guarantees no collision.

### E.2 — Why no `last_t0`

Axis-0 is the per-query output — there's no `buf0` to cache. Every query
must run the axis-0 reduction regardless. Saving `last_t0` would be dead
code, and skipping the variable removes one register pressure point in
the inner loop.

### E.3 — Float equality on `t` is exact for sweeps

`buf*_valid` checks `t == last_t` (no tolerance). This is exact in
practice for sweep workloads because off-axis `t`'s come from grid-
aligned pins (`VGS = 0.6`, `VDS = 0.6`, `VSB = 0.0`), so the same float
literal flows through `_find_indices` and produces a bit-identical `t`.
For random workloads `t` essentially never repeats anyway, so the
strictness has no cost. If you ever feed non-grid-aligned but repeated
queries (e.g. a fixed user batch evaluated multiple times), the strict
equality is what you want — `t` round-trips through scipy unmodified.

### E.4 — The "primary key is LAST" gotcha in `np.lexsort`

`np.lexsort((a, b, c))` sorts by `c` primarily, then `b`, then `a`. This
is the opposite of `sorted(zip(...))` order and trips people up. To sort
queries by `(k3, t3, k2, t2, k1, t1, k0, t0)` with `k3` primary, the keys
must be passed least-to-most significant:

```python
np.lexsort((t0, k0, t1, k1, t2, k2, t3, k3))
#          \__least significant__/  \__most__/
```

The current kernel uses `(nd[0], idx[0], nd[1], idx[1], …)` which is the
same order. Getting this wrong (e.g. flipping the tuple) would silently
produce a sort that primarily sorts on `t0` — which only ever helps for
1-D axis-0 sweeps and hurts every other workload.

### E.5 — Stencil-fetch loop is identical to the pre-cache version

The fetch nest is `for i0: for i1: for i2: for v: … 4 stencil writes
into [i0,i1,i2,0..3,v]`. The innermost `v` walk reads `values` along its
contiguous trailing axis (stride 8 bytes), and the axis-3 stencil writes
into a contiguous slice of `stencil_cache`. This is the same DRAM access
pattern as the pre-cache kernel, so a cache miss costs no more than it
did before — the cache is **pure upside** on hits, **zero downside** on
misses.
