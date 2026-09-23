# benchmark_interp.py — behavior & pchip-slowdown audit

Scope: explain what the script does, how it times work, and whether the
observed ~1000× pchip slowdown is real or a measurement artifact.

---

## Layer 1 — What the script does (top-level)

- **Inputs**: every `output/uniform/{device}_{corner}_T{p|m}{T}_uvgs{X}mV_uvds{Y}mV_vsb{N}.mat`
  matching one `(device, corner, temp)` triple. Sorted by `vgs_step`, finest first.
- **Reference**: the finest grid. All other grids report error vs. ref.
- **Sweep axes**:
  1. grid spacing  — discovered files (e.g. 5/10/25/50/100 mV)
  2. method        — `linear`, `pchip`
  3. domain        — `linear`, `log` (asinh transform with per-var scale)
  4. variable      — 14 BSIM4 outputs (ID, VT, GM, …, VDSAT)
- **Query workloads** (3 per `(method, domain, grid)`):
  - 3D random in interior of (VGS, VDS, VSB)
  - VGS sweep at pinned (VDS=0.6, VSB=0)  ← perpendicular axes grid-aligned
  - VDS sweep at pinned (VGS=0.6, VSB=0)
- **Outputs**: a results table + 8 plots
  (`{rms,max}` × {scaled, true, true-VGS, true-VDS}).

---

## Layer 2 — Timing methodology

### 2.1 What `build_s` measures (`main`, lines 638-640)
- A single `make_combined_interp(sl, method, domain)` call.
- That call: optional `arcsinh(vals/scale)`, `np.stack` into a 4-D
  `(nVGS, nVDS, nVSB, n_vars)` array, then
  `RegularGridInterpolator(grid, values, method=method, …)`.

### 2.2 What `time_query` measures (lines 340-349)
```python
y = f(query_pts)              # 1 warmup
for _ in range(repeats):       # repeats default = 3
    t0 = time.perf_counter()
    y = f(query_pts)
    times.append(time.perf_counter() - t0)
return float(np.median(times)), y
```
- `f` is `_CombinedInterp.__call__`: `rgi(pts)` then, in log domain,
  `scales * np.sinh(y)`.
- Median of 3 timed calls, **N = 300** query points by default.
- `qps = n_query / query_s` — divides by wall-time-per-call, not
  per-point CPU time.

### 2.3 Loop topology (lines 634-645)
- For each `(method, domain, slice)` → ONE `make_combined_interp` build.
- Then 3 sweeps reuse that same `interp`; each sweep has its own warmup.
- Same `interp` covers all 14 vars in one call; per-var rows are just
  column slices of the same timed call (lines 647-650).

---

## Layer 3 — Where the 1000× actually comes from (scipy `_rgi.py` internals)

> Audit basis: scipy 1.17.1 `interpolate/_rgi.py` (line refs below) +
> the empirical micro-benchmark in `benchmark_interp_micro.py`
> (run on this box; full table at the end of L3).

### 3.1 How linear handles 8 corners × m queries × 14 vars
- Dispatch (`_rgi.py:448-464`): for `values.ndim == 4` (our case — the
  trailing 14-var axis disqualifies the 2-D Cython fast path), control
  goes to `_evaluate_linear`.
- `_evaluate_linear` (`_rgi.py:520-549`) is an `itertools.product` loop
  over the **8 hypercube corners** of the 3-D cell:
  ```python
  for h in hypercube:                          # 8 iterations
      edge_indices, weights = zip(*h)
      weight = 1
      for w in weights: weight = weight * w    # per-corner scalar weight
      term = self._values[edge_indices] * weight[vslice]
      value = value + term
  ```
- Inside each iteration, `self._values[edge_indices]` is one numpy
  fancy-index returning shape `(m, 14)`, multiplied by a broadcasted
  `(m, 1)` weight. **All m query points and all 14 channels are handled
  in one vectorized numpy op per corner.**
- Net per-call cost: 8 vectorized numpy ops + a handful of axis
  searchsorts. That's it.

### 3.2 How pchip handles the same thing
- Dispatch (`_rgi.py:468-474`): pchip is in `_SPLINE_METHODS_recursive`,
  so calls `_evaluate_spline`.
- `_evaluate_spline` (`_rgi.py:565-618`, key lines 604-616):
  ```python
  for j in range(m):                           # OUTER LOOP: per query pt
      folded_values = first_values[j, ...]
      for i in range(last_dim-1, -1, -1):      # INNER LOOP: per axis
          folded_values = _eval_func(self._grid[i], folded_values,
                                     xi[j, i], k)
      result[j, ...] = folded_values
  ```
- Each `_eval_func` is `_do_pchip` (`_rgi.py:627-630`), which builds a
  **fresh `PchipInterpolator` on every call** and runs
  `_find_derivatives` from scratch on the entire 1-D stencil.
- For 3-D and m=300 that's **600 PchipInterpolator constructions per
  timed call**, none of them cached or reused across calls.

### 3.3 Trailing 14-var axis is NOT the cost driver (was wrong in v1)
- v1 of this analysis said pchip "iterates internally per trailing
  channel". **That was incorrect.** Both linear and pchip vectorize
  across the 14 channels (linear via fancy indexing, pchip via
  `PchipInterpolator(x, y, axis=0)` operating on the full y array).
- Micro-benchmark confirms — pchip wall-time ratio for `n_vars=14`
  vs `n_vars=1` is only **~2.5× at large m**, not 14×. The trailing
  axis adds memory-bandwidth pressure to the per-call slope rebuild,
  not a Python-level loop multiplier.

### 3.4 The dominant cost is the per-query Python loop + per-call slope rebuild
- For each timed call to a pchip RGI:
  - **m × (ndim − 1)** Python-level `PchipInterpolator` constructions
  - Each construction calls `_find_derivatives` which is O(grid_size ·
    n_vars) numpy work — vectorized, but redone from scratch every call
  - No cross-call cache exists (`_rgi.py:627-630` — fresh object each
    invocation)
- For linear: 8 vectorized ops total, no Python per-point loop, no
  per-call rebuild.
- This is a scipy implementation artifact, **not** a property of pchip
  math. A pchip RGI that hoisted slope computation out of the per-call
  path would close most of the gap.

### 3.5 Micro-benchmark numbers (grid 40×40×15, this box, scipy 1.17.1)

```
m       | linear µs/q | pchip µs/q | pchip / linear
------- | ----------- | ---------- | --------------
   10   |     8.700   |  3303.812  |     380×
  100   |     1.096   |  1049.273  |     958×
 1000   |     0.399   |   877.350  |    2199×
10000   |     0.303   |   854.121  |    2819×
```

- **Linear µs/q drops 30× from m=10 → m=10000** as fixed per-call
  overhead amortizes. This is what an amortizing vectorized kernel
  looks like.
- **Pchip µs/q is flat at ~850 µs** for m ≥ 100 — the per-query Python
  loop dominates so the marginal cost per added query barely changes.
  This is what a per-point Python loop looks like.
- **The ratio GROWS with m** because linear amortizes while pchip
  does not. The script's default N=300 lands the ratio in the
  **~1000× band**, exactly matching what is observed in the headline
  benchmark. At larger N the gap would widen, not narrow.

---

## Layer 4 — Script-side amplifiers (after the scipy story)

After accounting for the L3 implementation gap, only a handful of
script-side issues meaningfully affect the headline number. The rest
are reporting nits.

### 4.1 N = 300 is awkward for steady-state throughput
- For linear, N=300 still includes substantial fixed per-call overhead
  (compare 1.10 µs/q at m=100 vs 0.30 µs/q at m=10000 in §3.5).
- For pchip, N has almost no effect on µs/q (already flat at ~850 µs).
- So bumping `--n-query` makes the **linear** number look faster
  (better-amortized), which **widens** the printed pchip/linear ratio
  rather than narrowing it. Don't expect a higher N to "fix" the
  headline; it will make it bigger.
- **Useful fix anyway**: report µs/q (or qps) alongside total wall time
  so a reader can see the amortization shape themselves.

### 4.2 `repeats = 3` is fragile
- Median of 3 is the middle sample — one OS scheduling hiccup becomes
  the reported time.
- pchip has the longer wall window, so it absorbs more jitter.
- **Fix**: `--repeats 7-11`.

### 4.3 Log-domain `np.sinh` is on the critical path (lines 240-244)
- `_CombinedInterp.__call__` always runs `scales * np.sinh(y)` on
  shape `(N, 14)` in log domain.
- Sub-µs per call (numpy ufunc on 4200 floats); negligible vs pchip's
  ms-level cost, and adds <1 µs to linear's per-call time.
- Not an inflation source. Leave as-is.

### 4.4 Per-variable rows reuse a single combined `qps`
- Lines 647-691: each of the 14 var rows reports the **same** `qps`
  (it's the combined call's qps), making it visually look like every
  variable suffers the same per-var slowdown.
- Reporting bug, not measurement bug.
- **Fix**: label the column `qps_call` or split into `qps_call` and
  `qps_per_var = qps_call × n_vars`.

### 4.5 First slice pays cold-import cost
- `import scipy.interpolate` + Cython kernel load happens on the first
  RGI instantiation. The warmup absorbs it, but only for the first
  `(method, domain)` combo. Bounded effect; not the headline driver.
- **Fix**: one-shot dummy `RGI(...)(pts)` per `(method, domain)` before
  the timed loop.

---

## Layer 5 — Recommended changes (ordered by impact)

1. **Report µs/q (or per-query qps) explicitly** so readers can see
   amortization vs flat-per-point cost. The current `qps` column hides
   the shape that §3.5 reveals.
2. **Bump `--repeats 7`** to harden the median against scheduling jitter.
3. **Add a global per-(method, domain) warmup** before the timed loop
   so cold-import is excluded uniformly.
4. **Label `qps` as call-level**, or split into `qps_call` and
   `qps_per_var`, so the 14-row repetition is not misread.
5. **Optional `--profile` flag** that runs `cProfile` on one
   `(method='pchip', domain='linear')` call and dumps the top-20 — this
   would point straight at `_evaluate_spline` / `PchipInterpolator.__init__`
   for anyone wanting to dig deeper.

Note: **bumping N alone will NOT close the gap** — §4.1 — because
linear amortizes faster than pchip with increasing N. The gap is
structural inside scipy.

---

## Layer 6 — One-line verdict (revised)

The 1000× headline is **mostly a scipy implementation artifact**, not a
fundamental property of pchip math. Linear's `_evaluate_linear` is an
8-corner Python loop where each iteration is one vectorized numpy op
over `(m, 14)`; pchip's `_evaluate_spline` is a `for j in range(m)`
Python loop that rebuilds a `PchipInterpolator` (with full slope
recompute) for every query point on every call. The 14-channel trailing
axis is **not** the cost driver — the micro-benchmark shows pchip-14 is
only ~2.5× pchip-1, not 14×. A well-implemented 3-D pchip that hoisted
slope computation out of the per-call path would land closer to ~10–50×
slower than linear, not ~1000×. The script's default N=300 / repeats=3
are minor measurement-hygiene issues; tightening them won't change the
verdict, because the gap lives inside scipy's evaluator, not inside
this benchmark.
