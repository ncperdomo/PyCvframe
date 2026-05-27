# PyCvframe Performance Optimization — Design Spec

**Date:** 2026-05-27  
**Scope:** Vectorize `transform_array`, add pytest suite, add benchmark script  
**Constraint:** Preserve all numerical results and public API exactly

---

## Problem

`PyCvframe.transform_array` uses a Python `for` loop that calls `apply_frame_rotation`
once per site. Each call:

- Computes `sin/cos(lat)` and `sin/cos(lon)` **twice** — once in `geod_to_xyz`, once
  in `rotation_matrix_neu` — four redundant trig operations per site.
- Invokes `np.cross` on a 3-element array, paying Python function-call overhead for
  a 6-operation computation.
- Builds a full 3×3 rotation matrix even though only rows 0 (North) and 1 (East) are
  ever used.
- Pays Python interpreter overhead once per site for the loop iteration itself.

At 429 sites the wall-clock cost is small but the pattern scales poorly and leaves
measurable overhead that vectorization eliminates entirely.

---

## What changes

| File | Change |
|---|---|
| `pycvframe.py` | Add private `_batch_transform`; replace loop in `transform_array` with call to it |
| `tests/test_pycvframe.py` | New — pytest suite (5 test functions) |
| `benchmarks/benchmark_transform.py` | New — before/after timing script |

### What does NOT change

- `apply_frame_rotation`, `geod_to_xyz`, `rotation_matrix_neu`, `cross_product` — scalar helpers kept intact for documentation fidelity and Fortran-mapping reference
- `_process`, `run`, `main` — file I/O path unchanged
- `frame_registry.py` — static lookup, no performance issue
- All public API signatures

---

## Core optimization: `_batch_transform`

A private module-level function. Accepts NumPy arrays of shape `(N,)` for lons, lats,
ve, vn and the scalar rotation vector `rot_prime`. Returns `(ve_new, vn_new)` arrays.

```python
def _batch_transform(lons, lats, ve, vn, rot_prime):
    lat = np.radians(lats)
    lon = np.radians(lons)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)

    # geod_to_xyz — vectorized
    Nr = EARTH_RAD / np.sqrt(1.0 - EARTH_E2 * slat**2)
    px = Nr * clat * clon
    py = Nr * clat * slon
    pz = Nr * (1.0 - EARTH_E2) * slat

    # omega × pos — inline (avoids np.cross per-call overhead)
    ox, oy, oz = rot_prime
    cvx = oy*pz - oz*py
    cvy = oz*px - ox*pz
    cvz = ox*py - oy*px

    # Only North (R row 0) and East (R row 1) needed — no full matrix built
    # R[0] = [-slat*clon, -slat*slon,  clat]
    # R[1] = [-slon,       clon,       0   ]
    neu_N = -slat*clon*cvx - slat*slon*cvy + clat*cvz
    neu_E = -slon*cvx      + clon*cvy

    return ve - neu_E * 1000.0, vn - neu_N * 1000.0
```

`transform_array` delegates to it:

```python
def transform_array(self, lons, lats, ve, vn):
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    ve   = np.asarray(ve,   dtype=float)
    vn   = np.asarray(vn,   dtype=float)
    return _batch_transform(lons, lats, ve, vn, self.rot_prime)
```

---

## Why each optimization improves performance

| Optimization | Mechanism |
|---|---|
| Single `np.radians` call over array | One ufunc dispatch instead of N Python calls |
| Shared `sin/cos` arrays | Trig computed once; used for both position and rotation |
| Inline cross product | 6 NumPy scalar/broadcast ops; no Python function-call overhead |
| Skip rows 2 of R (Up component) | 3 fewer multiply-add operations per site, times N |
| NumPy broadcasting throughout | All arithmetic is contiguous-memory C operations, no Python loop |

---

## Test suite: `tests/test_pycvframe.py`

Five test functions using pytest:

1. **`test_reference_file_equivalence`** — Run `PyCvframe.run()` on
   `data/reilinger_2006_igb14.vel` with the Reilinger 2006 pole
   `[0.32840, -0.03504, 0.40682]` deg/Myr. Parse the output and the reference file
   `reference/reilinger_2006_arab.vel`. Assert max |Ve residual| ≤ 0.01 mm/yr and
   max |Vn residual| ≤ 0.01 mm/yr. Assert Vu unchanged.

2. **`test_transform_array_matches_scalar`** — Load all 429 sites. Run
   `transform_array` (vectorized) and compare against a reference computed with the
   original scalar loop using `apply_frame_rotation`. Assert `np.allclose` with
   `atol=1e-10`.

3. **`test_zero_pole_is_noop`** — With `rot_prime = [0, 0, 0]` (rad/yr), confirm
   output equals input exactly for all 429 sites.

4. **`test_single_site_regression`** — Hard-code the first site from the reference
   file (HERS_GPS, lon=0.336, lat=50.867) and its expected transformed velocities.
   Assert to 0.01 mm/yr tolerance.

5. **`test_frame_registry`** — `frame_to_frame("ITRF14", "ITRF14")` returns zeros;
   `frame_to_frame("EURA_I14", "ITRF14")` returns a non-zero 3-vector.

---

## Benchmark script: `benchmarks/benchmark_transform.py`

Uses `timeit` to time:
- **Old path:** `transform_array` via manual loop calling `apply_frame_rotation` per site
- **New path:** `transform_array` via `_batch_transform`
- **File path:** `_process` end-to-end (I/O + transform)

Prints wall-clock time and speedup ratio for each. Runs enough repetitions (e.g. 1000)
to produce stable numbers at 429-site scale.

---

## Numerical preservation guarantee

`_batch_transform` implements exactly the same mathematical operations as
`apply_frame_rotation`:
- Same ellipsoid constants (`EARTH_RAD`, `EARTH_E2`)
- Same cross-product formula
- Same `R[0]` and `R[1]` expressions (verified by inspection against `rotation_matrix_neu`)
- Same mm/yr scale factor (1000.0)

The only floating-point difference from the scalar path is operation ordering across
sites, which may produce differences at the level of floating-point rounding (~1e-12 mm/yr),
well within the existing test tolerance of 0.01 mm/yr.
