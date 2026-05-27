# PyCvframe Performance Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-site Python loop in `transform_array` with a fully vectorized NumPy implementation, add a pytest suite, and add a benchmark script.

**Architecture:** Add one private module-level function `_batch_transform` to `pycvframe.py`; wire `transform_array` to call it; create `tests/test_pycvframe.py` as the authoritative test suite; create `benchmarks/benchmark_transform.py` for before/after timing. No other files change.

**Tech Stack:** Python 3.14.4, NumPy 2.4.4, pytest

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pycvframe.py` | Add `_batch_transform`; update `transform_array` body |
| Create | `tests/__init__.py` | Empty — makes `tests/` a package |
| Create | `tests/test_pycvframe.py` | All 5 pytest tests |
| Create | `benchmarks/__init__.py` | Empty |
| Create | `benchmarks/benchmark_transform.py` | Before/after timing script |

---

## Task 1: Install pytest and write the full test suite

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_pycvframe.py`

- [ ] **Step 1: Install pytest for Python 3.14**

```bash
python3.14 -m pip install pytest
```

Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Create `tests/__init__.py`**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 3: Write `tests/test_pycvframe.py`**

```python
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pycvframe import PyCvframe, apply_frame_rotation, _parse_data_line, PI
from frame_registry import frame_to_frame

POLE_DEG_MYR = [0.32840, -0.03504, 0.40682]
POLE_RAD_YR  = np.array(POLE_DEG_MYR) * PI / 180.0 / 1e6

_ROOT     = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(_ROOT, "data",      "reilinger_2006_igb14.vel")
REF_FILE  = os.path.join(_ROOT, "reference", "reilinger_2006_arab.vel")


def _load_vel(path):
    """Return (lons, lats, ve, vn, vu) arrays from a .vel file."""
    rows = {"lon": [], "lat": [], "ve": [], "vn": [], "vu": []}
    with open(path) as fh:
        for line in fh:
            if line and line[0] == " ":
                rec = _parse_data_line(line)
                if rec:
                    for k in rows:
                        rows[k].append(rec[k])
    return tuple(np.array(rows[k]) for k in ("lon", "lat", "ve", "vn", "vu"))


def test_reference_file_equivalence(tmp_path):
    """PyCvframe.run() output matches Fortran reference to <=0.01 mm/yr."""
    outfile = str(tmp_path / "out.vel")
    cv = PyCvframe()
    cv.run(DATA_FILE, outfile, "ITRF14", POLE_DEG_MYR, pole_units="deg/Myr")

    _, _, ve_out, vn_out, vu_out = _load_vel(outfile)
    _, _, ve_ref, vn_ref, vu_ref = _load_vel(REF_FILE)

    assert len(ve_out) == len(ve_ref) == 429
    assert np.max(np.abs(ve_out - ve_ref)) <= 0.01, \
        f"Ve residual {np.max(np.abs(ve_out - ve_ref)):.4f} exceeds 0.01 mm/yr"
    assert np.max(np.abs(vn_out - vn_ref)) <= 0.01, \
        f"Vn residual {np.max(np.abs(vn_out - vn_ref)):.4f} exceeds 0.01 mm/yr"
    assert np.allclose(vu_out, vu_ref, atol=1e-9), "Vu must be unchanged"


def test_transform_array_matches_scalar():
    """Vectorized transform_array agrees with per-site apply_frame_rotation to 1e-10."""
    lons, lats, ve, vn, _ = _load_vel(DATA_FILE)
    cv = PyCvframe()
    cv.rot_prime = POLE_RAD_YR

    ve_vec, vn_vec = cv.transform_array(lons, lats, ve, vn)

    ve_sc = np.empty(len(lons))
    vn_sc = np.empty(len(lons))
    for i in range(len(lons)):
        ve_sc[i], vn_sc[i] = apply_frame_rotation(
            lons[i], lats[i], ve[i], vn[i], cv.rot_prime
        )

    assert np.allclose(ve_vec, ve_sc, atol=1e-10), \
        f"Max Ve diff: {np.max(np.abs(ve_vec - ve_sc)):.2e} mm/yr"
    assert np.allclose(vn_vec, vn_sc, atol=1e-10), \
        f"Max Vn diff: {np.max(np.abs(vn_vec - vn_sc)):.2e} mm/yr"


def test_zero_pole_is_noop():
    """Zero rotation pole leaves all velocities unchanged."""
    lons, lats, ve, vn, _ = _load_vel(DATA_FILE)
    cv = PyCvframe()
    cv.rot_prime = np.zeros(3)

    ve_new, vn_new = cv.transform_array(lons, lats, ve, vn)

    assert np.allclose(ve_new, ve, atol=1e-15), "Ve should be unchanged with zero pole"
    assert np.allclose(vn_new, vn, atol=1e-15), "Vn should be unchanged with zero pole"


def test_single_site_regression():
    """HERS_GPS: lon=0.336 lat=50.867 ve=16.62 vn=16.64 → ve≈16.19 vn≈12.53."""
    cv = PyCvframe()
    cv.rot_prime = POLE_RAD_YR
    ve_new, vn_new = apply_frame_rotation(0.33600, 50.86700, 16.62, 16.64, cv.rot_prime)
    assert abs(ve_new - 16.19) <= 0.01, f"ve_new={ve_new:.4f}, expected ≈16.19"
    assert abs(vn_new - 12.53) <= 0.01, f"vn_new={vn_new:.4f}, expected ≈12.53"


def test_frame_registry():
    """frame_to_frame returns zeros for identical frames; non-zero for distinct frames."""
    same = frame_to_frame("ITRF14", "ITRF14")
    assert np.allclose(same, np.zeros(3)), "Same-frame rotation must be zero"

    diff = frame_to_frame("EURA_I14", "ITRF14")
    assert diff.shape == (3,)
    assert not np.allclose(diff, np.zeros(3)), "Different frames must yield non-zero vector"
```

- [ ] **Step 4: Run the full test suite to establish green baseline**

Run from the repository root:

```bash
cd /Users/jcastrop/Documents/PhD_IU/research/software/PyCvframe
python3.14 -m pytest tests/ -v
```

Expected output — all 5 tests PASS:
```
tests/test_pycvframe.py::test_reference_file_equivalence PASSED
tests/test_pycvframe.py::test_transform_array_matches_scalar PASSED
tests/test_pycvframe.py::test_zero_pole_is_noop PASSED
tests/test_pycvframe.py::test_single_site_regression PASSED
tests/test_pycvframe.py::test_frame_registry PASSED
5 passed
```

If any test fails at this stage, stop — the baseline is broken and the optimization must not proceed until the failure is diagnosed.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_pycvframe.py
git commit -m "Add pytest suite with 5 baseline tests"
```

---

## Task 2: Add `_batch_transform` and update `transform_array`

**Files:**
- Modify: `pycvframe.py` — add `_batch_transform` after line 79 (after `cross_product`); replace `transform_array` body (lines 344–349)

- [ ] **Step 1: Add `_batch_transform` to `pycvframe.py`**

Insert the following block immediately after the `cross_product` function (after line 79,
before the comment `# ---------------------------------------------------------------------------`
that precedes `# Frame-to-frame rotation lookup`):

```python
def _batch_transform(lons: np.ndarray, lats: np.ndarray,
                     ve: np.ndarray, vn: np.ndarray,
                     rot_prime: np.ndarray):
    """
    Vectorized equivalent of apply_frame_rotation for N sites.

    All trig is computed once over the whole array; the cross product is
    inlined; only the North (R row 0) and East (R row 1) projections of
    the rotation matrix are evaluated — the Up component is never needed.
    """
    lat = np.radians(lats)
    lon = np.radians(lons)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)

    # geod_to_xyz — vectorized, (N,) each
    Nr = EARTH_RAD / np.sqrt(1.0 - EARTH_E2 * slat ** 2)
    px = Nr * clat * clon
    py = Nr * clat * slon
    pz = Nr * (1.0 - EARTH_E2) * slat

    # omega × pos — inline, avoids np.cross per-call overhead
    ox, oy, oz = rot_prime
    cvx = oy * pz - oz * py
    cvy = oz * px - ox * pz
    cvz = ox * py - oy * px

    # R[0] (North) = [-slat*clon, -slat*slon,  clat]
    # R[1] (East)  = [-slon,       clon,       0   ]
    neu_N = -slat * clon * cvx - slat * slon * cvy + clat * cvz
    neu_E = -slon * cvx        + clon * cvy

    return ve - neu_E * 1000.0, vn - neu_N * 1000.0
```

- [ ] **Step 2: Replace the body of `transform_array`**

Find the current `transform_array` method (lines ~319–349). Replace its body so it reads:

```python
    def transform_array(self,
                        lons: np.ndarray,
                        lats: np.ndarray,
                        ve: np.ndarray,
                        vn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the frame rotation to arrays of velocities.

        Parameters
        ----------
        lons, lats : array-like
            Site longitudes and latitudes in decimal degrees.
        ve, vn : array-like
            East and North velocities in mm/yr.

        Returns
        -------
        ve_new, vn_new : np.ndarray
            Corrected velocities in mm/yr.
        """
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        ve   = np.asarray(ve,   dtype=float)
        vn   = np.asarray(vn,   dtype=float)
        return _batch_transform(lons, lats, ve, vn, self.rot_prime)
```

- [ ] **Step 3: Run the full test suite — confirm still green**

```bash
python3.14 -m pytest tests/ -v
```

Expected: all 5 tests PASS. If any fail, diff the `_batch_transform` formula against `rotation_matrix_neu` (lines 69–73 of `pycvframe.py`) before proceeding.

- [ ] **Step 4: Commit**

```bash
git add pycvframe.py
git commit -m "Vectorize transform_array via _batch_transform"
```

---

## Task 3: Create benchmark script and run it

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/benchmark_transform.py`

- [ ] **Step 1: Create `benchmarks/__init__.py`**

Create an empty file at `benchmarks/__init__.py`.

- [ ] **Step 2: Write `benchmarks/benchmark_transform.py`**

```python
"""
Before/after timing for PyCvframe.transform_array.

Old path: Python loop calling apply_frame_rotation per site.
New path: transform_array → _batch_transform (fully vectorized).
File path: PyCvframe._process end-to-end (I/O + transform).
"""
import sys
import os
import timeit
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pycvframe import PyCvframe, apply_frame_rotation, _parse_data_line, PI

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "reilinger_2006_igb14.vel")

POLE_RAD_YR = np.array([0.32840, -0.03504, 0.40682]) * PI / 180.0 / 1e6

# ── Load data ──────────────────────────────────────────────────────────────
lons_l, lats_l, ves_l, vns_l = [], [], [], []
with open(DATA_FILE) as fh:
    for line in fh:
        if line and line[0] == " ":
            rec = _parse_data_line(line)
            if rec:
                lons_l.append(rec["lon"]); lats_l.append(rec["lat"])
                ves_l.append(rec["ve"]);   vns_l.append(rec["vn"])

lons = np.array(lons_l); lats = np.array(lats_l)
ves  = np.array(ves_l);  vns  = np.array(vns_l)
N    = len(lons)

cv = PyCvframe()
cv.rot_prime = POLE_RAD_YR

# ── Benchmark functions ────────────────────────────────────────────────────
def old_path():
    """Original: Python loop over apply_frame_rotation."""
    ve_new = np.empty(N)
    vn_new = np.empty(N)
    for i in range(N):
        ve_new[i], vn_new[i] = apply_frame_rotation(
            lons[i], lats[i], ves[i], vns[i], cv.rot_prime
        )
    return ve_new, vn_new


def new_path():
    """Optimized: transform_array → _batch_transform."""
    return cv.transform_array(lons, lats, ves, vns)


def file_path():
    """File I/O path: _process end-to-end (stdout suppressed)."""
    import io, contextlib
    with tempfile.NamedTemporaryFile(suffix=".vel", delete=False, mode="w") as tmp:
        tmpname = tmp.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cv._process(DATA_FILE, tmpname)
    finally:
        os.unlink(tmpname)


# ── Run timings ────────────────────────────────────────────────────────────
REPS      = 1000
FILE_REPS = 20

print(f"\nPyCvframe benchmark — {N} sites")
print(f"Array paths: {REPS} repetitions | File path: {FILE_REPS} repetitions\n")

t_old  = timeit.timeit(old_path,   number=REPS)
t_new  = timeit.timeit(new_path,   number=REPS)
t_file = timeit.timeit(file_path,  number=FILE_REPS)

print(f"{'Path':<22}  {'Total (s)':>10}  {'Per call (µs)':>14}  {'Speedup':>9}")
print("─" * 62)
print(f"{'Old (loop)':22}  {t_old:10.4f}  {t_old/REPS*1e6:14.1f}  {'1.00×':>9}")
print(f"{'New (vectorized)':22}  {t_new:10.4f}  {t_new/REPS*1e6:14.1f}  {t_old/t_new:8.2f}×")
print(f"{'File (_process)':22}  {t_file/FILE_REPS:10.4f}  {'I/O-bound':>14}  {'N/A':>9}")
print()
```

- [ ] **Step 3: Run the benchmark**

```bash
python3.14 benchmarks/benchmark_transform.py
```

Expected output shape (exact numbers will vary by machine):
```
PyCvframe benchmark — 429 sites
Array paths: 1000 repetitions | File path: 20 repetitions

Path                    Total (s)  Per call (µs)    Speedup
──────────────────────────────────────────────────────────────
Old (loop)                  x.xxxx           xxx.x      1.00×
New (vectorized)            x.xxxx            xx.x      x.xx×
File (_process)             x.xxxx       I/O-bound        N/A
```

Record the actual speedup number. A speedup ≥ 3× on 429 sites is expected given the
elimination of the Python loop and redundant trig calls.

- [ ] **Step 4: Run the full test suite one final time**

```bash
python3.14 -m pytest tests/ -v
```

Expected: 5 passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/__init__.py benchmarks/benchmark_transform.py
git commit -m "Add benchmark script for transform_array before/after timing"
```
