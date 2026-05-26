# PyCvframe

Python reimplementation of the GAMIT/GLOBK Fortran program **`cvframe.f`**
(T. Herring et al.).

PyCvframe rotates a GNSS velocity field from one reference frame to another by
subtracting the velocity predicted by a known Euler rotation pole.
Only the horizontal (East and North) components are modified;
the vertical (Up) component is passed through unchanged.

---

## Features

- Accepts an Euler pole as `[wx, wy, wz]` in deg/Myr or rad/yr, or a named frame pair
- Processes standard GAMIT `.vel` files
- Matches Fortran `cvframe` output exactly (to within `.vel` display precision)
- Vectorised API for applying the same rotation to NumPy arrays

---

## Repository layout

```
PyCvframe/
├── pycvframe.py                    # Python module
├── requirements.txt
├── PyCvframe_validation.ipynb      # Validation notebook (no GAMIT needed)
├── data/
│   └── reilinger_2006_igb14.vel    # 429 GNSS velocities in ITRF14
└── reference/
    └── reilinger_2006_arab.vel     # Precomputed Fortran cvframe output
```

---

## Installation

```bash
pip install numpy pandas matplotlib jupyter cartopy   # notebook dependencies
```

No compiled code; no GAMIT/GLOBK installation required.

---

## Usage

### Command line

```bash
python pycvframe.py <input.vel> <output.vel> <inframe> "<wx wy wz>"
```

**Example** (replicates the included test — Arabian plate rotation):

```bash
python pycvframe.py data/reilinger_2006_igb14.vel \
                    output/reilinger_2006_arab.vel \
                    ITRF14 \
                    "0.32840 -0.03504 0.40682"
```

### Python API

```python
from pycvframe import PyCvframe

cv = PyCvframe()
cv.run(
    invel ="data/reilinger_2006_igb14.vel",
    outvel="output/reilinger_2006_arab.vel",
    inframe="ITRF14",
    outframe_or_pole=[0.32840, -0.03504, 0.40682],  # deg/Myr
    pole_units="deg/Myr",
)
```

### Apply to NumPy arrays (after calling `run()` to set the pole)

```python
ve_new, vn_new = cv.transform_array(lons, lats, ve, vn)
```

### Arguments — `run()`

| Argument | Description |
|----------|-------------|
| `invel` | Path to input `.vel` file |
| `outvel` | Path to output `.vel` file |
| `inframe` | Name of the input reference frame (e.g. `ITRF14`, `NONE`) |
| `outframe_or_pole` | Frame name **or** `[wx, wy, wz]` Euler pole vector |
| `pole_units` | `"deg/Myr"` (default) or `"rad/yr"` |

---

## Transformation formula

For each site at geodetic position (lon, lat):

```
site_pos = geod_to_xyz(lat, lon)         # ECEF position (m)
site_vel = rot_prime × site_pos          # predicted velocity (m/yr)
neu_vel  = R(lat, lon) @ site_vel        # rotate to NEU frame
ve_new   = ve  −  neu_vel[East]  × 1000  # subtract East  (mm/yr)
vn_new   = vn  −  neu_vel[North] × 1000  # subtract North (mm/yr)
vu       = unchanged
```

where `rot_prime = pole × π/180 / 10⁶` converts deg/Myr → rad/yr.

---

## Validation

Run the included notebook from the `PyCvframe/` directory:

```bash
jupyter notebook PyCvframe_validation.ipynb
```

The notebook does **not** require GAMIT/GLOBK. It runs PyCvframe on the included test
data and compares the output against the precomputed Fortran reference file
`reference/reilinger_2006_arab.vel`.

### Test results

Test: 429 sites in ITRF14 frame rotated into the Arabian-plate-fixed frame using the
Viltres et al. (2022) Euler pole `[0.32840, -0.03504, 0.40682]` deg/Myr.

| Component | Max residual (PyCvframe − Fortran) | Notes |
|-----------|-----------------------------------|-------|
| East (Ve) | 0.00000 mm/yr | Exact match |
| North (Vn) | 0.01000 mm/yr | Display-precision rounding (±0.005) |
| Up (Vu) | 0.00000 mm/yr | Unchanged by design |

PyCvframe reproduces the Fortran `cvframe` output to within the precision of the
`.vel` file format. The 0.01 mm/yr North residual is a rounding artifact: the `.vel`
format stores velocities to 2 decimal places (±0.005 mm/yr), so two analytically
identical results can differ by up to 0.01 mm/yr after rounding.

### Physical interpretation

| Frame | Typical speed of Arabian plate sites |
|-------|--------------------------------------|
| ITRF14 (input) | ~25 mm/yr toward NE |
| Arabian-plate (output) | ~0–5 mm/yr (residual deformation only) |

The `PyCvframe_validation.ipynb` notebook includes a side-by-side quiver map showing
this velocity reduction.

---

## Mapping to Fortran

| Python function | Fortran equivalent | Purpose |
|----------------|-------------------|---------|
| `geod_to_xyz()` | `GEOD_to_XYZ` | Geodetic (lat, lon, h) → ECEF (m) |
| `rotation_matrix_neu()` | `rotate_geod` | Build XYZ → NEU rotation matrix |
| `apply_frame_rotation()` | Main loop body | Cross-product + NEU rotation + subtract |
| `_parse_data_line()` | Read format 220 | Parse one `.vel` data line |
| `_format_data_line()` | Write format 220 | Format one `.vel` data line |

---

## `.vel` file format

```
* comment lines start with any character other than a space
 lon(°)    lat(°)    ve    vn    de    dn    se    sn    rho    vu    du    su   site
```

All velocities in mm/yr. `de/dn` = rate adjustment; `se/sn/su` = 1-sigma uncertainty
(mm/yr); `rho` = NE correlation coefficient; `vu/du/su` = vertical rate, adjustment, sigma.

---

## Reference

Reilinger, R., et al. (2006). GPS constraints on continental deformation in the
Africa‐Arabia‐Eurasia continental collision zone and implications for the dynamics of
plate interactions. *Journal of Geophysical Research*, 111(B5).

Viltres, Renier, et al. (2022). Present‐day motion of the Arabian Plate. *Tectonics* 41.3. https://doi.org/10.1029/2021TC007013
