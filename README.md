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
├── frame_registry.py               # Built-in GAMIT frame registry (78 frames)
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
Reilinger et al. (2006) Euler pole `[0.32840, -0.03504, 0.40682]` deg/Myr.

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

## Reference Frames

PyCvframe includes `frame_registry.py`, a self-contained Python module that
implements the same 78 named reference frames defined in the GAMIT/GLOBK
Fortran subroutine `frame_to_fra.f` (T. A. Herring, last updated 2018-01-17).

All rotation-pole components are stored in **deg/Myr** relative to NNR-NUVEL-1A.
No external GAMIT installation or `frames.dat` file is required.

### Supported frame families

| Family | Frames | Source |
|--------|--------|--------|
| NUVEL-1A plates | `PCFC`, `COCO`, `NAZC`, `CARB`, `SAFD`, `ANTA`, `INDI`, `AUST`, `AFRC`, `ARAB`, `EURA`, `NAFD`, `JUAN`, `PHIL`, `RIVERA`, `SCOTIA` | DeMets et al. (1990) |
| Special / legacy | `NUV-NNR`, `AM-02`, `ITRF93`, `ITRF94`, `GG_PCFC` | GAMIT internal |
| ITRF2000 PMM | `ANTA_I00`, `AUST_I00`, `EURA_I00`, `NOAM_I00`, `PCFC_I00`, `SOAM_I00`, `ITRF00` | Altamimi et al. (2002) |
| ITRF2005 PMM | `AMUR_I05` … `ITRF05` (15 entries) | Altamimi et al. (2007) |
| ITRF2008 PMM | `ANTA_I08` … `AMUR_I08` (15 entries) | Altamimi et al. (2012) |
| ITRF2014 PMM | `ANTA_I14` … `SOMA_I14` (11 entries) | Altamimi et al. (2017) |
| Arabian special | `ARAB_MCC`, `ARAB_M06` | McClusky et al. (2003); Reilinger et al. (2006) |
| Aliases | `ITRF14`, `IGS14`, `NAM14`, `ANT14`, `NAM08`, `IGS08` | — |

### Usage

```python
from frame_registry import frame_to_frame, list_frames

# Print the full frame table
list_frames()

# Rotation vector (rad/yr) from ARAB_I14 to ITRF14
rot = frame_to_frame("ARAB_I14", "ITRF14")
```

Pass any registered frame name as `outframe_or_pole` in `PyCvframe.run()`.
Use `"NONE"` to skip frame rotation (no-op).

The fallback to `./frames.dat` or `~/gg/tables/frames.dat` that exists in the
Fortran source is **not implemented**.  All 78 standard GAMIT frames are
built-in.

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
