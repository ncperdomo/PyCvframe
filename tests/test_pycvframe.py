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
    # Both files are formatted at 2 dp (0.01 mm/yr resolution); floating-point
    # subtraction of two numbers that differ by exactly 0.01 can produce a result
    # marginally above 0.01 (e.g. 0.010000000000001563) due to binary representation.
    # We round to 10 dp to remove that sub-femtometre artefact before comparing.
    assert np.max(np.round(np.abs(ve_out - ve_ref), 10)) <= 0.01, \
        f"Ve residual {np.max(np.abs(ve_out - ve_ref)):.4f} exceeds 0.01 mm/yr"
    assert np.max(np.round(np.abs(vn_out - vn_ref), 10)) <= 0.01, \
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
    # Values are ~1e-9 rad/yr; np.allclose default atol=1e-8 would treat them as zero.
    # Use atol=1e-11 to verify they are physically non-zero.
    assert not np.allclose(diff, np.zeros(3), atol=1e-11), "Different frames must yield non-zero vector"
