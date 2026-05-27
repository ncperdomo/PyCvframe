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


def old_path():
    ve_new = np.empty(N)
    vn_new = np.empty(N)
    for i in range(N):
        ve_new[i], vn_new[i] = apply_frame_rotation(
            lons[i], lats[i], ves[i], vns[i], cv.rot_prime
        )
    return ve_new, vn_new


def new_path():
    return cv.transform_array(lons, lats, ves, vns)


def file_path():
    import io, contextlib
    with tempfile.NamedTemporaryFile(suffix=".vel", delete=False, mode="w") as tmp:
        tmpname = tmp.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cv._process(DATA_FILE, tmpname)
    finally:
        os.unlink(tmpname)


REPS = 1000
FILE_REPS = 20

print(f"\nPyCvframe benchmark — {N} sites")
print(f"Array paths: {REPS} repetitions | File path: {FILE_REPS} repetitions\n")

t_old  = timeit.timeit(old_path,  number=REPS)
t_new  = timeit.timeit(new_path,  number=REPS)
t_file = timeit.timeit(file_path, number=FILE_REPS)

print(f"{'Path':<22}  {'Total (s)':>10}  {'Per call (µs)':>14}  {'Speedup':>9}")
print("─" * 62)
print(f"{'Old (loop)':22}  {t_old:10.4f}  {t_old/REPS*1e6:14.1f}  {'1.00×':>9}")
print(f"{'New (vectorized)':22}  {t_new:10.4f}  {t_new/REPS*1e6:14.1f}  {t_old/t_new:8.2f}×")
print(f"{'File (_process)':22}  {t_file/FILE_REPS:10.4f}  {'I/O-bound':>14}  {'N/A':>9}")
print()
