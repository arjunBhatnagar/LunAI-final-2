"""Body-fixed orientation of the Earth and Moon in the ICRF/J2000 frame.

Both rotations follow the IAU WGCCRE 2009 models (Archinal et al., 2011).
They ignore precession/nutation of the Earth (< 0.4 deg since J2000), which
is far below what the viewer can show.

Every matrix returned here maps body-fixed vectors to ICRF:
``icrf = matrix @ body_fixed``. Its columns are the body X (0 deg lon on the
equator), Y (90 deg E) and Z (north pole) axes expressed in ICRF.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

J2000_JD = 2_451_545.0
_UNIX_EPOCH_JD = 2_440_587.5


def days_since_j2000(when: datetime) -> float:
    """Days from J2000.0 (UTC-based; the ~69 s TT offset is negligible here)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp() / 86_400.0 + _UNIX_EPOCH_JD - J2000_JD


def _rotation_from_pole(ra_deg: float, dec_deg: float, w_deg: float) -> np.ndarray:
    ra, dec, w = (math.radians(value) for value in (ra_deg, dec_deg, w_deg))
    pole = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    # Ascending node of the body equator on the ICRF equator; W is measured from here.
    node = np.array([-math.sin(ra), math.cos(ra), 0.0])
    x_axis = math.cos(w) * node + math.sin(w) * np.cross(pole, node)
    y_axis = np.cross(pole, x_axis)
    return np.column_stack([x_axis, y_axis, pole])


def earth_orientation(when: datetime) -> np.ndarray:
    d = days_since_j2000(when)
    return _rotation_from_pole(0.0, 90.0, 190.147 + 360.9856235 * d)


def moon_orientation(when: datetime) -> np.ndarray:
    d = days_since_j2000(when)
    t = d / 36_525.0
    e = [
        math.radians(angle)
        for angle in (
            125.045 - 0.0529921 * d,
            250.089 - 0.1059842 * d,
            260.008 + 13.0120009 * d,
            176.625 + 13.3407154 * d,
            357.529 + 0.9856003 * d,
            311.589 + 26.4057084 * d,
            134.963 + 13.0649930 * d,
            276.617 + 0.3287146 * d,
            34.226 + 1.7484877 * d,
            15.134 - 0.1589763 * d,
            119.743 + 0.0036096 * d,
            239.961 + 0.1643573 * d,
            25.053 + 12.9590088 * d,
        )
    ]
    s = [math.sin(angle) for angle in e]
    c = [math.cos(angle) for angle in e]
    ra = (
        269.9949 + 0.0031 * t - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2] - 0.0172 * s[3]
        + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12]
    )
    dec = (
        66.5392 + 0.0130 * t + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2] + 0.0068 * c[3]
        - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9] - 0.0009 * c[12]
    )
    w = (
        38.3213 + 13.17635815 * d - 1.4e-12 * d * d
        + 3.5610 * s[0] + 0.1208 * s[1] - 0.0642 * s[2] + 0.0158 * s[3] + 0.0252 * s[4]
        - 0.0066 * s[5] - 0.0047 * s[6] - 0.0046 * s[7] + 0.0028 * s[8] + 0.0052 * s[9]
        + 0.0040 * s[10] + 0.0019 * s[11] - 0.0044 * s[12]
    )
    return _rotation_from_pole(ra, dec, w)


def body_frames(when: datetime) -> dict[str, dict[str, list[float]]]:
    """Body axes in ICRF for the viewer (JSON-friendly)."""
    frames = {}
    for body, matrix in (("earth", earth_orientation(when)), ("moon", moon_orientation(when))):
        frames[body] = {axis: [float(v) for v in matrix[:, index]] for index, axis in enumerate("xyz")}
    return frames


def lat_lon_of(vector_body: np.ndarray) -> tuple[float, float]:
    x, y, z = (float(v) for v in vector_body)
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))
