"""Lambert's problem solver (universal variable formulation, Curtis'
"Orbital Mechanics for Engineering Students" algorithm 5.2) -- given two
position vectors and a time of flight, find the velocity at the first
point that puts you on a Keplerian arc through the second point in exactly
that time. This lets us aim a transfer burn at where the Moon WILL be on
arrival, instead of just matching its current distance.
"""

import numpy as np


def stumpff_C(z):
    if z > 1e-6:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1) / (-z)
    return 0.5


def stumpff_S(z):
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0


def lambert_solve(r0, r1, tof, mu, prograde=True, max_iter=200, tol=1e-6):
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    r0n = np.linalg.norm(r0)
    r1n = np.linalg.norm(r1)

    cross = np.cross(r0, r1)
    cos_dnu = np.clip(np.dot(r0, r1) / (r0n * r1n), -1.0, 1.0)
    dnu = np.arccos(cos_dnu)

    if prograde:
        if cross[2] < 0:
            dnu = 2 * np.pi - dnu
    else:
        if cross[2] >= 0:
            dnu = 2 * np.pi - dnu

    A = np.sin(dnu) * np.sqrt(r0n * r1n / (1 - cos_dnu))
    if abs(A) < 1e-12:
        raise ValueError("Lambert geometry degenerate (A ~ 0)")

    def y_of_z(z, C, S):
        if C <= 1e-10:
            return None
        return r0n + r1n + A * (z * S - 1) / np.sqrt(C)

    def t_of_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        y = y_of_z(z, C, S)
        if y is None or y < 0:
            return None
        chi = np.sqrt(y / C)
        val = (chi**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return val if np.isfinite(val) else None

    # t(z) is monotonically increasing in z (Curtis sec. 5.3) -- bracket
    # and bisect instead of Newton-Raphson, which avoids any derivative
    # sign/scale bugs entirely. Stumpff C(z) has removable singularities
    # at z = (2*pi*k)^2 for integer k (C -> 0 there) -- start just off
    # one of those points, not exactly on it.
    z_lo, z_hi = -4 * np.pi**2 + 0.5, 4 * np.pi**2 - 0.5
    while t_of_z(z_hi) is None or t_of_z(z_hi) < tof:
        z_hi *= 1.6
        if z_hi > 1e9:
            raise RuntimeError("Could not bracket a valid z_hi")

    t_lo = t_of_z(z_lo)
    while t_lo is None or t_lo > tof:
        z_lo = z_lo / 1.6 if z_lo < 0 else -1.0
        t_lo = t_of_z(z_lo)
        if z_lo < -1e9:
            raise RuntimeError("Could not bracket a valid z_lo")

    z = 0.5 * (z_lo + z_hi)
    for _ in range(max_iter):
        z = 0.5 * (z_lo + z_hi)
        t_z = t_of_z(z)

        if t_z is None:
            z_lo = z
            continue

        if abs(t_z - tof) < tol * max(1.0, tof):
            break

        if t_z < tof:
            z_lo = z
        else:
            z_hi = z
    else:
        raise RuntimeError(f"Lambert solver did not converge (last z={z}, t_z={t_z}, target={tof})")

    C = stumpff_C(z)
    S = stumpff_S(z)
    y = y_of_z(z, C, S)

    f = 1 - y / r0n
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / r1n

    v0 = (r1 - f * r0) / g
    v1 = (gdot * r1 - r0) / g

    return v0, v1


if __name__ == "__main__":
    # Sanity check: circular-to-circular Hohmann transfer (same plane),
    # compare against the textbook closed-form vis-viva result.
    mu = 3.986004418e14  # Earth mu, m^3/s^2
    r0_mag = 6.678e6  # ~300km LEO
    r1_mag = 4.2164e7  # GEO-ish, for a clean test case

    a_transfer = (r0_mag + r1_mag) / 2
    v0_expected = np.sqrt(mu * (2 / r0_mag - 1 / a_transfer))

    tof = np.pi * np.sqrt(a_transfer**3 / mu)  # half the transfer ellipse period

    # 179.9 deg, not exactly 180 -- exact 180 is a genuine singularity for
    # Lambert's problem (orbital plane undefined), not specific to this
    # solver. Close enough to 180 that the answer should still match
    # vis-viva closely.
    angle = np.radians(179.9)
    r0 = np.array([r0_mag, 0.0, 0.0])
    r1 = np.array([r1_mag * np.cos(angle), r1_mag * np.sin(angle), 0.0])

    v0, v1 = lambert_solve(r0, r1, tof, mu)
    v0_mag = np.linalg.norm(v0)

    print(f"Expected v0 (vis-viva): {v0_expected:.3f} m/s")
    print(f"Lambert solver v0:      {v0_mag:.3f} m/s")
    print(f"Difference: {abs(v0_mag - v0_expected):.6f} m/s ({100*abs(v0_mag-v0_expected)/v0_expected:.4f}%)")
