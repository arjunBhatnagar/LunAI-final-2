"""Phase 1 targeting, take 3: real Lambert targeting.

Instead of matching the Moon's CURRENT distance (take 1) or guessing a
coast delay (take 2), this predicts where the Moon will actually BE at a
chosen arrival time, solves Lambert's problem for the burn that gets the
rocket there in exactly that time, and executes it. This is what real
trans-lunar trajectory design actually does.
"""

import json
import time
from multiprocessing import Pool

import numpy as np

from simulation import (
    SLSEnv,
    T_PITCH,
    PITCH,
    MU_EARTH,
    MU_EARTH_MOON,
    LUNAR_DESCENT_START_ALT_M,
    MOON_TARGET_LAT_DEG,
    MOON_TARGET_LON_DEG,
    selenographic_to_frame,
)
from lambert import lambert_solve


def predict_moon_position(r_moon0, v_moon0, tof, dt=3600.0):
    r = np.array(r_moon0, dtype=float)
    v = np.array(v_moon0, dtype=float)
    t = 0.0

    def accel(r):
        return -MU_EARTH_MOON * r / np.linalg.norm(r) ** 3

    while t < tof:
        h = min(dt, tof - t)
        k1v = accel(r)
        k1r = v
        k2v = accel(r + 0.5 * h * k1r)
        k2r = v + 0.5 * h * k1v
        k3v = accel(r + 0.5 * h * k2r)
        k3r = v + 0.5 * h * k2v
        k4v = accel(r + h * k3r)
        k4r = v + h * k3v
        r = r + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h

    return r, v


class LambertPolicy:
    def __init__(self, tof):
        self.tof = tof
        self.state = {}

    def __call__(self, env):
        cfg = env.cfg

        if env.stage_name != "icps":
            return np.concatenate([[0.0], env.attitude])

        state = self.state
        r_mag = float(np.linalg.norm(env.r))
        radial_speed = float(np.dot(env.r, env.v) / r_mag)

        if "prev_radial_speed" not in state:
            state["prev_radial_speed"] = radial_speed
            state["ready"] = False

        if not state["ready"] and "v_target" not in state:
            if state["prev_radial_speed"] * radial_speed <= 0.0 and env.t > 200.0:
                r1, _ = predict_moon_position(env.r_moon, env.v_moon, self.tof)
                v0, _ = lambert_solve(env.r, r1, self.tof, MU_EARTH, prograde=True)
                state["v_target"] = v0
                state["ready"] = True
            state["prev_radial_speed"] = radial_speed
            return np.concatenate([[0.0], env.attitude])

        v_target = state["v_target"]
        dv = v_target - env.v
        dv_mag = float(np.linalg.norm(dv))

        if dv_mag > 5.0:
            direction = dv / dv_mag
            return np.concatenate([[cfg.icps_engine.mdot_ref], direction])

        return np.concatenate([[0.0], env.attitude])


def closest_approach(params, dt=60.0, max_days=10.0):
    """Impulsive version: apply the Lambert delta-v instantly (standard
    practice for transfer planning) instead of a gradually-executed,
    continuously-re-aimed thrust. The gradual version let position drift
    during the (slow, ~20 min) burn while re-aiming at a fixed target
    velocity vector -- the required direction swings faster than the
    attitude slew rate can track, mis-pointed thrust drives the orbit's
    low point dangerously low, and the exponential atmosphere model goes
    numerically unstable there. Impulsive application sidesteps all of it.
    """
    theta0, tof_days = params
    tof = tof_days * 86400.0

    env = SLSEnv(dt=dt, max_time=max_days * 86400.0, theta0=theta0, start_mode="orbit")
    obs, info = env.reset()

    # Aim at the actual landing target (lat/lon, at the landing-phase's own
    # start altitude) instead of the Moon's bare center -- same fixed
    # offset the rest of the project already uses, applied to the Moon's
    # PREDICTED position at arrival time.
    target_offset = selenographic_to_frame(
        MOON_TARGET_LAT_DEG, MOON_TARGET_LON_DEG, LUNAR_DESCENT_START_ALT_M
    )

    min_dist = float("inf")
    min_t = 0.0
    min_speed_rel = 0.0
    burn_dv = None

    # start_mode="orbit" begins on an EXACTLY circular orbit -- velocity is
    # purely tangential at every point on it already, so (unlike the launch
    # ascent's messy post-boost orbit) there's no apsis to wait for. Fire
    # right away.
    r_moon1, v_moon1 = predict_moon_position(env.r_moon, env.v_moon, tof)
    r1 = r_moon1 + target_offset
    v0, _ = lambert_solve(env.r, r1, tof, MU_EARTH, prograde=True)
    dv_vec = v0 - env.v
    dv_mag = float(np.linalg.norm(dv_vec))

    m0 = env.mass
    Ve = env.cfg.icps_engine.Ve
    mf = m0 * np.exp(-dv_mag / Ve)
    fuel_used = m0 - mf
    burn_dv = dv_mag

    if fuel_used <= env.icps_prop:
        env.icps_prop -= fuel_used
        env.y[3:6] = v0
        env.attitude = v0 / np.linalg.norm(v0)

    while env.t < env.max_time:
        action = np.concatenate([[0.0], env.attitude])
        obs, rew, term, trunc, info = env.step(action)

        if info["target_distance"] < min_dist:
            min_dist = info["target_distance"]
            min_t = info["t"]
            min_speed_rel = info["landing_speed"]

        if term or trunc:
            break

    return {
        "theta0": theta0,
        "tof_days": tof_days,
        "closest_dist_km": min_dist / 1e3,
        "t_days": min_t / 86400.0,
        "speed_rel": min_speed_rel,
        "burn_dv": burn_dv,
        "reason": info.get("reason"),
    }


def _worker(params):
    return closest_approach(params)


if __name__ == "__main__":
    t0 = time.time()

    thetas = list(np.linspace(3.5, 5.5, 21))
    tofs = list(np.linspace(3.5, 7.5, 17))

    grid = [(float(th), float(tf)) for th in thetas for tf in tofs]
    print(f"Grid: {len(grid)} trials ({len(thetas)} theta0 x {len(tofs)} tof_days)")

    with Pool(8) as pool:
        results = pool.map(_worker, grid)

    impacts = [r for r in results if r["reason"] == "lunar impact"]
    print(f"\n{len(impacts)} / {len(results)} trials hit the Moon")

    results.sort(key=lambda r: r["closest_dist_km"])
    print("\nTop 10 closest to target coordinates (any outcome):")
    for r in results[:10]:
        print(r)

    impacts_by_speed = sorted(impacts, key=lambda r: r["speed_rel"])
    print("\nTop 10 impacts by LOWEST landing speed (best handoff candidates):")
    for r in impacts_by_speed[:10]:
        print(r)

    final_best = impacts_by_speed[0] if impacts_by_speed else results[0]
    print("\nBEST RESULT (lowest-speed impact):")
    print(final_best)

    with open("tli_lambert_result.json", "w") as f:
        json.dump({"all": results, "best": final_best}, f, indent=2)

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
