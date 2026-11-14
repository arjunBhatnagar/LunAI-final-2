"""Phase 1 targeting, take 2: a properly computed trans-lunar injection burn
instead of a blind "burn for N seconds" search.

The first search (tli_search.py) only ever found a sharp trade-off between
"close" (at escape-speed, ~20 km/s -- useless, can't be captured) and "slow
enough to matter" (~1000 m/s, but still 330,000+ km short). That's because
it was burning in whatever direction the ascent pitch program happened to
point, for an arbitrary duration -- not a real transfer burn.

This version: once the core stage separates (ICPS ignition), compute the
actual velocity needed for a Hohmann-like transfer orbit whose apogee
reaches the Moon's current orbital radius (vis-viva equation), then burn
prograde (continuously realigned with the current velocity vector) until
that speed is reached, then coast. Only theta0 (launch timing / orbital
phasing) is swept -- the burn itself is physics-computed, not guessed.
"""

import json
import time
from multiprocessing import Pool

import numpy as np

from simulation import SLSEnv, T_PITCH, PITCH, MU_EARTH


def hohmann_transfer_speed(r_mag, r_apogee):
    a = (r_mag + r_apogee) / 2.0
    return np.sqrt(MU_EARTH * (2.0 / r_mag - 1.0 / a))


def hohmann_policy(env, state, coast_time=0.0):
    cfg = env.cfg

    if env.stage_name in ("boost", "core"):
        pitch = np.interp(env.t, T_PITCH, PITCH)
        direction = env.local_direction(pitch, 90.0)
        return np.concatenate([[cfg.core_engine.mdot_ref], direction])

    if env.stage_name != "icps":
        return np.concatenate([[0.0], env.attitude])

    r_mag = float(np.linalg.norm(env.r))
    radial_speed = float(np.dot(env.r, env.v) / r_mag)

    if "icps_start_t" not in state:
        state["icps_start_t"] = env.t
        state["prev_radial_speed"] = radial_speed
        state["ready_to_burn"] = False
        state["burning"] = False

    # Coast for coast_time seconds (spans days -- long enough to actually
    # change which part of its 27-day orbit the Moon is in), THEN wait for
    # the next apsis (purely tangential velocity) before burning -- firing
    # the transfer burn at an arbitrary, non-apsis point would not be a
    # valid Hohmann injection regardless of timing.
    if not state["ready_to_burn"] and "target_speed" not in state:
        if env.t - state["icps_start_t"] >= coast_time:
            if state["prev_radial_speed"] * radial_speed <= 0.0:
                state["ready_to_burn"] = True
                r_apogee = float(np.linalg.norm(env.r_moon))
                state["target_speed"] = hohmann_transfer_speed(r_mag, r_apogee)
                state["burning"] = True
        state["prev_radial_speed"] = radial_speed
        return np.concatenate([[0.0], env.attitude])

    speed = float(np.linalg.norm(env.v))
    direction = env.v / speed if speed > 1e-6 else env.attitude

    if state["burning"] and speed < state["target_speed"]:
        return np.concatenate([[cfg.icps_engine.mdot_ref], direction])

    state["burning"] = False
    return np.concatenate([[0.0], direction])


def closest_approach(params, dt=60.0, max_days=25.0):
    theta0, coast_time = params
    env = SLSEnv(dt=dt, max_time=max_days * 86400.0, theta0=theta0)
    obs, info = env.reset()
    state = {}

    min_dist = float("inf")
    min_t = 0.0
    min_speed_rel = 0.0

    while env.t < env.max_time:
        action = hohmann_policy(env, state, coast_time)
        obs, rew, term, trunc, info = env.step(action)

        if info["moon_distance"] < min_dist:
            min_dist = info["moon_distance"]
            min_t = info["t"]
            min_speed_rel = info["speed_rel"]

        if term or trunc:
            break

    return {
        "theta0": theta0,
        "coast_time_days": coast_time / 86400.0,
        "closest_dist_km": min_dist / 1e3,
        "t_days": min_t / 86400.0,
        "speed_rel": min_speed_rel,
        "reason": info.get("reason"),
    }


def _worker(params):
    return closest_approach(params)


if __name__ == "__main__":
    t0 = time.time()

    # Coasting a handful of early apsis crossings didn't change anything --
    # they're all within hours of each other, nowhere near enough to matter
    # against the Moon's 27.3-day orbit. Sweep real elapsed coast TIME
    # instead, out to most of a lunar month, to actually find the right
    # launch-to-burn phasing.
    thetas = [2.2, 2.5, 2.8]
    coast_days = list(np.linspace(0.0, 24.0, 25))

    grid = [(float(th), float(d) * 86400.0) for th in thetas for d in coast_days]
    print(f"Grid: {len(grid)} trials ({len(thetas)} theta0 x {len(coast_days)} coast_days)")

    with Pool(8) as pool:
        results = pool.map(_worker, grid)

    results.sort(key=lambda r: r["closest_dist_km"])
    print("\nTop 20 closest approaches:")
    for r in results[:20]:
        print(r)

    final_best = results[0]
    print("\nBEST RESULT:")
    print(final_best)

    with open("tli_hohmann_result.json", "w") as f:
        json.dump({"all": results, "best": final_best}, f, indent=2)

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
