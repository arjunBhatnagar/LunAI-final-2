"""Find a launch time (theta0) and trans-lunar burn cutoff duration for which
the scripted ascent actually gets close to the Moon, via a grid search +
local refinement against the simulator itself (a shooting method). No ML --
this is a trajectory targeting search, run once offline to find a usable
Phase 1 (scripted ascent + coast) path for the demo.

The first version of this search only swept launch timing and found the
ascent profile burns the ICPS stage to exhaustion, giving ~21.5 km/s relative
speed at closest approach (way past Earth escape velocity) -- the engine
needs to cut off well before the tank is empty, which is the second
parameter here.
"""

import json
import time
from multiprocessing import Pool

import numpy as np

from simulation import SLSEnv, T_PITCH, PITCH


def targeted_policy(env, burn_duration):
    cfg = env.cfg
    pitch = np.interp(env.t, T_PITCH, PITCH)
    direction = env.local_direction(pitch, 90.0)

    if env.stage_name in ("boost", "core"):
        mdot = cfg.core_engine.mdot_ref
    elif env.stage_name == "icps":
        if not hasattr(env, "_icps_start_t"):
            env._icps_start_t = env.t
        mdot = cfg.icps_engine.mdot_ref if (env.t - env._icps_start_t) < burn_duration else 0.0
    else:
        mdot = 0.0

    return np.concatenate([[mdot], direction])


def closest_approach(params, dt=20.0, max_days=6.0):
    theta0, burn_duration = params
    env = SLSEnv(dt=dt, max_time=max_days * 86400.0, theta0=theta0)
    obs, info = env.reset()

    min_dist = float("inf")
    min_t = 0.0
    min_speed_rel = 0.0

    while env.t < env.max_time:
        action = targeted_policy(env, burn_duration)
        obs, rew, term, trunc, info = env.step(action)

        if info["moon_distance"] < min_dist:
            min_dist = info["moon_distance"]
            min_t = info["t"]
            min_speed_rel = info["speed_rel"]

        if term or trunc:
            break

    return {
        "theta0": theta0,
        "burn_duration": burn_duration,
        "closest_dist_km": min_dist / 1e3,
        "t_days": min_t / 86400.0,
        "speed_rel": min_speed_rel,
        "reason": info.get("reason"),
    }


def _worker(params):
    return closest_approach(params)


if __name__ == "__main__":
    t0 = time.time()

    n_theta = 24
    n_burn = 13
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    burns = np.linspace(0.0, 1070.0, n_burn)  # 0 = no TLI burn, 1070s = full ICPS exhaustion

    grid = [(float(th), float(b)) for th in thetas for b in burns]
    print(f"Coarse grid: {len(grid)} trials ({n_theta} theta0 x {n_burn} burn_duration)")

    with Pool(8) as pool:
        results = pool.map(_worker, grid)

    results.sort(key=lambda r: r["closest_dist_km"])
    print("\nTop 15 closest approaches (coarse grid):")
    for r in results[:15]:
        print(r)

    best = results[0]
    best_theta, best_burn = best["theta0"], best["burn_duration"]

    # Local refinement around the best coarse point.
    refine_thetas = np.linspace(best_theta - 0.15, best_theta + 0.15, 9)
    refine_burns = np.linspace(max(0.0, best_burn - 60), best_burn + 60, 9)
    refine_grid = [(float(th), float(b)) for th in refine_thetas for b in refine_burns]

    with Pool(8) as pool:
        refine_results = pool.map(_worker, refine_grid)

    refine_results.sort(key=lambda r: r["closest_dist_km"])
    print("\nTop 15 after local refinement:")
    for r in refine_results[:15]:
        print(r)

    final_best = refine_results[0]
    print("\nBEST RESULT:")
    print(final_best)

    with open("tli_search_result.json", "w") as f:
        json.dump(
            {"coarse": results, "refine": refine_results, "best": final_best},
            f,
            indent=2,
        )

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    print("Saved full results to tli_search_result.json")
