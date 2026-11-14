"""One continuous Earth-to-Moon-landing demo trajectory.

Three physically-simulated segments, stitched into a single SLSEnv history
so the whole thing exports as one continuous CSV:

  1. ASCENT (genuine): real pad liftoff, SRB + core burn, scripted pitch
     program (same baseline_policy as simulation.py's own demo). This is
     actually flown, not faked.

  2. TRANS-LUNAR INJECTION (scripted, validated): at core separation, the
     rocket is re-posed into the same clean circular parking orbit the
     already-validated tli_lambert.py grid search used ("orbit" start
     mode's own formula, applied mid-flight instead of at t=0). This is a
     known, pre-existing simplification -- confirmed in this session that
     the simulator's crude pitch program cannot reach a usable transfer
     orbit on its own (needs 4,500-9,500 m/s of Lambert delta-v depending
     on launch phasing, vs. the ICPS's ~3,500 m/s total budget), which is
     exactly why "orbit" mode exists in simulation.py already. From the
     clean parking orbit, the validated Lambert-targeted TLI burn (theta0,
     tof from the earlier grid search) fires immediately, then coasts to
     closest approach to the lat/lon landing target.

  3. LANDING (RL): at closest approach, the rocket is re-posed into the
     exact starting condition the throttle-only descent policy was trained
     on (15 km above the target, near-zero relative speed) -- the
     Phase1/Phase2 handoff gap already flagged in the audit, made
     walkable for a demo. Elapsed time, trajectory history, and remaining
     fuel all carry over for real across every splice; only position and
     velocity are re-posed.
"""

import numpy as np
from stable_baselines3 import PPO

from simulation import (
    SLSEnv,
    T_PITCH,
    PITCH,
    MU_EARTH,
    MOON_POS0,
    MOON_VEL0,
    earth_radius,
    MOON_TARGET_LAT_DEG,
    MOON_TARGET_LON_DEG,
    LUNAR_DESCENT_START_ALT_M,
    selenographic_to_frame,
    local_frame,
)
from lambert import lambert_solve
from tli_lambert import predict_moon_position

DESCENT_MODEL = "checkpoints_throttle_v2/ppo_throttle_v2_2000000_steps.zip"
TLI_THETA0 = 4.4        # validated: tli_lambert.py grid search winner
TLI_TOF_DAYS = 5.5      # validated: tli_lambert.py grid search winner
ORBIT_ALTITUDE_M = 300_000.0


def run_full_demo(out_csv="full_mission.csv"):

    env = SLSEnv(dt=5.0, max_time=10 * 86400.0, start_mode="launch")
    obs, info = env.reset()

    print("Phase 1: ascent (SRB + core, scripted pitch program)...")
    while env.stage_name in ("boost", "core"):
        pitch = np.interp(env.t, T_PITCH, PITCH)
        direction = env.local_direction(pitch, 90.0)
        action = np.concatenate([[env.cfg.core_engine.mdot_ref], direction])
        obs, rew, term, trunc, info = env.step(action, dt=5.0)

        if term or trunc:
            print(f"Ascent ended early: {info.get('reason')}")
            return

    print(f"  core separation at t={env.t:.1f}s, altitude={info['altitude']/1e3:.1f} km, "
          f"speed={info['speed_inertial']:.1f} m/s")

    print("Phase 2: re-pose into validated trans-lunar parking orbit, then Lambert TLI burn...")

    r_mag = earth_radius + ORBIT_ALTITUDE_M
    v_circ = np.sqrt(MU_EARTH / r_mag)
    n = np.cross(MOON_POS0, MOON_VEL0)
    n = n / np.linalg.norm(n)
    e1 = MOON_POS0 / np.linalg.norm(MOON_POS0)
    e2 = np.cross(n, e1)

    r0 = r_mag * (np.cos(TLI_THETA0) * e1 + np.sin(TLI_THETA0) * e2)
    v0 = v_circ * (-np.sin(TLI_THETA0) * e1 + np.cos(TLI_THETA0) * e2)

    env.y[0:3] = r0
    env.y[3:6] = v0
    env.attitude = v0 / np.linalg.norm(v0)

    tof = TLI_TOF_DAYS * 86400.0
    target_offset = selenographic_to_frame(
        MOON_TARGET_LAT_DEG, MOON_TARGET_LON_DEG, LUNAR_DESCENT_START_ALT_M
    )
    r_moon1, _ = predict_moon_position(env.r_moon, env.v_moon, tof)
    r1 = r_moon1 + target_offset
    v_tli, _ = lambert_solve(env.r, r1, tof, MU_EARTH, prograde=True)

    dv_mag = float(np.linalg.norm(v_tli - env.v))
    m0 = env.mass
    Ve = env.cfg.icps_engine.Ve
    fuel_used = m0 * (1 - np.exp(-dv_mag / Ve))

    if fuel_used > env.icps_prop:
        print(f"TLI burn infeasible: needs {dv_mag:.1f} m/s / {fuel_used:.1f} kg, "
              f"only {env.icps_prop:.1f} kg available.")
        return

    env.icps_prop -= fuel_used
    env.y[3:6] = v_tli
    env.attitude = v_tli / np.linalg.norm(v_tli)
    print(f"  TLI burn: dv={dv_mag:.1f} m/s, fuel used={fuel_used:.1f} kg, "
          f"fuel remaining={env.icps_prop:.1f} kg")

    print("Phase 2b: coasting to closest approach...")
    min_dist = float("inf")
    increasing_steps = 0
    while env.t < env.max_time:
        action = np.concatenate([[0.0], env.attitude])
        obs, rew, term, trunc, info = env.step(action, dt=60.0)

        if term or trunc:
            print(f"Coast ended early: {info.get('reason')}")
            break

        d = info["target_distance"]
        if d < min_dist:
            min_dist = d
            increasing_steps = 0
        else:
            increasing_steps += 1
        if increasing_steps > 20:
            break

    print(f"  closest approach: {min_dist/1e3:.1f} km, "
          f"landing_speed(rel. Moon)={info['landing_speed']:.1f} m/s, "
          f"t={env.t/86400:.2f} days")

    print("Phase 3: re-pose into the descent policy's trained starting condition, then land...")

    offset = selenographic_to_frame(
        MOON_TARGET_LAT_DEG, MOON_TARGET_LON_DEG, LUNAR_DESCENT_START_ALT_M
    )
    r0 = env.r_moon + offset
    up = offset / np.linalg.norm(offset)
    east, _, _ = local_frame(offset)
    v0 = env.v_moon + 50.0 * east

    env.y[0:3] = r0
    env.y[3:6] = v0
    env.attitude = up
    env.max_turn_rate = np.radians(15.0)

    model = PPO.load(DESCENT_MODEL)
    mdot_ref_scale = max(env.cfg.core_engine.mdot_ref, env.cfg.icps_engine.mdot_ref)

    for _ in range(600):
        obs = env.observe()
        action_raw, _ = model.predict(obs, deterministic=True)
        throttle = float(np.clip(action_raw[0], 0.0, 1.0))

        v_rel = env.v - env.v_moon
        speed = float(np.linalg.norm(v_rel))
        direction = -v_rel / speed if speed > 1e-6 else up

        action = np.concatenate([[throttle * mdot_ref_scale], direction])
        obs, rew, term, trunc, info = env.step(action, dt=1.0)

        if term or trunc:
            break

    print(f"  Phase 3 complete: reason={info.get('reason')}, "
          f"target_distance={info['target_distance']:.1f} m, "
          f"landing_speed={info['landing_speed']:.2f} m/s, "
          f"fuel_remaining={env.icps_prop:.1f} kg")

    env.save_history_csv(out_csv)
    print(f"\nSaved {len(env.history)} steps spanning all three phases to {out_csv}")

    return env


if __name__ == "__main__":
    run_full_demo()
