"""Run a trained PPO policy in the simulator and plot the resulting trajectory."""

import argparse

import numpy as np
from stable_baselines3 import PPO

from lunar_gym_env import LunarLandingEnv
from simulation import plot_ascent, plot_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_lunar_lander.zip")
    parser.add_argument("--scenario", choices=["launch", "lunar_descent"], default="launch")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--descent-horizontal-speed", type=float, default=None)
    parser.add_argument("--descent-start-alt", type=float, default=15_000.0)
    parser.add_argument("--max-turn-rate-deg", type=float, default=None)
    parser.add_argument("--safe-landing-speed", type=float, default=3.0)
    parser.add_argument("--speed-penalty-coef", type=float, default=1.0)
    parser.add_argument("--dense-speed-coef", type=float, default=2e-3)
    parser.add_argument("--auto-retrograde", action="store_true")
    parser.add_argument(
        "--scripted-throttle",
        action="store_true",
        help="Bypass the learned/model throttle with a hardcoded bang-bang "
        "sink-rate controller (fire when sinking faster than --target-sink-rate, "
        "otherwise cut the engine). No --model needed if combined with --auto-retrograde.",
    )
    parser.add_argument("--target-sink-rate", type=float, default=-2.0)
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Path to write the full step-by-step trajectory as CSV (for an external visualizer)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip opening the Plotly windows")
    args = parser.parse_args()

    if args.scenario == "lunar_descent":
        dt = args.dt if args.dt is not None else 1.0
        max_time = args.max_time if args.max_time is not None else 600.0
        descent_horizontal_speed = (
            args.descent_horizontal_speed if args.descent_horizontal_speed is not None else 50.0
        )
        max_turn_rate_deg = args.max_turn_rate_deg if args.max_turn_rate_deg is not None else 15.0
    else:
        dt = args.dt if args.dt is not None else 5.0
        max_time = args.max_time if args.max_time is not None else 4 * 86400.0
        descent_horizontal_speed = args.descent_horizontal_speed or 0.0
        max_turn_rate_deg = args.max_turn_rate_deg if args.max_turn_rate_deg is not None else 2.0

    env = LunarLandingEnv(
        dt=dt,
        max_time=max_time,
        start_mode=args.scenario,
        descent_start_alt_m=args.descent_start_alt,
        descent_horizontal_speed=descent_horizontal_speed,
        max_turn_rate_deg=max_turn_rate_deg,
        safe_landing_speed_mps=args.safe_landing_speed,
        speed_penalty_coef=args.speed_penalty_coef,
        dense_speed_coef=args.dense_speed_coef,
        auto_retrograde=args.auto_retrograde,
        scripted_throttle=args.scripted_throttle,
        target_sink_rate=args.target_sink_rate,
    )

    needs_model = not (args.scripted_throttle and args.auto_retrograde)
    model = PPO.load(args.model) if needs_model else None

    obs, info = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    print("Reason:", info.get("reason"))
    print(f"Final target distance: {env.sim.target_distance / 1e3:.2f} km")
    print(f"Landing speed (rel. to Moon): {info['landing_speed']:.2f} m/s")
    print(f"Remaining ICPS propellant: {env.sim.icps_prop:,.1f} kg")

    if args.save_csv:
        env.sim.save_history_csv(args.save_csv)
        print(f"Saved {len(env.sim.history)} steps of trajectory data to {args.save_csv}")

    if not args.no_plot:
        plot_trajectory(env.sim).show()
        plot_ascent(env.sim).show()


if __name__ == "__main__":
    main()
