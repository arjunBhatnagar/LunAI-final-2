"""Train a PPO agent to fly the SLS/ICPS stack to a landing target on the Moon.

Reward (see SLSEnv.compute_reward): shaped by closing distance to the target,
penalized per kg of propellant burned, and scored on final landing accuracy.
"""

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lunar_gym_env import LunarLandingEnv


class ClipLogStdCallback(BaseCallback):
    """Hard-clamp the policy's learned log_std after every update.

    Needed when ent_coef alone doesn't stop std from drifting up -- e.g. a
    low-dimensional action feeding a bang-bang actuator barely penalizes
    large std (most of the variance gets clipped away by the actuator
    itself), so there's little gradient pressure pushing it back down and
    it can random-walk upward indefinitely even with ent_coef=0.
    """

    def __init__(self, max_std=3.0, verbose=0):
        super().__init__(verbose)
        self.max_log_std = float(np.log(max_std))

    def _on_step(self) -> bool:
        with torch.no_grad():
            self.model.policy.log_std.clamp_(max=self.max_log_std)
        return True


class PenaltyRampCallback(BaseCallback):
    """Linearly ramp the non-landing (truncation) penalty from `start` to
    `end` over training.

    A fixed penalty kept failing to dominate a "park in a stable hover
    oscillation forever" equilibrium -- whatever fixed value was tried, the
    policy found it cheaper than risking a bad terminal accuracy/speed
    score. Ramping starts gentle (easy to learn *something*) and keeps
    tightening the screws, so the hovering strategy's cost keeps rising
    even if the policy has already settled into it.
    """

    def __init__(self, start, end, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps

    def _on_rollout_start(self) -> None:
        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        value = self.start + progress * (self.end - self.start)
        self.training_env.env_method("set_truncated_penalty", value)
        if self.verbose:
            print(f"[PenaltyRamp] t={self.num_timesteps} truncated_penalty={value:.1f}")

    def _on_step(self) -> bool:
        return True


def make_env(
    dt,
    max_time,
    fuel_penalty_coef,
    start_mode,
    descent_start_alt_m,
    descent_horizontal_speed,
    max_turn_rate_deg,
    safe_landing_speed_mps,
    speed_penalty_coef,
    dense_speed_coef,
    auto_retrograde,
    truncated_penalty_init,
):
    def _init():
        return Monitor(
            LunarLandingEnv(
                dt=dt,
                max_time=max_time,
                fuel_penalty_coef=fuel_penalty_coef,
                start_mode=start_mode,
                descent_start_alt_m=descent_start_alt_m,
                descent_horizontal_speed=descent_horizontal_speed,
                max_turn_rate_deg=max_turn_rate_deg,
                safe_landing_speed_mps=safe_landing_speed_mps,
                speed_penalty_coef=speed_penalty_coef,
                dense_speed_coef=dense_speed_coef,
                auto_retrograde=auto_retrograde,
                truncated_penalty=truncated_penalty_init,
            )
        )

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--scenario",
        choices=["launch", "lunar_descent"],
        default="launch",
        help="'launch' = full pad-to-Moon flight. 'lunar_descent' = short "
        "episode starting just above the landing target (fast to train).",
    )
    parser.add_argument("--dt", type=float, default=None, help="Simulation step size (s)")
    parser.add_argument("--max-time", type=float, default=None, help="Episode time limit (s)")
    parser.add_argument("--fuel-penalty-coef", type=float, default=1e-3)
    parser.add_argument(
        "--safe-landing-speed", type=float, default=3.0, help="Touchdown speed (m/s) below which no penalty applies"
    )
    parser.add_argument(
        "--speed-penalty-coef", type=float, default=1.0, help="Reward lost per m/s above the safe landing speed"
    )
    parser.add_argument(
        "--dense-speed-coef",
        type=float,
        default=2e-3,
        help="Per-step (not just terminal) penalty per m/s above the safe landing speed",
    )
    parser.add_argument("--descent-start-alt", type=float, default=15_000.0)
    parser.add_argument("--descent-horizontal-speed", type=float, default=None)
    parser.add_argument(
        "--auto-retrograde",
        action="store_true",
        help="Lock thrust direction to anti-velocity and let PPO control only "
        "throttle (duty-cycle). Much easier to learn than joint attitude+throttle.",
    )
    parser.add_argument(
        "--max-turn-rate-deg",
        type=float,
        default=None,
        help="Attitude slew rate limit (deg/s). lunar_descent needs a faster "
        "slew than the launch stack to be able to correct lateral drift.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="PPO entropy bonus. lunar_descent defaults this > 0 because the "
        "immediate fuel penalty otherwise collapses exploration to zero-thrust.",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=3.0,
        help="Hard clamp on the policy's learned action std (via log_std), "
        "applied every step. Needed when a low-dim action feeding a "
        "bang-bang actuator lets std drift up even with ent_coef=0.",
    )
    parser.add_argument(
        "--penalty-start",
        type=float,
        default=-300.0,
        help="Non-landing (truncation) penalty at the start of training.",
    )
    parser.add_argument(
        "--penalty-end",
        type=float,
        default=-2000.0,
        help="Non-landing (truncation) penalty by the end of training. Ramped "
        "linearly from --penalty-start so a 'hover forever, never land' "
        "equilibrium keeps getting worse even after the policy settles into it.",
    )
    parser.add_argument("--out", default="ppo_lunar_lander.zip")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=500_000,
        help="Save a checkpoint every N timesteps (total across all envs), so a "
        "long unattended run doesn't lose everything on a crash/interruption.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing model .zip to load and continue training "
        "(e.g. a checkpoint) instead of starting from scratch.",
    )
    args = parser.parse_args()

    if args.scenario == "lunar_descent":
        dt = args.dt if args.dt is not None else 1.0
        max_time = args.max_time if args.max_time is not None else 600.0
        descent_horizontal_speed = (
            args.descent_horizontal_speed if args.descent_horizontal_speed is not None else 50.0
        )
        max_turn_rate_deg = args.max_turn_rate_deg if args.max_turn_rate_deg is not None else 15.0
        ent_coef = args.ent_coef if args.ent_coef is not None else 0.02
    else:
        dt = args.dt if args.dt is not None else 5.0
        max_time = args.max_time if args.max_time is not None else 4 * 86400.0
        descent_horizontal_speed = args.descent_horizontal_speed or 0.0
        max_turn_rate_deg = args.max_turn_rate_deg if args.max_turn_rate_deg is not None else 2.0
        ent_coef = args.ent_coef if args.ent_coef is not None else 0.0

    env_fns = [
        make_env(
            dt,
            max_time,
            args.fuel_penalty_coef,
            args.scenario,
            args.descent_start_alt,
            descent_horizontal_speed,
            max_turn_rate_deg,
            args.safe_landing_speed,
            args.speed_penalty_coef,
            args.dense_speed_coef,
            args.auto_retrograde,
            args.penalty_start,
        )
        for _ in range(args.n_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if args.n_envs == 1 else SubprocVecEnv(env_fns)

    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            ent_coef=ent_coef,
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix=os.path.splitext(os.path.basename(args.out))[0],
    )
    clip_std_callback = ClipLogStdCallback(max_std=args.max_std)
    penalty_ramp_callback = PenaltyRampCallback(
        start=args.penalty_start,
        end=args.penalty_end,
        total_timesteps=model.num_timesteps + args.timesteps,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, clip_std_callback, penalty_ramp_callback],
        reset_num_timesteps=args.resume is None,
    )
    model.save(args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()
