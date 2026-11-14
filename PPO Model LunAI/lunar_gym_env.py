"""Gymnasium wrapper around SLSEnv so it can be trained with stable-baselines3 PPO.

Action = [throttle (0-1), dx, dy, dz] where throttle is a fraction of the
active stage's reference mass flow, and (dx, dy, dz) is a thrust direction
in ECI (normalized inside SLSEnv.step).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation import SLSEnv


class LunarLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=5.0,
        max_time=4 * 86400.0,
        fuel_penalty_coef=1e-3,
        auto_retrograde=False,
        scripted_throttle=False,
        target_sink_rate=-2.0,
        **sls_kwargs,
    ):
        super().__init__()

        self.sim = SLSEnv(
            dt=dt,
            max_time=max_time,
            fuel_penalty_coef=fuel_penalty_coef,
            **sls_kwargs,
        )

        self.auto_retrograde = auto_retrograde

        # Hardcoded final-approach rule: once the agent is *not* sinking at
        # least this fast (m/s, negative = descending), cut the engine and
        # let gravity pull it down; once it's sinking faster than this,
        # fire the engine to brake. Converges to a steady, safe descent
        # instead of hovering indefinitely.
        self.scripted_throttle = scripted_throttle
        self.target_sink_rate = target_sink_rate

        obs_dim = self.sim.observe().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        if auto_retrograde:
            # Agent only controls throttle; thrust direction is locked to
            # anti-velocity (relative to the Moon) by the environment. Used
            # to isolate whether PPO can learn the throttle duty-cycle
            # without also having to learn attitude pointing.
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self._mdot_ref = max(
            self.sim.cfg.core_engine.mdot_ref,
            self.sim.cfg.icps_engine.mdot_ref,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.sim.reset()

    def set_truncated_penalty(self, value):
        """Called via VecEnv.env_method() by a training-time ramp callback."""
        self.sim.truncated_penalty = value

    def _up_vector(self):
        up = self.sim.r - self.sim.r_moon
        return up / np.linalg.norm(up)

    def _retrograde_direction(self):
        v_rel = self.sim.v - self.sim.v_moon
        speed = np.linalg.norm(v_rel)

        if speed < 1e-6:
            return self._up_vector()

        return -v_rel / speed

    def _sink_rate(self):
        v_rel = self.sim.v - self.sim.v_moon
        return float(np.dot(v_rel, self._up_vector()))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if self.scripted_throttle:
            throttle = 1.0 if self._sink_rate() < self.target_sink_rate else 0.0
        else:
            throttle = float(np.clip(action[0], 0.0, 1.0))

        m_dot_cmd = throttle * self._mdot_ref

        direction = self._retrograde_direction() if self.auto_retrograde else action[1:4]

        scaled_action = np.concatenate([[m_dot_cmd], direction])

        return self.sim.step(scaled_action)
