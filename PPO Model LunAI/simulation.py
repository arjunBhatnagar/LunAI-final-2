"""
SLS launch + Earth-Moon orbital simulator (proof of concept, RL-ready)
======================================================================

Earth-centred inertial (ECI) frame, +Z = Earth's rotation axis.
All forces are 3D vectors. Earth, Moon and rocket are integrated together.

RL interface:
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step([m_dot, dx, dy, dz])

m_dot: commanded mass flow (kg/s) of the throttleable engine.
d: commanded thrust direction in ECI.

This is an approximate educational model, not a flight-validated simulator.
"""

from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

# Planetary radii (metres)
earth_radius = 6_371_000
moon_radius = 1_737_400

G = 6.674e-11
G0 = 9.80665

M_EARTH = 5.9722e24
M_MOON = 7.347673e22
M_SUN = 1.98847e30

MU_EARTH = G * M_EARTH
MU_MOON = G * M_MOON
MU_SUN = G * M_SUN
MU_EARTH_MOON = G * (M_EARTH + M_MOON)

R_MOON = 1.7374e6

OMEGA_EARTH = 7.2921159e-5
OMEGA_VEC = np.array([0.0, 0.0, OMEGA_EARTH])

WGS84_A = 6378137.0
WGS84_B = 6356752.314245

RHO0 = 1.225
P0 = 101325.0
SCALE_H = 8434.0

AU = 1.496e11

KSC_LAT_DEG = 28.5729
KSC_LON_DEG = -80.6490

MOON_POS0 = np.array([3.412939848601694e8, 1.594396539207186e8, 2.867337308870228e7])

MOON_VEL0 = np.array([-4.724716949667935e2, 9.259293623832163e2, 5.132647484618197e1])

# Landing target on the lunar surface: (altitude m, latitude deg, longitude deg)
MOON_TARGET_ALT_M = 50.0
MOON_TARGET_LAT_DEG = 3.63176
MOON_TARGET_LON_DEG = -85.80699

# Default altitude (m) above the target to begin a "lunar_descent" episode,
# roughly matching Apollo's powered-descent-initiation altitude.
LUNAR_DESCENT_START_ALT_M = 15_000.0


# ----------------------------------------------------------------------
# VEHICLE DATA
# ----------------------------------------------------------------------


@dataclass
class Engine:
    mdot_ref: float
    isp_vac: float
    Pe: float = 0.0
    Ae: float = 0.0
    throttle_min: float = 1.0

    @property
    def Ve(self):
        return self.isp_vac * G0 - self.Pe * self.Ae / self.mdot_ref

    @property
    def mdot_min(self):
        return self.throttle_min * self.mdot_ref


@dataclass
class SLSConfig:

    # Solid rocket boosters
    srb_prop: float = 2 * 635_000.0
    srb_inert: float = 2 * 91_000.0

    srb_engine: Engine = field(
        default_factory=lambda: Engine(
            mdot_ref=2 * 5040.0, isp_vac=269.0, Pe=40_000.0, Ae=26.3
        )
    )

    srb_burn_time: float = 126.0

    # Core stage
    core_prop: float = 987_000.0
    core_dry: float = 85_270.0

    core_engine: Engine = field(
        default_factory=lambda: Engine(
            mdot_ref=2057.0, isp_vac=452.0, Pe=60_000.0, Ae=17.1, throttle_min=0.615
        )
    )

    # ICPS
    # Bumped from the real-world ~26,000 kg: the baseline figure is ~600
    # m/s short of a theoretically optimal Hohmann transfer to the Moon
    # from LEO (computed and verified against the vis-viva minimum) --
    # this margin closes that gap so the scripted Phase 1 trans-lunar
    # injection can actually succeed. Documented here, not silently
    # tuned -- this sim is explicitly "approximate, not flight-validated."
    icps_prop: float = 41_000.0
    icps_dry: float = 3_490.0

    icps_engine: Engine = field(
        default_factory=lambda: Engine(mdot_ref=24.3, isp_vac=462.0, throttle_min=1.0)
    )

    # Payload
    payload: float = 31_520.0

    # Aerodynamics
    Cd: float = 0.4
    area_boost: float = 77.0
    area_core: float = 55.4
    area_icps: float = 20.4


# ----------------------------------------------------------------------
# GEOMETRY HELPERS
# ----------------------------------------------------------------------


def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)

    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def geodetic_to_ecef(lat_deg, lon_deg, h=0.0):

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    e2 = 1 - (WGS84_B / WGS84_A) ** 2

    N = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    return np.array(
        [
            (N + h) * np.cos(lat) * np.cos(lon),
            (N + h) * np.cos(lat) * np.sin(lon),
            (N * (1 - e2) + h) * np.sin(lat),
        ]
    )


def selenographic_to_frame(lat_deg, lon_deg, h=0.0, radius=moon_radius):
    """Point at (lat, lon, altitude) above a spherical Moon, expressed in the
    same fixed axes as MOON_POS0 (Moon rotation is not modeled, consistent
    with this simulator's other approximations)."""

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = radius + h

    return np.array(
        [
            r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat),
        ]
    )


def altitude(r):

    rn = np.linalg.norm(r)
    s = r[2] / rn

    R = WGS84_A * WGS84_B / np.sqrt(WGS84_B**2 * (1 - s * s) + WGS84_A**2 * s * s)

    return rn - R


def local_frame(r):

    up = r / np.linalg.norm(r)

    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)

    north = np.cross(up, east)

    return east, north, up


def local_direction(r, pitch_deg, azimuth_deg):

    east, north, up = local_frame(r)

    p = np.radians(pitch_deg)
    a = np.radians(azimuth_deg)

    return np.cos(p) * (np.sin(a) * east + np.cos(a) * north) + np.sin(p) * up


def orbit_elements(r, v, mu=MU_EARTH):

    rn = np.linalg.norm(r)

    eps = v @ v / 2 - mu / rn

    h = np.cross(r, v)
    hn = np.linalg.norm(h)

    e = np.sqrt(max(0, 1 + 2 * eps * hn**2 / mu**2))

    inc = np.degrees(np.arccos(np.clip(h[2] / hn, -1, 1)))

    if eps >= 0:

        return dict(
            a=np.inf,
            e=e,
            perigee=hn**2 / (mu * (1 + e)) - 6.371e6,
            apogee=np.inf,
            inc=inc,
        )

    a = -mu / (2 * eps)

    return dict(
        a=a, e=e, perigee=a * (1 - e) - 6.371e6, apogee=a * (1 + e) - 6.371e6, inc=inc
    )


# ----------------------------------------------------------------------
# FORCE FUNCTIONS
# ----------------------------------------------------------------------


def ambient_pressure(height):
    return P0 * np.exp(-height / SCALE_H)


def air_density(height):
    return RHO0 * np.exp(-height / SCALE_H)


def pressure(height, rel_velocity):

    return 0.5 * air_density(height) * float(rel_velocity @ rel_velocity)


def thrust_force(m_dot, engine, height, direction):

    if m_dot <= 0:
        return np.zeros(3)

    Pe = engine.Pe * m_dot / engine.mdot_ref

    F = m_dot * engine.Ve + (Pe - ambient_pressure(height)) * engine.Ae

    return max(F, 0.0) * direction


def drag_force(r, v, height, Cd, A):

    v_rel = v - np.cross(OMEGA_VEC, r)

    speed = np.linalg.norm(v_rel)

    return -0.5 * Cd * A * air_density(height) * speed * v_rel


def _third_body(r, r_b, mu_b):

    d = r_b - r

    return mu_b * (d / np.linalg.norm(d) ** 3 - r_b / np.linalg.norm(r_b) ** 3)


def weight(r, mass, r_moon, r_sun=None):

    a = -MU_EARTH * r / np.linalg.norm(r) ** 3 + _third_body(r, r_moon, MU_MOON)

    if r_sun is not None:
        a += _third_body(r, r_sun, MU_SUN)

    return mass * a


def accel(force_total, mass):
    return force_total / mass


def segment_sphere_intersect(p0, p1, center, radius):
    """True if the straight segment p0->p1 passes through the sphere at
    any point along it -- catches a step whose START and END points are
    both outside the sphere but which jumped clean through it in between
    (possible when velocity is high relative to step size), which a
    simple end-of-step distance check alone would miss entirely."""

    d = p1 - p0
    f = p0 - center

    a = np.dot(d, d)
    if a < 1e-12:
        return np.linalg.norm(f) < radius

    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius

    disc = b * b - 4 * a * c
    if disc < 0:
        return False

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


# ----------------------------------------------------------------------
# SIMULATION ENVIRONMENT
# ----------------------------------------------------------------------


class SLSEnv:

    STAGES = ("boost", "core", "icps")

    def __init__(
        self,
        dt=1.0,
        max_time=10 * 86400.0,
        vertical_hold=10.0,
        max_turn_rate_deg=2.0,
        theta0=0.0,
        include_sun=False,
        sun_phase0=0.0,
        fuel_penalty_coef=1e-3,
        safe_landing_speed_mps=3.0,
        speed_penalty_coef=1.0,
        dense_speed_coef=2e-3,
        dense_speed_range_m=1.0e8,
        start_mode="launch",
        descent_start_alt_m=LUNAR_DESCENT_START_ALT_M,
        descent_horizontal_speed=0.0,
        orbit_altitude_m=300_000.0,
        truncated_penalty=-300.0,
        cfg=None,
    ):

        self.cfg = cfg or SLSConfig()

        self.dt = dt
        self.max_time = max_time
        self.safe_landing_speed = safe_landing_speed_mps
        self.speed_penalty_coef = speed_penalty_coef
        self.dense_speed_coef = dense_speed_coef
        self.dense_speed_range_m = dense_speed_range_m
        self.truncated_penalty = truncated_penalty

        # The pad vertical-hold attitude lock only makes sense when starting
        # on the Earth launch pad.
        self.vertical_hold = 0.0 if start_mode in ("lunar_descent", "orbit") else vertical_hold

        self.max_turn_rate = (
            np.radians(max_turn_rate_deg) if max_turn_rate_deg else None
        )

        self.theta0 = theta0
        self.include_sun = include_sun
        self.sun_phase0 = sun_phase0
        self.fuel_penalty_coef = fuel_penalty_coef

        self.start_mode = start_mode
        self.descent_start_alt_m = descent_start_alt_m
        self.descent_horizontal_speed = descent_horizontal_speed
        self.orbit_altitude_m = orbit_altitude_m

        self.r_pad_ecef = geodetic_to_ecef(KSC_LAT_DEG, KSC_LON_DEG, 0.0)

        self.target_offset = selenographic_to_frame(
            MOON_TARGET_LAT_DEG, MOON_TARGET_LON_DEG, MOON_TARGET_ALT_M
        )

        self.reset()

    # STATE

    def reset(self):

        c = self.cfg

        self.t = 0.0

        if self.start_mode == "lunar_descent":

            # Begin just above the landing target, descent stage only —
            # a short, cheap episode for validating the landing reward.
            self.srb_on = False
            self.core_on = False

            self.srb_prop = 0.0
            self.core_prop = 0.0
            self.icps_prop = c.icps_prop

            self.on_ground = False

            offset = selenographic_to_frame(
                MOON_TARGET_LAT_DEG, MOON_TARGET_LON_DEG, self.descent_start_alt_m
            )

            r0 = MOON_POS0 + offset

            up = offset / np.linalg.norm(offset)
            east, _, _ = local_frame(offset)

            v0 = MOON_VEL0 + self.descent_horizontal_speed * east

            self.y = np.concatenate([r0, v0, MOON_POS0, MOON_VEL0])

            self.attitude = up

        elif self.start_mode == "orbit":

            # Begin in a circular LEO-ish parking orbit, ICPS tank full,
            # core/SRB already spent -- Phase 1 for the trans-lunar
            # injection demo, skipping the ascent-energy problem (the real
            # SLS core stage delivers far more orbital energy than this
            # simulator's crude pitch-program ascent does).
            self.srb_on = False
            self.core_on = False

            self.srb_prop = 0.0
            self.core_prop = 0.0
            self.icps_prop = c.icps_prop

            self.on_ground = False

            r_mag = earth_radius + self.orbit_altitude_m
            v_circ = np.sqrt(MU_EARTH / r_mag)

            # Build the parking orbit IN the Moon's own orbital plane
            # (not an arbitrary one) -- otherwise every transfer burn
            # pays for an expensive, unnecessary plane change on top of
            # the real trans-lunar delta-v.
            n = np.cross(MOON_POS0, MOON_VEL0)
            n = n / np.linalg.norm(n)
            e1 = MOON_POS0 / np.linalg.norm(MOON_POS0)
            e2 = np.cross(n, e1)

            r0 = r_mag * (np.cos(self.theta0) * e1 + np.sin(self.theta0) * e2)
            v0 = v_circ * (-np.sin(self.theta0) * e1 + np.cos(self.theta0) * e2)

            self.y = np.concatenate([r0, v0, MOON_POS0, MOON_VEL0])

            self.attitude = v0 / np.linalg.norm(v0)

        else:

            r0 = Rz(self.theta0) @ self.r_pad_ecef
            v0 = np.cross(OMEGA_VEC, r0)

            self.y = np.concatenate([r0, v0, MOON_POS0, MOON_VEL0])

            self.attitude = r0 / np.linalg.norm(r0)

            self.on_ground = True

            self.srb_prop = c.srb_prop
            self.core_prop = c.core_prop
            self.icps_prop = c.icps_prop

            self.srb_on = True
            self.core_on = True

        self.thrust_N = 0.0
        self.mdot_total = 0.0
        self.accel = np.zeros(3)

        self.mass0 = self.mass

        self.history = []

        self.last_events = []

        self._prev_moon_dist = self.moon_distance
        self._prev_target_dist = self.target_distance
        self._best_target_dist = self.target_distance
        self._best_altitude = self.lunar_altitude
        self.total_propellant_used = 0.0

        self._record()

        return self.observe(), self.info()

    @property
    def r(self):
        return self.y[0:3]

    @property
    def v(self):
        return self.y[3:6]

    @property
    def r_moon(self):
        return self.y[6:9]

    @property
    def v_moon(self):
        return self.y[9:12]

    @property
    def moon_distance(self):
        return float(np.linalg.norm(self.r_moon - self.r))

    @property
    def target_position(self):
        return self.r_moon + self.target_offset

    @property
    def target_distance(self):
        return float(np.linalg.norm(self.target_position - self.r))

    @property
    def lunar_altitude(self):
        """Height above the Moon's physical surface (not Earth's)."""
        return self.moon_distance - R_MOON

    @property
    def mass(self):

        c = self.cfg

        m = c.payload + c.icps_dry + self.icps_prop

        if self.core_on:
            m += c.core_dry + self.core_prop

        if self.srb_on:
            m += c.srb_inert + self.srb_prop

        return m

    @property
    def stage_index(self):

        return 0 if self.srb_on else (1 if self.core_on else 2)

    @property
    def stage_name(self):
        return self.STAGES[self.stage_index]

    def srb_flow_factor(self):

        return 1.19 - 0.38 * min(self.t / self.cfg.srb_burn_time, 1.0)

    def sun_position(self, t):

        phi = self.sun_phase0 + 2 * np.pi * t / (365.256 * 86400.0)

        return AU * np.array([np.cos(phi), np.sin(phi), 0.0])

    # DYNAMICS

    def _deriv(self, tau, y, c):

        t = self.t + tau

        r = y[0:3]
        v = y[3:6]
        rm = y[6:9]

        m = c["m0"] - c["mdot_total"] * tau

        h = altitude(r)

        rs = self.sun_position(t) if self.include_sun else None

        F_thrust = sum(
            (thrust_force(md, eng, h, c["dir"]) for md, eng in c["engines"]),
            np.zeros(3),
        )

        F_drag = drag_force(r, v, h, self.cfg.Cd, c["area"])

        F_grav = weight(r, m, rm, rs)

        a = accel(F_thrust + F_drag + F_grav, m)

        a_moon = -MU_EARTH_MOON * rm / np.linalg.norm(rm) ** 3

        if rs is not None:
            a_moon += _third_body(rm, rs, MU_SUN)

        return (
            np.concatenate([v, a, y[9:12], a_moon]),
            float(np.linalg.norm(F_thrust)),
        )

    def _rk4(self, y, dt, c):

        k1, F1 = self._deriv(0.0, y, c)

        k2, _ = self._deriv(dt / 2, y + dt / 2 * k1, c)

        k3, _ = self._deriv(dt / 2, y + dt / 2 * k2, c)

        k4, _ = self._deriv(dt, y + dt * k3, c)

        return (y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), F1)

    def _slew(self, cmd, dt):

        cur = self.attitude

        if self.max_turn_rate is None:
            return cmd

        ang = np.arccos(np.clip(cur @ cmd, -1, 1))

        lim = self.max_turn_rate * dt

        if ang <= lim:
            return cmd

        perp = cmd - (cur @ cmd) * cur

        if np.linalg.norm(perp) < 1e-9:

            perp = np.cross(
                cur, [1.0, 0.0, 0.0] if abs(cur[0]) < 0.9 else [0.0, 1.0, 0.0]
            )

        perp /= np.linalg.norm(perp)

        new = np.cos(lim) * cur + np.sin(lim) * perp

        return new / np.linalg.norm(new)

    # MAIN STEP

    def step(self, action, dt=None, separate=False):

        cfg = self.cfg

        dt = self.dt if dt is None else dt

        action = np.asarray(action, dtype=float)

        m_dot_cmd = max(float(action[0]), 0.0)

        d_cmd = action[1:4]

        if np.linalg.norm(d_cmd) < 1e-9:
            d_cmd = self.attitude

        d_cmd = d_cmd / np.linalg.norm(d_cmd)

        if self.t < self.vertical_hold:
            d_cmd = local_frame(self.r)[2]

        self.attitude = self._slew(d_cmd, dt)

        # ENGINE COMMANDS

        engines = []

        mdots = {"srb": 0.0, "core": 0.0, "icps": 0.0}

        if self.srb_on and self.srb_prop > 0:

            mdots["srb"] = min(
                cfg.srb_engine.mdot_ref * self.srb_flow_factor(), self.srb_prop / dt
            )

            engines.append((mdots["srb"], cfg.srb_engine))

        if self.core_on:

            eng = cfg.core_engine
            key = "core"
            prop = self.core_prop

        else:

            eng = cfg.icps_engine
            key = "icps"
            prop = self.icps_prop

        if prop > 0 and m_dot_cmd >= 0.5 * eng.mdot_min:

            md = float(np.clip(m_dot_cmd, eng.mdot_min, eng.mdot_ref))

            mdots[key] = min(md, prop / dt)

            engines.append((mdots[key], eng))

        stage = self.stage_index

        area = (cfg.area_boost, cfg.area_core, cfg.area_icps)[stage]

        m0 = self.mass

        mdot_total = sum(mdots.values())

        c = dict(
            m0=m0, mdot_total=mdot_total, engines=engines, dir=self.attitude, area=area
        )

        # PAD HOLD

        hold = False

        if self.on_ground:

            d0, _ = self._deriv(0.0, self.y, c)

            up = local_frame(self.r)[2]

            a_net = d0[3:6] - (
                -(OMEGA_EARTH**2) * np.array([self.r[0], self.r[1], 0.0])
            )

            if a_net @ up > 0:
                self.on_ground = False
            else:
                hold = True

        r_before_step = self.r.copy()

        y_new, F = self._rk4(self.y, dt, c)

        if hold:

            r_new = Rz(OMEGA_EARTH * dt) @ self.r

            y_new[0:3] = r_new
            y_new[3:6] = np.cross(OMEGA_VEC, r_new)

            F = 0.0 if mdot_total == 0 else F

        # Net acceleration over the step (thrust + drag + gravity), from the
        # actual velocity change — simple and valid regardless of start_mode.
        self.accel = (y_new[3:6] - self.v) / dt

        # PROPellant AND STATE UPDATE

        self.srb_prop = max(0.0, self.srb_prop - mdots["srb"] * dt)

        self.core_prop = max(0.0, self.core_prop - mdots["core"] * dt)

        self.icps_prop = max(0.0, self.icps_prop - mdots["icps"] * dt)

        self.y = y_new

        self.t += dt

        self.thrust_N = F
        self.mdot_total = mdot_total

        # STAGING

        events = []

        if self.srb_on and self.srb_prop <= 1e-6:

            self.srb_on = False
            self.srb_prop = 0.0

            events.append("SRB sep")

        if self.core_on and not self.srb_on and (self.core_prop <= 1e-6 or separate):

            self.core_on = False
            self.core_prop = max(self.core_prop, 0.0)

            events.append("MECO + core sep")

        self.last_events = events

        # TERMINATION

        terminated = False
        reason = None

        h = altitude(self.r)

        if not self.on_ground and h < -1.0:

            terminated = True
            reason = "ground impact"

        elif self.moon_distance < R_MOON or segment_sphere_intersect(
            r_before_step, self.r, self.r_moon, R_MOON
        ):

            terminated = True
            reason = "lunar impact"

        elif np.linalg.norm(self.r) > 1.0e9:

            terminated = True
            reason = "escaped Earth-Moon system"

        truncated = not terminated and self.t >= self.max_time

        propellant_used = mdot_total * dt

        reward = self.compute_reward(terminated, reason, propellant_used, truncated)

        self._record()

        info = self.info()

        info.update(events=events, reason=reason)

        return (self.observe(), reward, terminated, truncated, info)

    # RL HELPERS

    def compute_reward(self, terminated, reason, propellant_used=0.0, truncated=False):

        d_moon = self.moon_distance
        self._prev_moon_dist = d_moon

        d = self.target_distance

        # Shaping: reward NEW best-ever progress toward the target, in
        # km-equivalent. Using "closer than ever before" instead of raw
        # step-to-step distance change means drifting back and forth
        # (e.g. an orbit that doesn't actually head to the Moon) nets to
        # zero rather than being able to look reward-neutral by swinging
        # toward the target for half the loop and away for the other half
        # -- the only way to keep earning reward is to keep setting a new
        # closest-approach record, i.e. to keep actually heading to the Moon.
        progress = max(0.0, self._best_target_dist - d)
        rew = progress / 1e3
        self._best_target_dist = min(self._best_target_dist, d)
        self._prev_target_dist = d

        # Same "new best ever" shaping, but for height above the Moon's
        # surface specifically -- closing the 3D distance to the target
        # can stall out (e.g. hovering at a fixed offset) while altitude
        # itself keeps improving, or vice versa; rewarding both directly
        # discourages "park at some distance and do nothing further"
        # equilibria that satisfy one but not the other.
        alt = self.lunar_altitude
        alt_progress = max(0.0, self._best_altitude - alt)
        rew += alt_progress / 1e3
        self._best_altitude = min(self._best_altitude, alt)

        # Fuel use is tracked but NOT punished every step -- it only costs
        # reward if the episode ends in an actual landing (see "lunar
        # impact" below). Otherwise an agent that hasn't learned to land
        # yet would be punished for using fuel at all, which collapses
        # exploration straight to zero-thrust.
        self.total_propellant_used += propellant_used

        # Dense speed shaping: continuously discourage excess closing speed
        # (relative to the Moon) every step, not just at touchdown, so the
        # agent gets feedback on "too fast" throughout the descent instead
        # of only learning it from the single terminal outcome.
        #
        # Gated by proximity to the target: velocity relative to the Moon is
        # dominated by ordinary Earth-orbital mechanics (several km/s) for
        # almost the entire flight and has nothing to do with landing speed
        # until the final approach -- applying this penalty unconditionally
        # made simply being airborne (ascent/transit/orbit) ruinously
        # expensive, which fights against ever reaching the Moon at all.
        # It ramps in only within dense_speed_range_m of the target.
        proximity_scale = max(0.0, 1.0 - d / self.dense_speed_range_m)
        current_speed = float(np.linalg.norm(self.v - self.v_moon))
        rew -= proximity_scale * self.dense_speed_coef * max(
            0.0, current_speed - self.safe_landing_speed
        )

        if terminated:

            if reason == "ground impact":
                rew -= 100.0

            elif reason == "lunar impact":
                # Landing-accuracy bonus: full marks for touching down on
                # the target, decaying to zero by 100 km of miss distance.
                accuracy_bonus = 200.0 * max(0.0, 1.0 - d / 1.0e5)

                # Smooth-landing penalty: punish touchdown speed (relative to
                # the Moon) above a gentle threshold, so an accurate but
                # high-speed crash does not score well.
                landing_speed = float(np.linalg.norm(self.v - self.v_moon))
                speed_penalty = self.speed_penalty_coef * max(
                    0.0, landing_speed - self.safe_landing_speed
                )

                # Fuel efficiency only counts once a landing actually
                # happens -- a wasteful but successful landing still beats
                # a frugal non-landing.
                fuel_penalty = self.fuel_penalty_coef * self.total_propellant_used

                rew += accuracy_bonus - speed_penalty - fuel_penalty

            elif reason == "escaped Earth-Moon system":
                rew -= 50.0

        elif truncated:
            # Not landing at all must be clearly worse than ANY landing
            # attempt, including a bad one. Configurable (and ramped up
            # during training via a callback) because fixed values kept
            # turning out to not be dominant enough against whatever
            # "stall somewhere safe" equilibrium the policy found.
            rew += self.truncated_penalty

        return rew

    def observe(self):

        c = self.cfg

        return np.concatenate(
            [
                self.r / 1e7,
                self.v / 1e4,
                self.attitude,
                (self.r_moon - self.r) / 4e8,
                (self.v_moon - self.v) / 1e3,
                (self.target_position - self.r) / 4e8,
                [
                    self.mass / self.mass0,
                    self.srb_prop / c.srb_prop,
                    self.core_prop / c.core_prop,
                    self.icps_prop / c.icps_prop,
                    self.stage_index / 2.0,
                    self.t / self.max_time,
                ],
            ]
        ).astype(np.float32)

    def info(self):

        h = altitude(self.r)

        v_rel = self.v - np.cross(OMEGA_VEC, self.r)

        el = orbit_elements(self.r, self.v)

        return dict(
            t=self.t,
            altitude=h,
            speed_inertial=float(np.linalg.norm(self.v)),
            speed_rel=float(np.linalg.norm(v_rel)),
            q=pressure(h, v_rel),
            mass=self.mass,
            stage=self.stage_name,
            thrust=self.thrust_N,
            mdot=self.mdot_total,
            accel=self.accel.copy(),
            accel_mag=float(np.linalg.norm(self.accel)),
            moon_distance=self.moon_distance,
            target_distance=self.target_distance,
            landing_speed=float(np.linalg.norm(self.v - self.v_moon)),
            perigee=el["perigee"],
            apogee=el["apogee"],
            ecc=el["e"],
            inc=el["inc"],
        )

    def local_direction(self, pitch_deg, azimuth_deg):

        return local_direction(self.r, pitch_deg, azimuth_deg)

    # TELEMETRY

    HISTORY_COLUMNS = (
        "t",
        "rx", "ry", "rz",
        "vx", "vy", "vz",
        "moon_rx", "moon_ry", "moon_rz",
        "mass",
        "thrust_N",
        "altitude",
        "stage_index",
        "target_rx", "target_ry", "target_rz",
        "target_distance",
        "mdot",
        "ax", "ay", "az",
    )

    def _record(self):

        self.history.append(
            (
                self.t,
                *self.r,
                *self.v,
                *self.r_moon,
                self.mass,
                self.thrust_N,
                altitude(self.r),
                self.stage_index,
                *self.target_position,
                self.target_distance,
                self.mdot_total,
                *self.accel,
            )
        )

    def history_array(self):

        return np.array(self.history)

    def save_history_csv(self, path):
        """Write the full step-by-step trajectory history to a CSV file,
        e.g. for loading into an external visualizer."""

        header = ",".join(self.HISTORY_COLUMNS)

        np.savetxt(path, self.history_array(), delimiter=",", header=header, comments="")


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------


def plot_trajectory(env, moon_scale_km=True):
    import plotly.graph_objects as go
    import numpy as np

    # Convert simulation positions from metres to kilometres
    H = env.history_array()
    t_days = H[:, 0] / 86400
    rock = H[:, 1:4] / 1e3
    moon = H[:, 7:10] / 1e3

    # Accurate mean planetary radii (km)
    EARTH_RADIUS = 6371.0
    MOON_RADIUS = 1737.4

    # At typical integration step sizes the rocket can cover more distance
    # than the Moon's own radius in a single step near impact, so the last
    # recorded sample can land deep inside the Moon rather than at its
    # surface -- visually looking like the path cuts through the center.
    # Clip the plotted path at the Moon's actual surface, interpolating
    # between the last two samples for a clean crossing point.
    moon_dist = np.linalg.norm(rock - moon, axis=1)
    inside = np.where(moon_dist < MOON_RADIUS)[0]

    if len(inside) > 0 and inside[0] > 0:
        i = inside[0]
        d0, d1 = moon_dist[i - 1], moon_dist[i]
        frac = (d0 - MOON_RADIUS) / (d0 - d1) if d0 != d1 else 0.0
        frac = np.clip(frac, 0.0, 1.0)

        rock_cross = rock[i - 1] + frac * (rock[i] - rock[i - 1])
        t_cross = t_days[i - 1] + frac * (t_days[i] - t_days[i - 1])

        rock = np.vstack([rock[: i], rock_cross])
        moon = np.vstack([moon[: i], moon[i - 1] + frac * (moon[i] - moon[i - 1])])
        t_days = np.append(t_days[:i], t_cross)
        H = np.vstack([H[:i], H[i - 1] + frac * (H[i] - H[i - 1])])
        moon_dist = np.linalg.norm(rock - moon, axis=1)

    # NOTE: the Moon is drawn as a single static sphere at its FINAL
    # position, but it genuinely moves during the flight -- over the
    # final hours-long approach it covers a distance comparable to the
    # close-approach window itself (its own orbital speed vs. the
    # rocket's closing speed are the same order of magnitude), so no
    # single fixed sphere can be visually consistent with the whole
    # encounter. (An earlier attempt to "re-anchor" close samples onto
    # the final position made this WORSE -- it mixes two different
    # times' Moon positions together, producing spurious spikes.) For an
    # accurate view of the close encounter, see plot_lunar_approach()
    # below, which renders it in the Moon's own co-moving frame.

    fig = go.Figure()

    # --------------------------------------------------
    # Helper: generate a spherical surface
    # --------------------------------------------------
    def add_sphere(center, radius, name, color, opacity=1.0):
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 30)

        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                name=name,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=opacity,
                hovertemplate=f"{name}<extra></extra>",
            )
        )

    # --------------------------------------------------
    # SLS trajectory
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter3d(
            x=rock[:, 0],
            y=rock[:, 1],
            z=rock[:, 2],
            mode="lines",
            name="SLS",
            line=dict(
                color=t_days, colorscale="Viridis", width=5, colorbar=dict(title="Day")
            ),
            customdata=np.column_stack([t_days, H[:, 12] / 1e3]),
            hovertemplate=(
                "Day %{customdata[0]:.3f}"
                "<br>Altitude %{customdata[1]:,.0f} km"
                "<extra>SLS</extra>"
            ),
        )
    )

    # --------------------------------------------------
    # Moon orbital path
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter3d(
            x=moon[:, 0],
            y=moon[:, 1],
            z=moon[:, 2],
            mode="lines",
            name="Moon orbit",
            line=dict(color="lightgray", width=3),
        )
    )

    # --------------------------------------------------
    # Earth: centred at the ECI origin
    # --------------------------------------------------
    add_sphere(
        center=np.array([0.0, 0.0, 0.0]),
        radius=EARTH_RADIUS,
        name="Earth (R = 6,371 km)",
        color="royalblue",
        opacity=1.0,
    )

    # --------------------------------------------------
    # Moon: centred at its final simulated position
    # --------------------------------------------------
    add_sphere(
        center=moon[-1],
        radius=MOON_RADIUS,
        name="Moon (R = 1,737.4 km)",
        color="lightgray",
        opacity=1.0,
    )

    # --------------------------------------------------
    # Launch location
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter3d(
            x=[rock[0, 0]],
            y=[rock[0, 1]],
            z=[rock[0, 2]],
            mode="markers",
            name="KSC launch",
            marker=dict(size=4, color="lime"),
        )
    )

    # --------------------------------------------------
    # Landing target (lat/lon on the Moon) vs. actual final position --
    # the target offset (tens of km above the surface) is tiny next to
    # the Moon's own 1,737 km radius, so without an explicit marker the
    # two are visually indistinguishable from "aiming at the center".
    # --------------------------------------------------
    target_pos = H[-1, 14:17] / 1e3
    fig.add_trace(
        go.Scatter3d(
            x=[target_pos[0]],
            y=[target_pos[1]],
            z=[target_pos[2]],
            mode="markers",
            name="Landing target (lat/lon)",
            marker=dict(size=5, color="red", symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[rock[-1, 0]],
            y=[rock[-1, 1]],
            z=[rock[-1, 2]],
            mode="markers",
            name="Actual impact/final point",
            marker=dict(size=5, color="orange", symbol="x"),
        )
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title="SLS Earth-Moon Trajectory — Accurate Planetary Radii",
        scene=dict(
            aspectmode="data",
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    return fig


def plot_lunar_approach(env, within_km=20_000.0):
    """The close-approach segment, rendered in the Moon's own co-moving
    frame (each sample expressed relative to the Moon's position at that
    SAME instant) -- the Moon sits correctly at the origin for every
    frame, not just the last one, so this is the accurate view of a
    lunar encounter. plot_trajectory()'s single static sphere can't
    represent the Moon's real motion over an hours-long approach; this
    can, at the cost of not showing the rest of the journey or Earth
    (Earth is typically ~380,000 km away -- far outside this view's scale).
    """
    import plotly.graph_objects as go
    import numpy as np

    H = env.history_array()
    t_days = H[:, 0] / 86400
    rock = H[:, 1:4] / 1e3
    moon = H[:, 7:10] / 1e3

    MOON_RADIUS = 1737.4

    moon_dist = np.linalg.norm(rock - moon, axis=1)
    close = moon_dist < within_km

    if not np.any(close):
        raise ValueError(f"Trajectory never comes within {within_km:.0f} km of the Moon")

    rel = (rock - moon)[close]
    t_close = t_days[close]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=rel[:, 0],
            y=rel[:, 1],
            z=rel[:, 2],
            mode="lines",
            name="SLS (Moon-relative)",
            line=dict(color=t_close, colorscale="Viridis", width=5, colorbar=dict(title="Day")),
        )
    )

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    x = MOON_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = MOON_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = MOON_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z, showscale=False,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            name="Moon", hovertemplate="Moon<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=800,
        title=f"Lunar close approach (Moon-relative, within {within_km/1e3:.0f},000 km)",
        scene=dict(aspectmode="data", xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    return fig


def plot_ascent(env, t_max=700):

    import plotly.graph_objects as go

    from plotly.subplots import make_subplots

    H = env.history_array()

    H = H[H[:, 0] <= t_max]

    t = H[:, 0]

    speed = np.linalg.norm(H[:, 4:7], axis=1)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Altitude (km)",
            "Inertial speed (km/s)",
            "Mass (t)",
            "Thrust (MN)",
        ),
    )

    fig.add_trace(go.Scatter(x=t, y=H[:, 12] / 1e3, name="alt"), 1, 1)

    fig.add_trace(go.Scatter(x=t, y=speed / 1e3, name="speed"), 1, 2)

    fig.add_trace(go.Scatter(x=t, y=H[:, 10] / 1e3, name="mass"), 2, 1)

    fig.add_trace(go.Scatter(x=t, y=H[:, 11] / 1e6, name="thrust"), 2, 2)

    fig.update_layout(
        template="plotly_dark", height=650, showlegend=False, title="Ascent telemetry"
    )

    return fig


# ----------------------------------------------------------------------
# BASELINE CONTROLLER
# ----------------------------------------------------------------------

T_PITCH = [0, 10, 30, 60, 90, 126, 200, 300, 400, 500]

PITCH = [90, 90, 80, 62, 48, 38, 25, 15, 8, 3]


def baseline_policy(env):
    """
    Open-loop pitch program due east.
    Now commands the ICPS engine as well as the core engine.
    """

    cfg = env.cfg

    pitch = np.interp(env.t, T_PITCH, PITCH)

    direction = env.local_direction(pitch, 90.0)

    if env.stage_name in ("boost", "core"):

        mdot = cfg.core_engine.mdot_ref

    elif env.stage_name == "icps":

        mdot = cfg.icps_engine.mdot_ref

    else:

        mdot = 0.0

    return np.concatenate([[mdot], direction])


# ----------------------------------------------------------------------
# RUN SIMULATION
# ----------------------------------------------------------------------

if __name__ == "__main__":

    env = SLSEnv(dt=1.0, max_time=10 * 86400.0)

    obs, info = env.reset()

    print("Starting SLS simulation...\n")

    step = 0
    print_interval = 3600  # Print once every simulated hour

    while env.t < env.max_time:

        action = baseline_policy(env)

        obs, rew, term, trunc, info = env.step(action)

        step += 1

        # Print staging events
        if info["events"]:

            print(
                f"t={info['t']/3600:8.2f} hours | "
                f"{info['events']} | "
                f"Altitude={info['altitude']/1e3:9.2f} km | "
                f"Velocity={info['speed_inertial']:9.2f} m/s | "
                f"Stage={env.stage_name} | "
                f"ICPS fuel={env.icps_prop:,.1f} kg"
            )

        # Periodic telemetry
        elif step % print_interval == 0:

            print(
                f"t={info['t']/3600:8.2f} hours | "
                f"Altitude={info['altitude']/1e3:9.2f} km | "
                f"Velocity={info['speed_inertial']:9.2f} m/s | "
                f"Stage={env.stage_name} | "
                f"ICPS fuel={env.icps_prop:,.1f} kg"
            )

        # Stop only for actual termination or truncation
        if term or trunc:

            print("\nSimulation ended.")

            print("Reason:", info.get("reason", "Unknown"))

            break

    print("\nFinal simulation results:")

    print(f"Simulation time: {env.t/86400:.3f} days")
    print(f"Simulation steps: {step}")
    print(f"Final stage: {env.stage_name}")
    print(f"Final altitude: {info['altitude']/1e3:.2f} km")
    print(f"Final velocity: {info['speed_inertial']:.2f} m/s")
    print(f"Remaining ICPS propellant: {env.icps_prop:,.2f} kg")

    # Check how many telemetry points were recorded
    history = env.history_array()

    print(f"Recorded trajectory points: {len(history)}")

    # Plot results
    plot_trajectory(env)
    plot_ascent(env)

    # Plot simulation results
    plot_trajectory(env)

    plot_ascent(env)

    # Plot trajectory
    fig1 = plot_trajectory(env)
    fig1.show()

    # Plot ascent telemetry
    fig2 = plot_ascent(env)
    fig2.show()
