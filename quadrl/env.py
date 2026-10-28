"""A Gymnasium environment built on the same CasADi model LQR and MPC use.

One model, three controllers. The environment does not re-derive the
dynamics: it integrates ``sys.f`` from ``plants/quadrotor3d/model.py``, so a
difference between PPO and MPC in the results is a difference between the
controllers and not between two hand-written simulators.

Everything that a trained policy depends on at run time - the observation
map, the action map - is a module-level function, imported by both this file
and ``policy.py``. A policy trained on one scaling and deployed on another is
a classic silent failure; sharing the function makes it impossible.

The full MDP is written out, with the reasoning for each choice, in
DERIVATION.md section 7.
"""

from __future__ import annotations

import casadi as ca
import gymnasium as gym
import numpy as np

from .problem import hover, system

# ------------------------------------------------------------ shared maps

# Observation scale: each error component divided by a "typical large" value
# so every input to the network is O(1). Order follows the state vector.
OBS_SCALE = np.array([2, 2, 2,  2, 2, 2,  0.5, 0.5, 1.0,  5, 5, 5], dtype=float)
OBS_CLIP = 10.0
ANGLES = np.array([6, 7, 8])


def wrap(a):
    """Map an angle to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def observe(x, x_ref):
    """Policy input: the scaled state error, angles wrapped.

    The same error LQR feeds its gain, divided by OBS_SCALE. Because the
    dynamics do not depend on position, "fly to a waypoint" and "recover to
    the origin" are the same problem in error coordinates - one policy does
    both.
    """
    e = np.asarray(x, float) - np.asarray(x_ref, float)
    e[ANGLES] = wrap(e[ANGLES])
    return np.clip(e / OBS_SCALE, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def to_rotor(a, u_hover, u_max):
    """Action in [-1, 1]^4 -> rotor forces, centred on NOMINAL hover.

    a = 0 is hover, a = -1 is rotor off, a = +1 is twice hover thrust. Centring
    on hover means an untrained network (outputs near zero) already roughly
    holds altitude, instead of starting every episode in free fall.
    """
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    return np.clip(u_hover * (1.0 + a), 0.0, u_max)


# ------------------------------------------------------------ the environment

class QuadrotorEnv(gym.Env):
    """Fly from a random state to hover at the target.

    Parameters
    ----------
    dt : float
        Control period. 0.01 s (100 Hz).
    substeps : int
        RK4 steps inside one control period.
    horizon : float
        Episode length in seconds before truncation.
    randomize : bool
        Sample mass and yaw inertia from params.yaml's uncertainty ranges at
        every reset (domain randomization).
    tilt_limit : float
        The flight-envelope limit in radians, as in params.yaml. Penalised in
        the reward, never enforced - which is the honest difference from MPC.
    """

    metadata = {"render_modes": []}

    def __init__(self, dt=0.01, substeps=4, horizon=5.0, randomize=True,
                 tilt_limit=None, start_scale=1.0):
        super().__init__()
        self.sys = system()
        self.x_hover, u0 = hover(self.sys)
        self.u_hover = float(u0[0])
        self.u_max = float(self.sys.ulim[0, 1])
        self.tilt_limit = float(self.sys.xlim[7, 1] if tilt_limit is None else tilt_limit)
        self.dt, self.substeps = float(dt), int(substeps)
        self.max_steps = int(round(horizon / dt))
        self.randomize = bool(randomize)
        self.start_scale = float(start_scale)

        self.p_nominal = self.sys.pvec()
        self.pnames = list(self.sys.names["p"])
        ranges = (self.sys.uncertainty or {}).get("ranges", {})
        self.ranges = {k: tuple(map(float, v)) for k, v in ranges.items()}

        self._step_fn = _rk4(self.sys, self.dt, self.substeps)

        self.observation_space = gym.spaces.Box(-OBS_CLIP, OBS_CLIP, (12,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

        self.x = np.zeros(12)
        self.x_ref = np.zeros(12)
        self.p = self.p_nominal.copy()
        self.k = 0

    # -- start-state distribution, DERIVATION.md 7.3
    def _sample_start(self):
        s = self.start_scale
        r = self.np_random
        x = np.zeros(12)
        x[0:2] = r.uniform(-2.0, 2.0, 2) * s
        x[2] = r.uniform(-1.0, 1.0) * s
        x[3:6] = r.uniform(-0.5, 0.5, 3) * s
        x[6:8] = r.uniform(-0.3, 0.3, 2) * s
        x[8] = r.uniform(-0.5, 0.5) * s
        x[9:12] = r.uniform(-1.0, 1.0, 3) * s
        return x

    def _sample_params(self):
        p = self.p_nominal.copy()
        if self.randomize:
            for name, (lo, hi) in self.ranges.items():
                p[self.pnames.index(name)] = self.np_random.uniform(lo, hi)
        return p

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.p = (self.sys.pvec(options["params"]) if "params" in options
                  else self._sample_params())
        self.x = (np.asarray(options["x0"], float) if "x0" in options
                  else self._sample_start())
        self.x_ref = np.asarray(options.get("x_ref", self.x_hover), float)
        self.k = 0
        return observe(self.x, self.x_ref), {"params": dict(zip(self.pnames, self.p))}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        u = to_rotor(a, self.u_hover, self.u_max)
        self.x = np.asarray(self._step_fn(self.x, u, self.p)).reshape(-1)
        self.k += 1

        cost, parts = stage_cost(self.x, self.x_ref, a, self.tilt_limit)
        reward = float(np.exp(-cost))
        terminated = bool(crashed(self.x, self.x_ref))
        truncated = self.k >= self.max_steps
        info = {"cost": cost, **parts}
        return observe(self.x, self.x_ref), reward, terminated, truncated, info


# ------------------------------------------------------------ reward pieces

W = dict(pos=1.0, vel=0.1, tilt=0.5, yaw=0.1, rate=0.01, act=0.05, limit=20.0)


def stage_cost(x, x_ref, a, tilt_limit):
    """The cost c >= 0 whose exponential is the reward. DERIVATION.md 7.4."""
    e = np.asarray(x, float) - np.asarray(x_ref, float)
    e[ANGLES] = wrap(e[ANGLES])
    phi, theta = x[6], x[7]
    over = (max(0.0, abs(phi) - tilt_limit) ** 2
            + max(0.0, abs(theta) - tilt_limit) ** 2)
    parts = {
        "c_pos": W["pos"] * float(np.linalg.norm(e[0:3])),
        "c_vel": W["vel"] * float(np.linalg.norm(e[3:6])),
        "c_tilt": W["tilt"] * float(phi**2 + theta**2),
        "c_yaw": W["yaw"] * float(e[8] ** 2),
        "c_rate": W["rate"] * float(np.dot(e[9:12], e[9:12])),
        "c_act": W["act"] * float(np.dot(a, a)),
        "c_limit": W["limit"] * float(over),
    }
    return sum(parts.values()), parts


TILT_CRASH = np.radians(60.0)


def crashed(x, x_ref):
    """End the episode well before the Euler singularity at 90 degrees pitch."""
    return (abs(x[6]) > TILT_CRASH or abs(x[7]) > TILT_CRASH
            or np.linalg.norm(x[0:3] - x_ref[0:3]) > 5.0
            or np.linalg.norm(x[9:12]) > 30.0
            or not np.all(np.isfinite(x)))


def _rk4(sys, dt, substeps):
    """One control period of RK4 as a single compiled CasADi call.

    Built from the System's own symbolic ``f``, with the disturbance held at
    zero. One call per env step instead of 4 x substeps Python-level calls.
    """
    x = ca.SX.sym("x", sys.nx)
    u = ca.SX.sym("u", sys.nu)
    p = ca.SX.sym("p", sys.npar)
    w = ca.DM.zeros(sys.nw)
    f = sys.f_fun
    h = dt / substeps
    xk = x
    for _ in range(substeps):
        k1 = f(xk, u, p, w)
        k2 = f(xk + h / 2 * k1, u, p, w)
        k3 = f(xk + h / 2 * k2, u, p, w)
        k4 = f(xk + h * k3, u, p, w)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("rk4_step", [x, u, p], [xk])
