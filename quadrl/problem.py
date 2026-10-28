"""The control problem, defined once. Every script imports from here.

A benchmark row only means something if the rows differ in one thing, so the
plant, the operating point, the weights, the manoeuvre and the simulation
settings live in this one file rather than being retyped per script.
"""

from __future__ import annotations

import numpy as np

import quadrl  # noqa: F401  (puts plants/ on cf's search path)
import cf

PLANT = "quadrotor3d"

# LQR / MPC weights, by Bryson's rule: 1 / (largest acceptable deviation)^2.
#            x  y  z   vx vy vz   phi       theta     psi       p        q        r
Q = np.array([4, 4, 4, 1, 1, 1, 1/0.3**2, 1/0.3**2, 1/0.5**2, 1/3.**2, 1/3.**2, 1/3.**2])
#   0.06 N of rotor deviation (about 90 % of hover thrust) is "large".
R = np.full(4, 1 / 0.06**2)

LQR_DT = 0.002          # LQR runs at 500 Hz - it is a matrix multiply
MPC_DT, MPC_N = 0.02, 20  # 50 Hz, 0.4 s horizon
SIM = dict(dt=0.002, substeps=4)   # 0.5 ms RK4 inside every 2 ms sim step


def system(**params):
    """The 12-state model, optionally with parameter overrides (e.g. m=0.035)."""
    return cf.load(PLANT, **params)


def hover(sys=None):
    """Hover trim (x0, u0) of the NOMINAL model.

    Every controller here is designed about this point - including the PPO
    policy, whose action is a deviation from nominal hover thrust. None of
    them is told the true mass in a mismatch test.
    """
    sys = sys or system()
    p = sys.p0
    u_h = p["m"] * p["g"] / 4
    x0, u0 = cf.trim(sys, x0=np.zeros(12), u0=[u_h] * 4, fix_x=list(range(12)))
    return np.asarray(x0, float), np.asarray(u0, float)


def lqr(sys=None):
    sys = sys or system()
    x0, u0 = hover(sys)
    c = cf.lqr(sys, Q=Q, R=R, x0=x0, u0=u0, dt=LQR_DT)
    return c


def mpc(sys=None):
    sys = sys or system()
    x0, u0 = hover(sys)
    return cf.controller("mpc", sys, Q=Q, R=R, N=MPC_N, dt=MPC_DT, x0=x0, u0=u0)


def step_reference(dx=2.0, dy=0.0, dz=1.0, yaw_deg=0.0):
    """Target state: hover at an offset, with an optional yaw."""
    ref = np.zeros(12)
    ref[0], ref[1], ref[2], ref[8] = dx, dy, dz, np.radians(yaw_deg)
    return ref


def fly(controller, sys_true=None, x0=None, ref=None, t=6.0, seed=0, label=None):
    """Closed loop on ``sys_true`` (default: the nominal plant)."""
    sys_true = sys_true or system()
    x0 = np.zeros(12) if x0 is None else np.asarray(x0, float)
    ref = step_reference() if ref is None else ref
    res = cf.sim(sys_true, controller, t=t, x0=x0, ref=ref, seed=seed, **SIM)
    if label:
        res.meta["controller"] = label
    return res
