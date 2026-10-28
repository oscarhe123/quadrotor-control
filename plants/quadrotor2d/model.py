"""Planar quadrotor — the smallest plant that is still a drone.

Two rotors in a vertical plane. Six states, two inputs:

    x, z        position                      [m]
    theta       pitch, from level             [rad]
    vx, vz      velocity                      [m/s]
    omega       pitch rate                    [rad/s]

    x'  = vx                      vx' = -(T1 + T2)/m * sin(theta) + wx
    z'  = vz                      vz' =  (T1 + T2)/m * cos(theta) - g + wz
    th' = omega                   om' =  (T2 - T1) * L / I

Six lines, and they contain the whole problem: **you cannot translate without
tilting first.** Thrust points along the body z axis, so moving sideways means
pitching, which means the position loop can only act through the attitude
loop. That is what underactuated means here, and it is why every real
quadrotor is a cascade.

Why planar rather than 6-DOF. The full machine needs two things cf does not
have: SO(3) attitude, which CLAUDE.md S8 defers until there is evidence to
pick the abstraction, and control allocation from four rotors to thrust plus
three torques, which `allocate/` was always going to provide and never did.
The planar model keeps a single angle - so the existing `units: rad` wrap
handles it - and a 2x2 rotor map that inverts by hand. Nothing essential is
lost: it is still nonlinear, still underactuated, and it still tumbles if you
get the thrust bound wrong.

The numbers are a Crazyflie 2.x; see params.yaml for the source of each.
"""

from pathlib import Path

import casadi as ca

from cf.loader import read_params
from cf.model import ode

DATA = Path(__file__).with_name("params.yaml")


def build(full_state: bool = True, **overrides):
    """Build the System. Keyword args override any parameter in params.yaml.

    ``full_state=True`` (default) measures everything, which is what a drone
    flying in a motion-capture volume effectively has and what a state
    feedback or predictive controller needs. ``full_state=False`` measures
    pose only - ``x, z, theta`` - which is the honest sensor set for a
    machine carrying an IMU and a position fix but no velocity sensor, and
    forces the loop to close through an estimator.
    """
    data = read_params(DATA)
    params = dict(data["params"])

    unknown = set(overrides) - set(params)
    if unknown:
        raise KeyError(f"quadrotor2d has no parameter(s) {sorted(unknown)}; "
                       f"known: {sorted(params)}")
    params.update({k: float(v) for k, v in overrides.items()})

    def dynamics(x, u, p, w):
        thrust = u.T1 + u.T2                      # along the body z axis
        torque = (u.T2 - u.T1) * p.L              # differential thrust
        return [
            x.vx,
            x.vz,
            x.omega,
            -thrust / p.m * ca.sin(x.theta) + w.wx,
            thrust / p.m * ca.cos(x.theta) - p.g + w.wz,
            torque / p.I,
        ]

    states = ["x", "z", "theta", "vx", "vz", "omega"]
    units = ["m", "m", "rad", "m/s", "m/s", "rad/s"]

    return ode(
        "quadrotor2d",
        states=states,
        inputs=["T1", "T2"],
        params=params,
        dynamics=dynamics,
        # Wind, as an acceleration. A disturbance you can measure or forecast,
        # not zero-mean noise - the same distinction the HVAC plant draws.
        disturbances=["wx", "wz"],
        outputs=None if full_state else ["x", "z", "theta"],
        bounds=data.get("bounds"),
        units={"x": units,
               "u": ["N", "N"],
               "y": units if full_state else ["m", "m", "rad"]},
        dt_nominal=data.get("dt_nominal"),
        uncertainty=data.get("uncertainty"),
    )


# ---------------------------------------------------------------- geometry
# The shape of the vehicle is plant data, exactly like its mass. cf owns the
# animation loop (cf.animate); this owns what a quadrotor looks like, so
# adding another 2D plant needs no edit to the library.

_P = read_params(DATA)["params"]
ARM = float(_P["L"])
HOVER = float(_P["m"]) * float(_P["g"]) / 2        # thrust per rotor at hover
ARROW = ARM / HOVER                                # so a hover arrow is 1 arm


def draw(ax, x, u, color="#1f77b4"):
    """Draw the quadrotor at state ``x`` with rotor thrusts ``u``.

    The thrust arrows are the point: they make the control action visible,
    and a reader can see the machine tilt *before* it translates, which is
    the one thing about an underactuated quadrotor worth seeing.
    """
    import numpy as np

    px, pz, th = float(x[0]), float(x[1]), float(x[2])
    c, s = np.cos(th), np.sin(th)

    # Rotors sit at +-ARM along the body x axis.
    rotors = [(px - ARM * c, pz - ARM * s), (px + ARM * c, pz + ARM * s)]
    ax.plot([rotors[0][0], rotors[1][0]], [rotors[0][1], rotors[1][1]],
            lw=3.0, color=color, solid_capstyle="round", zorder=3)
    ax.plot([r[0] for r in rotors], [r[1] for r in rotors], "o",
            ms=7, color=color, zorder=4)

    # Thrust acts along the body z axis, which is (-sin th, cos th).
    for (rx, rz), thrust in zip(rotors, np.atleast_1d(u)):
        length = ARROW * float(thrust)
        if length > 1e-4:
            ax.arrow(rx, rz, -s * length, c * length, width=ARM * 0.05,
                     head_width=ARM * 0.22, head_length=ARM * 0.22,
                     color="#d62728", alpha=0.85, length_includes_head=True,
                     zorder=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
