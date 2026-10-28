"""6-DOF quadrotor: twelve states, four rotors.

    x, y, z          position, world frame, z up        [m]
    vx, vy, vz       velocity, world frame              [m/s]
    phi, theta, psi  roll, pitch, yaw (ZYX Euler)       [rad]
    p, q, r          body angular rates                 [rad/s]

Four rotor forces in, six degrees of freedom out - which is the whole point
of the vehicle. The rotors make only this:

    T  = f1 + f2 + f3 + f4            thrust, along BODY z and nowhere else
    tx = L (f2 - f4)                  roll
    ty = L (f3 - f1)                  pitch
    tz = c (f1 - f2 + f3 - f4)        yaw, from unbalanced rotor DRAG

Four numbers, from four rotors, through a square invertible map. There is no
combination of rotor speeds that pushes sideways: every rotor points along
body z. So the machine is **underactuated by two** - it buys lateral motion
by tilting, which couples position to attitude and is exactly why real
quadrotors nest a position loop outside an attitude loop.

Yaw is the weak channel. It comes from drag rather than thrust, and
c / L = 0.54, so a given rotor imbalance produces about half the yaw torque
it does roll torque - against an inertia (Izz) nearly twice as large.

ATTITUDE REPRESENTATION. ZYX Euler angles, not quaternions. The kinematic
block below contains tan(theta) and 1/cos(theta), so it is singular at
theta = +-90 degrees: gimbal lock. That is not a defect of this file but of
any three-number parameterisation of 3D rotation - SO(3) admits no global
non-singular chart (CLAUDE.md S8). The trade accepted here: Euler angles are
three ordinary states with `rad` units, so cf's existing wrap handles them
and nothing in the library needs to change. Quaternions would need a norm
constraint the integrator does not enforce and a notion of error that is not
subtraction. Valid for hover, waypoints and moderate manoeuvres; invalid for
flips. If you need flips, that is the moment to do SO(3) properly.
"""

from pathlib import Path

import casadi as ca

from cf.loader import read_params
from cf.model import ode

DATA = Path(__file__).with_name("params.yaml")


def build(full_state: bool = True, **overrides):
    """Build the System. Keyword args override any parameter in params.yaml.

    ``full_state=True`` (default) measures all twelve states, which is what a
    machine in a motion-capture volume effectively has and what state
    feedback needs. ``full_state=False`` measures pose only - position and
    the three angles - which is the honest sensor set for a drone with an
    IMU and a position fix, and forces the loop through an estimator.
    """
    data = read_params(DATA)
    params = dict(data["params"])

    unknown = set(overrides) - set(params)
    if unknown:
        raise KeyError(f"quadrotor3d has no parameter(s) {sorted(unknown)}; "
                       f"known: {sorted(params)}")
    params.update({k: float(v) for k, v in overrides.items()})

    def dynamics(x, u, p, w):
        # ---- allocation: four rotors to one force and three torques
        thrust = u.f1 + u.f2 + u.f3 + u.f4
        tau_x = p.L * (u.f2 - u.f4)
        tau_y = p.L * (u.f3 - u.f1)
        tau_z = p.c * (u.f1 - u.f2 + u.f3 - u.f4)

        cphi, sphi = ca.cos(x.phi), ca.sin(x.phi)
        cth, sth = ca.cos(x.theta), ca.sin(x.theta)
        cpsi, spsi = ca.cos(x.psi), ca.sin(x.psi)

        # Third column of the body-to-world rotation: where thrust points.
        bz = (cpsi * sth * cphi + spsi * sphi,
              spsi * sth * cphi - cpsi * sphi,
              cth * cphi)

        return [
            x.vx, x.vy, x.vz,
            thrust / p.m * bz[0] + w.wx,
            thrust / p.m * bz[1] + w.wy,
            thrust / p.m * bz[2] - p.g + w.wz,
            # Euler kinematics. The tan and 1/cos are the gimbal lock.
            x.p + sphi * sth / cth * x.q + cphi * sth / cth * x.r,
            cphi * x.q - sphi * x.r,
            (sphi * x.q + cphi * x.r) / cth,
            # Euler's equations, with the gyroscopic cross terms.
            (tau_x + (p.Iyy - p.Izz) * x.q * x.r) / p.Ixx,
            (tau_y + (p.Izz - p.Ixx) * x.p * x.r) / p.Iyy,
            (tau_z + (p.Ixx - p.Iyy) * x.p * x.q) / p.Izz,
        ]

    states = ["x", "y", "z", "vx", "vy", "vz",
              "phi", "theta", "psi", "p", "q", "r"]
    units = ["m"] * 3 + ["m/s"] * 3 + ["rad"] * 3 + ["rad/s"] * 3
    pose = ["x", "y", "z", "phi", "theta", "psi"]

    return ode(
        "quadrotor3d",
        states=states,
        inputs=["f1", "f2", "f3", "f4"],
        params=params,
        dynamics=dynamics,
        # Wind as a world-frame acceleration: a disturbance you can forecast,
        # not zero-mean noise. Same distinction the HVAC plant draws.
        disturbances=["wx", "wy", "wz"],
        outputs=None if full_state else pose,
        bounds=data.get("bounds"),
        units={"x": units, "u": ["N"] * 4,
               "y": units if full_state else ["m"] * 3 + ["rad"] * 3},
        dt_nominal=data.get("dt_nominal"),
        uncertainty=data.get("uncertainty"),
    )


# ---------------------------------------------------------------- geometry
# What a quadrotor looks like is plant data, exactly like its mass. cf owns
# the animation loop (cf.animate); this owns the shape, so adding another
# plant needs no edit to the library.

import numpy as np                                               # noqa: E402

_P = read_params(DATA)["params"]
ARM = float(_P["L"])
HOVER = float(_P["m"]) * float(_P["g"]) / 4
SCALE = 4.0                          # airframe drawn larger than life
ARROW = 0.7 * SCALE * ARM / HOVER    # a hover arrow is 0.7 of a drawn arm

# Plus configuration, body frame: front, left, back, right.
_ROTORS = np.array([[ARM, 0, 0], [0, ARM, 0], [-ARM, 0, 0], [0, -ARM, 0]])


def rotation(phi, theta, psi):
    """Body-to-world rotation, ZYX Euler. The same convention as `dynamics`."""
    cx, sx = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(psi), np.sin(psi)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,     cy * sx,                cy * cx],
    ])


def draw(ax, x, u, color="#1f77b4", scale=SCALE):
    """Draw the quadrotor at state ``x`` with rotor forces ``u``.

    The thrust arrows are the point: they make the control visible, and you
    can watch the machine tilt *before* it translates - the one thing about
    an underactuated quadrotor worth seeing.

    ``scale`` draws the airframe larger than life. A Crazyflie is 84 mm
    across and a useful flight is metres, so at true scale the vehicle is a
    dot. Exaggerating it is normal in robotics figures; only the drawing is
    scaled, never the dynamics.
    """
    pos = np.asarray(x[:3], dtype=float)
    R = rotation(float(x[6]), float(x[7]), float(x[8]))
    hub = (R @ (_ROTORS * scale).T).T + pos       # rotor hubs in world frame
    body_z = R[:, 2]                              # thrust direction

    for a, b in ((0, 2), (1, 3)):                 # the two arms
        ax.plot(*[[hub[a][k], hub[b][k]] for k in range(3)],
                lw=2.5, color=color, solid_capstyle="round", zorder=3)
    ax.plot(*[hub[:, k] for k in range(3)], "o", ms=5, color=color, zorder=4)

    for h, f in zip(hub, np.atleast_1d(u)):
        tip = h + body_z * ARROW * float(f) * (scale / SCALE)
        ax.plot(*[[h[k], tip[k]] for k in range(3)],
                lw=1.4, color="#5f5e5a", alpha=0.9, zorder=2)   # neutral: not a series

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel("z [m]")
