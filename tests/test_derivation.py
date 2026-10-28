"""Every equation in DERIVATION.md that has a closed form is checked here.

The derivation is written by hand; these tests are what stop it drifting from
the code. Each test names the section it checks.
"""

import numpy as np
import pytest

import cf
from quadrl.problem import Q, R, hover, system


@pytest.fixture(scope="module")
def plant():
    sys = system()
    x0, u0 = hover(sys)
    return sys, sys.p0, x0, u0


def allocation(p):
    """S2: rotor forces -> [T, tau_x, tau_y, tau_z]."""
    L, c = p["L"], p["c"]
    return np.array([[1, 1, 1, 1],
                     [0, L, 0, -L],
                     [-L, 0, L, 0],
                     [c, -c, c, -c]], dtype=float)


def test_s2_allocation_is_invertible_with_the_stated_determinant(plant):
    _, p, _, _ = plant
    M = allocation(p)
    # det M = -8 c L^2 (S2.3)
    assert np.isclose(np.linalg.det(M), -8 * p["c"] * p["L"] ** 2, rtol=1e-12)
    # the inverse quoted in S2.3
    L, c = p["L"], p["c"]
    Minv = 0.25 * np.array([[1, 0, -2 / L, 1 / c],
                            [1, 2 / L, 0, -1 / c],
                            [1, 0, 2 / L, 1 / c],
                            [1, -2 / L, 0, -1 / c]])
    assert np.allclose(M @ Minv, np.eye(4), atol=1e-12)


def test_s2_allocation_matches_the_model(plant):
    """Apply each rotor alone at hover and read the wrench off f(x, u)."""
    sys, p, x0, u0 = plant
    M = allocation(p)
    for i in range(4):
        du = np.zeros(4)
        du[i] = 1e-3
        dxdot = sys.eval_f(x0, u0 + du) - sys.eval_f(x0, u0)
        wrench = np.array([dxdot[5] * p["m"], dxdot[9] * p["Ixx"],
                           dxdot[10] * p["Iyy"], dxdot[11] * p["Izz"]])
        assert np.allclose(wrench, M[:, i] * 1e-3, atol=1e-12)


def test_s4_hover_trim(plant):
    sys, p, x0, u0 = plant
    assert np.allclose(u0, p["m"] * p["g"] / 4, rtol=1e-12)
    assert np.allclose(x0, 0)
    assert np.abs(sys.eval_f(x0, u0)).max() < 1e-12


def test_s4_hover_is_a_family_in_yaw(plant):
    """Any yaw is a trim point: psi does not appear in the hover balance."""
    sys, p, _, u0 = plant
    for psi in (0.3, -2.0, 3.0):
        x = np.zeros(12)
        x[8] = psi
        assert np.abs(sys.eval_f(x, u0)).max() < 1e-12


def analytic_AB(p):
    """S5: the hand-derived linearization at hover."""
    g, m = p["g"], p["m"]
    A = np.zeros((12, 12))
    A[0, 3] = A[1, 4] = A[2, 5] = 1        # position <- velocity
    A[3, 7] = g                             # vx_dot = +g theta
    A[4, 6] = -g                            # vy_dot = -g phi
    A[6, 9] = A[7, 10] = A[8, 11] = 1       # Euler rates = body rates at hover
    B = np.zeros((12, 4))
    B[5, :] = 1 / m
    B[9:12, :] = np.diag([1 / p["Ixx"], 1 / p["Iyy"], 1 / p["Izz"]]) @ allocation(p)[1:]
    return A, B


def test_s5_linearization_matches_casadi_exactly(plant):
    sys, p, x0, u0 = plant
    A, B = cf.jacobians(sys, x0, u0)[:2]
    Aa, Ba = analytic_AB(p)
    assert np.allclose(A, Aa, atol=1e-10, rtol=0)
    assert np.allclose(B, Ba, rtol=1e-12, atol=1e-12)


def test_s5_linearization_matches_finite_differences(plant):
    """Independent of CasADi: central differences on f."""
    sys, p, x0, u0 = plant
    Aa, Ba = analytic_AB(p)
    h = 1e-6
    for j in range(12):
        e = np.zeros(12)
        e[j] = h
        col = (sys.eval_f(x0 + e, u0) - sys.eval_f(x0 - e, u0)) / (2 * h)
        assert np.allclose(col, Aa[:, j], atol=1e-6)
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        col = (sys.eval_f(x0, u0 + e) - sys.eval_f(x0, u0 - e)) / (2 * h)
        assert np.allclose(col, Ba[:, j], rtol=1e-6, atol=1e-3)


def test_s5_four_decoupled_chains(plant):
    """x-theta-q, y-phi-p, z, psi-r: no state couples two chains."""
    _, p, _, _ = plant
    A, B = analytic_AB(p)
    chains = [[0, 3, 7, 10], [1, 4, 6, 9], [2, 5], [8, 11]]
    label = np.empty(12, int)
    for k, idx in enumerate(chains):
        label[idx] = k
    rows, cols = np.nonzero(A)
    assert np.all(label[rows] == label[cols])


def test_s5_open_loop_poles_all_at_zero(plant):
    _, p, _, _ = plant
    A, _ = analytic_AB(p)
    assert np.allclose(np.linalg.eigvals(A), 0)
    # nilpotent of index 4 (the longest chain is a quadruple integrator)
    assert np.allclose(np.linalg.matrix_power(A, 4), 0)
    assert not np.allclose(np.linalg.matrix_power(A, 3), 0)


def test_s5_controllable_and_pose_observable(plant):
    _, p, _, _ = plant
    A, B = analytic_AB(p)
    ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(12)])
    assert np.linalg.matrix_rank(ctrb) == 12
    C = np.zeros((6, 12))
    C[[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8]] = 1      # pose only
    obsv = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(12)])
    assert np.linalg.matrix_rank(obsv) == 12


def test_s3_rotation_matrix_is_a_rotation_and_matches_the_model():
    mod = cf.plant_module("quadrotor3d")
    rng = np.random.default_rng(0)
    sys = system()
    _, u0 = hover(sys)
    for _ in range(20):
        phi, theta, psi = rng.uniform(-1, 1, 3)
        Rm = mod.rotation(phi, theta, psi)
        assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(Rm), 1.0)
        # S3: translational acceleration = (T/m) R e3 - g e3
        x = np.zeros(12)
        x[6:9] = phi, theta, psi
        a = sys.eval_f(x, u0)[3:6]
        T = 4 * u0[0]
        expect = T / sys.p0["m"] * Rm[:, 2] - np.array([0, 0, sys.p0["g"]])
        assert np.allclose(a, expect, atol=1e-12)


def test_s3_euler_kinematics_match_W_inverse():
    """S3.4: eta_dot = W(eta)^-1 omega, with W as written in the derivation."""
    sys = system()
    _, u0 = hover(sys)
    rng = np.random.default_rng(1)
    for _ in range(20):
        phi, theta, psi = rng.uniform(-1, 1, 3)
        om = rng.uniform(-2, 2, 3)
        W = np.array([[1, 0, -np.sin(theta)],
                      [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
                      [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]])
        x = np.zeros(12)
        x[6:9] = phi, theta, psi
        x[9:12] = om
        eta_dot = sys.eval_f(x, u0)[6:9]
        assert np.allclose(W @ eta_dot, om, atol=1e-12)
        assert np.isclose(np.linalg.det(W), np.cos(theta))     # singular at 90 deg


def test_s6_lqr_closed_loop_is_stable_and_matches_scipy(plant):
    from scipy.linalg import solve_continuous_are

    sys, p, x0, u0 = plant
    A, B = analytic_AB(p)
    P = solve_continuous_are(A, B, np.diag(Q), np.diag(R))
    K = np.linalg.solve(np.diag(R), B.T @ P)
    ctrl = cf.lqr(sys, Q=Q, R=R, x0=x0, u0=u0)
    assert np.allclose(ctrl.K, K, rtol=1e-8, atol=1e-10)
    assert np.all(np.linalg.eigvals(A - B @ K).real < 0)


def test_s6_2_zoh_discretization_follows_the_integrator_chains(plant):
    """S6.2: on a chain of integrators the ZOH entries are dt^k / k! terms."""
    from quadrl.problem import mpc

    sys, p, _, _ = plant
    m = mpc(sys)
    dt, g = m.dt, p["g"]
    kq = p["L"] / p["Iyy"]                       # q_dot per newton on rotor 3
    assert np.isclose(m.Ad[0, 7], g * dt**2 / 2)          # x  <- theta
    assert np.isclose(m.Ad[0, 10], g * dt**3 / 6)         # x  <- q
    assert np.isclose(m.Bd[10, 2], kq * dt)               # q  <- f3
    assert np.isclose(m.Bd[7, 2], kq * dt**2 / 2)         # theta <- f3
    assert np.isclose(m.Bd[3, 2], g * kq * dt**3 / 6)     # vx <- f3
    assert np.isclose(m.Bd[0, 2], g * kq * dt**4 / 24)    # x  <- f3
    assert np.isclose(m.Bd[2, 0], dt**2 / (2 * p["m"]))   # z  <- f1


def test_s6_3_terminal_weight_is_the_discrete_riccati_solution(plant):
    from scipy.linalg import solve_discrete_are

    from quadrl.problem import mpc

    sys, *_ = plant
    m = mpc(sys)
    P = solve_discrete_are(m.Ad, m.Bd, np.diag(Q), np.diag(R))
    assert np.allclose(m.Qf, P, rtol=1e-8)
