"""The environment and the deployed policy agree with the model and each other."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from cf.sim.integrate import Integrator
from quadrl.env import OBS_SCALE, QuadrotorEnv, observe, stage_cost, to_rotor


@pytest.fixture(scope="module")
def env():
    return QuadrotorEnv()


def test_gymnasium_api(env):
    check_env(QuadrotorEnv(), skip_render_check=True)


def test_env_integrates_the_same_model_cf_simulates(env):
    """One control period vs cf's RK4 at 50x finer steps."""
    env.reset(seed=3)
    x0, p = env.x.copy(), env.p.copy()
    a = np.array([0.3, -0.2, 0.1, 0.05])
    env.step(a)
    ref = Integrator(env.sys, "rk4", substeps=200).step(
        x0, to_rotor(a, env.u_hover, env.u_max), p, np.zeros(env.sys.nw), env.dt)
    assert np.abs(env.x - ref).max() < 1e-8


def test_zero_action_holds_nominal_hover(env):
    env.reset(options={"x0": np.zeros(12), "params": {}})
    for _ in range(200):
        _, r, term, _, _ = env.step(np.zeros(4))
    assert not term
    assert np.abs(env.x).max() < 1e-12
    assert r == pytest.approx(1.0)


def test_action_map_endpoints(env):
    uh, um = env.u_hover, env.u_max
    assert np.allclose(to_rotor(np.zeros(4), uh, um), uh)
    assert np.allclose(to_rotor(-np.ones(4), uh, um), 0.0)
    assert np.allclose(to_rotor(np.ones(4), uh, um), min(2 * uh, um))
    assert np.allclose(to_rotor(5 * np.ones(4), uh, um), min(2 * uh, um))   # clipped


def test_observation_is_scaled_wrapped_error():
    x = np.zeros(12)
    ref = np.zeros(12)
    x[8], ref[8] = np.pi - 0.1, -np.pi + 0.1          # 0.2 rad apart across the seam
    assert observe(x, ref)[8] == pytest.approx(-0.2 / OBS_SCALE[8])
    x[0] = 1.0
    assert observe(x, ref)[0] == pytest.approx(1.0 / OBS_SCALE[0])


def test_reward_is_in_unit_interval_so_crashing_never_pays(env):
    """S7.4: r = exp(-c) with c >= 0, so every surviving step earns > 0."""
    env.reset(seed=5)
    for _ in range(50):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        assert 0.0 < r <= 1.0
        assert info["cost"] >= 0.0
        if term or trunc:
            break


def test_tilt_limit_is_penalised_only_beyond_the_limit(env):
    lim = env.tilt_limit
    x = np.zeros(12)
    x[7] = lim * 0.999
    inside = stage_cost(x, np.zeros(12), np.zeros(4), lim)[1]["c_limit"]
    x[7] = lim + 0.1
    outside = stage_cost(x, np.zeros(12), np.zeros(4), lim)[1]["c_limit"]
    assert inside == 0.0 and outside > 0.0


def test_episode_ends_before_the_euler_singularity(env):
    x = np.zeros(12)
    x[7] = np.radians(55)
    x[10] = 20.0                       # pitching fast
    env.reset(options={"x0": x})
    for _ in range(100):
        *_, term, _, _ = env.step(np.zeros(4))
        if term:
            break
    assert term
    assert abs(env.x[7]) < np.radians(90)


def test_domain_randomization_stays_inside_the_declared_ranges(env):
    seen = []
    for s in range(50):
        _, info = env.reset(seed=s)
        seen.append(info["params"]["m"])
        for k, (lo, hi) in env.ranges.items():
            assert lo <= info["params"][k] <= hi
    assert np.ptp(seen) > 0.005                     # actually varies


def test_numpy_forward_pass_matches_stable_baselines(env):
    import torch.nn as nn
    from stable_baselines3 import PPO

    from quadrl.policy import extract_mlp, forward

    model = PPO("MlpPolicy", QuadrotorEnv(), seed=0, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                                   activation_fn=nn.Tanh))
    Ws, bs = extract_mlp(model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=12).astype(np.float32)
        a_sb3, _ = model.predict(obs, deterministic=True)
        a_np = np.clip(forward(Ws, bs, obs), -1, 1)
        assert np.allclose(a_np, a_sb3, atol=1e-5)
