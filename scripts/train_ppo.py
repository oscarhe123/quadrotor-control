"""Train the PPO policy.  python scripts/train_ppo.py [--steps 3000000]

Writes models/ppo_quadrotor.zip (final), models/best/best_model.zip (best
deterministic evaluation) and models/progress.csv (the learning curve).
Hyperparameters and the reasons for them: DERIVATION.md section 8.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn as nn  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from quadrl import ROOT  # noqa: E402
from quadrl.env import QuadrotorEnv  # noqa: E402

HYPER = dict(
    n_steps=1024,          # per env -> 8192 transitions per update
    batch_size=256,
    n_epochs=10,
    gamma=0.995,           # 1/(1-gamma) = 200 steps = 2 s look-ahead at 100 Hz
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                       activation_fn=nn.Tanh, log_std_init=-0.5),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    venv = make_vec_env(QuadrotorEnv, n_envs=args.envs, seed=args.seed)
    eval_env = make_vec_env(QuadrotorEnv, n_envs=1, seed=10_000 + args.seed)

    model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0, **HYPER)
    model.set_logger(configure(str(out), ["csv"]))
    cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=50_000 // args.envs,
                      best_model_save_path=str(out / "best"), deterministic=True,
                      verbose=1)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(out / "ppo_quadrotor")
    print(f"trained {args.steps:,} steps in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
