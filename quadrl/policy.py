"""A trained PPO policy as a cf Controller.

Wrapping the network as a ``cf.Controller`` is what makes the comparison
fair: PPO then runs through the same ``cf.sim`` loop, the same integrator,
the same reference and the same ``results_to_table`` as LQR and MPC.

At run time the network is evaluated in NumPy, not PyTorch. It is two hidden
layers - three matrix multiplies and two ``tanh`` - and timing it through
Stable-Baselines3's ``predict`` would mostly measure Python and tensor
conversion overhead. The weights are copied out of the trained model once,
and a test checks the NumPy forward pass matches SB3's to float precision.
"""

from __future__ import annotations

import numpy as np

from cf.core.controller import ControlOutput, Controller

from .env import observe, to_rotor
from .problem import hover


def extract_mlp(model):
    """(weights, biases) of the deterministic policy: obs -> mean action."""
    import torch.nn as nn

    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    acts = [m for m in pol.mlp_extractor.policy_net if not isinstance(m, nn.Linear)]
    if not all(isinstance(a, nn.Tanh) for a in acts):
        raise ValueError("expected tanh activations (policy_kwargs activation_fn=nn.Tanh)")
    Ws = [layer.weight.detach().cpu().numpy().astype(float) for layer in layers]
    bs = [layer.bias.detach().cpu().numpy().astype(float) for layer in layers]
    Ws.append(pol.action_net.weight.detach().cpu().numpy().astype(float))
    bs.append(pol.action_net.bias.detach().cpu().numpy().astype(float))
    return Ws, bs


def forward(Ws, bs, obs):
    """Mean action for one observation. tanh hidden layers, linear output."""
    h = np.asarray(obs, float)
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = np.tanh(W @ h + b)
    return Ws[-1] @ h + bs[-1]


class PPOController(Controller):
    """Run a trained policy in a cf closed loop.

    cfg
    ---
    model : a Stable-Baselines3 PPO model, or a path to its .zip
    dt : control period. Must match training (0.01 s).
    """

    requires = {"full_state"}

    def __init__(self, sys, cfg):
        cfg = dict(cfg)
        cfg.setdefault("dt", 0.01)
        super().__init__(sys, cfg)
        model = cfg["model"]
        if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
            from stable_baselines3 import PPO
            model = PPO.load(str(model), device="cpu")
        self.Ws, self.bs = extract_mlp(model)
        _, u0 = hover()                      # NOMINAL hover, as in training
        self.u_hover = float(u0[0])
        self.u_max = float(sys.ulim[0, 1])

    def reset(self, x0=None):
        pass                                 # feedforward network, no state

    def step(self, t, y, ref):
        obs = observe(y, ref.at(t))
        a = np.clip(forward(self.Ws, self.bs, obs), -1.0, 1.0)
        u = to_rotor(a, self.u_hover, self.u_max)
        return ControlOutput(u, {"a_norm": float(np.linalg.norm(a))})

    def __repr__(self):
        sizes = [W.shape[0] for W in self.Ws[:-1]]
        return f"PPOController(hidden={sizes}, dt={self.dt})"
