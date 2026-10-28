"""Quadrotor control: LQR, MPC and a PPO policy on one 6-DOF model.

Importing this package puts ``plants/`` on cf's plant search path, so
``cf.load("quadrotor3d")`` finds the model here rather than inside cf. That is
the point of the layout: the plant is data that lives with the project, and cf
needed no edit to fly it.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANTS = ROOT / "plants"

_existing = os.environ.get("CF_PLANTS", "")
if str(PLANTS) not in _existing.split(os.pathsep):
    os.environ["CF_PLANTS"] = os.pathsep.join(p for p in (str(PLANTS), _existing) if p)
