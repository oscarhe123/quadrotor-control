"""Compare LQR, MPC and PPO on the same simulator.  python scripts/evaluate.py

Writes figs/*.png, figs/three_way.gif and results/metrics.json. Protocol:
DERIVATION.md section 9. Every number in the README comes from this script.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import cf  # noqa: E402
from quadrl import ROOT  # noqa: E402
from quadrl.env import QuadrotorEnv  # noqa: E402
from quadrl.policy import PPOController  # noqa: E402
from quadrl.problem import fly, lqr, mpc, step_reference, system  # noqa: E402

# Reference categorical palette, slots 1-3 in fixed order: a controller is the
# same colour in every figure and in the GIF.
COLORS = {"LQR": "#2a78d6", "MPC": "#eb6834", "PPO": "#1baf7a"}
INK, MUTED, GRID = "#2b2b29", "#8a8984", "#e4e3de"
HEAVY = {"m": 0.035}                      # +30 % mass, told to no controller
T_END = 6.0


def controllers(sys_nominal, model_path):
    return {"LQR": lqr(sys_nominal), "MPC": mpc(sys_nominal),
            "PPO": PPOController(sys_nominal, {"model": model_path})}


def metrics(res, ref):
    e = res.x[:, :3] - ref[:3]
    dist = np.linalg.norm(e, axis=1)
    tilt = np.degrees(np.abs(res.x[:, 6:8]).max())
    calls = res.info["solve_ms"][res.info["solve_ms"] > 0]
    return {
        "pos_rmse_m": float(np.sqrt(np.mean(dist ** 2))),
        "final_err_m": float(dist[-1]),
        "final_z_err_m": float(e[-1, 2]),
        "peak_tilt_deg": float(tilt),
        "tilt_violation_steps": int(res.state_violations()),
        "tilt_excursion_deg": float(np.degrees(res.state_excursion())),
        "settle_5cm_s": _settle(res.t, dist, 0.05),
        "call_ms_median": float(np.median(calls)),
        "call_ms_p99": float(np.percentile(calls, 99)),
        "crashed": bool(np.degrees(np.abs(res.x[:, 6:8]).max()) > 60
                        or not np.all(np.isfinite(res.x))),
    }


def _settle(t, dist, tol):
    outside = np.nonzero(dist > tol)[0]
    if len(outside) == 0:
        return 0.0
    k = outside[-1] + 1
    return float(t[k]) if k < len(t) else float("nan")


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def label_end(ax, t, y, text, color, dy=0.0):
    """Direct label at the right end of a line, in text ink with a colour key."""
    ax.annotate(text, (t[-1], y[-1] + dy), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.plot([t[-1]], [y[-1]], "o", ms=4, color=color, clip_on=False)


def fig_step(runs, tilt_lim, out):
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.4), sharex=True,
                             height_ratios=[1.6, 1, 1])
    ax = axes[0]
    ax.axhspan(tilt_lim, 60, color="#f1efe8", lw=0)
    ax.axhspan(-60, -tilt_lim, color="#f1efe8", lw=0)
    for s in (tilt_lim, -tilt_lim):
        ax.axhline(s, color=INK, ls=(0, (4, 3)), lw=1)
    ax.text(5.95, tilt_lim + 1.2, f"tilt limit ±{tilt_lim:.0f}°", ha="right",
            fontsize=8.5, color=INK)
    for name, r in runs.items():
        ax.plot(r.t, np.degrees(r.x[:, 7]), color=COLORS[name], lw=2, label=name)
    ax.set_ylim(-25, 40)
    ax.set_ylabel("pitch θ [deg]", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    for ax_, idx, target, lab in ((axes[1], 0, 2.0, "x [m]"), (axes[2], 2, 1.0, "z [m]")):
        ax_.axhline(target, color=MUTED, lw=1, ls=":")
        for name, r in runs.items():
            ax_.plot(r.t, r.x[:, idx], color=COLORS[name], lw=2)
        ax_.set_ylabel(lab, color=INK)
    axes[2].set_xlabel("time [s]", color=INK)
    for a in axes:
        style(a)
    fig.suptitle("2 m sideways + 1 m up, nominal vehicle", fontsize=11, color=INK,
                 x=0.07, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_mismatch(runs, out):
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.text(0.05, 1.004, "target", fontsize=8.5, color=MUTED, va="bottom")
    finals = {name: r.x[-1, 2] for name, r in runs.items()}
    for name, r in runs.items():
        ax.plot(r.t, r.x[:, 2], color=COLORS[name], lw=2)
    # End labels, nudged apart so two finals 2 cm apart do not overprint.
    order = sorted(finals, key=finals.get)
    placed = []
    for name in order:
        y = finals[name]
        if placed and y - placed[-1] < 0.035:
            y = placed[-1] + 0.035
        placed.append(y)
        r = runs[name]
        ax.plot([r.t[-1]], [finals[name]], "o", ms=4, color=COLORS[name])
        ax.annotate(f"{name}  {1000*(finals[name]-1):+.0f} mm", (r.t[-1], y),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    lo = min(finals.values())
    ax.set_xlim(0, T_END + 1.3)
    ax.set_ylim(lo - 0.12, 1.08)
    ax.set_xlabel("time [s]", color=INK)
    ax.set_ylabel("z [m]", color=INK)
    ax.set_title("Same step, vehicle 30 % heavier than every controller assumes",
                 fontsize=11, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_learning(csv_path, out):
    import csv

    steps, mean = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("eval/mean_reward"):
                steps.append(float(row["time/total_timesteps"]))
                mean.append(float(row["eval/mean_reward"]))
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.axhline(500, color=MUTED, lw=1, ls=":")
    ax.text(steps[0] / 1e6, 505, "maximum possible return (500 steps × reward 1)",
            fontsize=8.5, color=MUTED, va="bottom")
    ax.plot(np.array(steps) / 1e6, mean, color=COLORS["PPO"], lw=2)
    ax.set_xlabel("environment steps [millions]", color=INK)
    ax.set_ylabel("eval return", color=INK)
    ax.set_ylim(0, 540)
    ax.set_title("PPO learning curve (20 deterministic episodes per point)",
                 fontsize=11, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def random_starts(ctrls, n):
    """Scenario C: starts drawn from the training distribution, nominal mass."""
    env = QuadrotorEnv(randomize=False)
    out = {k: {"success": 0, "crash": 0, "final_err": []} for k in ctrls}
    ref = np.zeros(12)
    for i in range(n):
        env.reset(seed=50_000 + i)
        x0 = env.x.copy()
        for name, c in ctrls.items():
            r = fly(c, x0=x0, ref=ref, t=T_END)
            m = metrics(r, ref)
            ok = (not m["crashed"]) and m["final_err_m"] < 0.05
            out[name]["success"] += int(ok)
            out[name]["crash"] += int(m["crashed"])
            out[name]["final_err"].append(m["final_err_m"])
    return {k: {"success_rate": v["success"] / n, "crashes": v["crash"],
                "median_final_err_m": float(np.median(v["final_err"]))}
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "best" / "best_model.zip"))
    ap.add_argument("--starts", type=int, default=50)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures and GIF; keep results/metrics.json")
    args = ap.parse_args()

    figs, results = ROOT / "figs", ROOT / "results"
    figs.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    nominal = system()
    tilt_lim = float(np.degrees(nominal.xlim[7, 1]))
    ref = step_reference()
    ctrls = controllers(nominal, args.model)

    report = {"model": Path(args.model).name}
    # A: nominal step
    runs_a = {k: fly(c, t=T_END, label=k) for k, c in ctrls.items()}
    report["A_nominal_step"] = {k: metrics(r, ref) for k, r in runs_a.items()}
    # B: heavy vehicle
    heavy = system(**HEAVY)
    runs_b = {k: fly(c, sys_true=heavy, t=T_END, label=k) for k, c in ctrls.items()}
    report["B_heavy_step"] = {k: metrics(r, ref) for k, r in runs_b.items()}
    if not args.figures_only:
        # C: random starts
        report["C_random_starts"] = random_starts(ctrls, args.starts)
        report["C_random_starts"]["n"] = args.starts
        print(cf.results_to_table(list(runs_a.values())))
        (results / "metrics.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))

    fig_step(runs_a, tilt_lim, figs / "step_nominal.png")
    fig_mismatch(runs_b, figs / "step_heavy_z.png")
    fig_learning(ROOT / "models" / "progress.csv", figs / "learning_curve.png")
    if not args.no_gif:
        draw = cf.plant_module("quadrotor3d").draw
        def pitch(x, u):
            th = np.degrees(x[7])
            flag = "  over limit" if abs(th) > tilt_lim + 0.05 else ""
            return f"pitch {th:+5.1f}°{flag}"

        cf.animate(list(runs_a.values()), draw, projection="3d", trail=(0, 1, 2),
                   fps=25, colors=[COLORS[k] for k in runs_a], panels=True,
                   readout=pitch, figsize=(11.5, 4.4), elev=14, azim=-80,
                   save=figs / "three_way.gif",
                   title=f"2 m sideways + 1 m up, tilt limit ±{tilt_lim:.0f}°")


if __name__ == "__main__":
    main()
