# Derivation

Everything between the physics and the trained policy, written so each step can
be checked. Wherever a result has a closed form, a test in
`tests/test_derivation.py` or `tests/test_env.py` asserts it against the code,
and the section number is in the test's name — so if this document and the code
ever disagree, `pytest` says so.

| Section | What | Checked by |
|---|---|---|
| 1 | Notation, frames, parameters | — |
| 2 | Rotor model and allocation | `test_s2_*` |
| 3 | Equations of motion | `test_s3_*` |
| 4 | Hover equilibrium | `test_s4_*` |
| 5 | Linearization, controllability, observability | `test_s5_*` |
| 6 | LQR and linear MPC | `test_s6_*` |
| 7 | The reinforcement-learning problem (MDP) | `tests/test_env.py` |
| 8 | PPO | `test_numpy_forward_pass_matches_stable_baselines` |
| 9 | Evaluation protocol | — |
| 10 | What is *not* claimed | — |

---

## 1. Notation, frames, parameters

**Frames.** World frame $\mathcal W$: $z$ up, gravity $-g\,e_3$. Body frame
$\mathcal B$: origin at the centre of mass, $x_b$ towards rotor 1, $y_b$ towards
rotor 2, $z_b$ along the rotor axes.

**State** ($n_x = 12$), in the order the code uses:

$$
x = \big(\underbrace{x,\,y,\,z}_{p\ \in\ \mathcal W},\ \underbrace{v_x,\,v_y,\,v_z}_{v=\dot p\ \in\ \mathcal W},\ \underbrace{\phi,\,\theta,\,\psi}_{\eta:\ \text{ZYX Euler}},\ \underbrace{p,\,q,\,r}_{\omega\ \in\ \mathcal B}\big)
$$

The body rates $p, q, r$ share a letter with position $p$; context always
disambiguates and the code names them `p, q, r` versus `x, y, z`.

**Input** ($n_u = 4$): rotor thrusts $f = (f_1, f_2, f_3, f_4)$ in newtons,
$0 \le f_i \le f_{\max}$.

**Parameters** — `plants/quadrotor3d/params.yaml`, with provenance there:

| Symbol | Value | Meaning |
|---|---|---|
| $m$ | 0.027 kg | mass |
| $L$ | 0.042 m | centre to rotor |
| $I_{xx}, I_{yy}, I_{zz}$ | 1.6572e-5, 1.6656e-5, 2.9262e-5 kg m² | principal inertias (diagonal $I$) |
| $c$ | 0.0228 m | rotor drag torque / thrust |
| $g$ | 9.81 m/s² | gravity |
| $f_{\max}$ | 0.1325 N | per rotor → thrust-to-weight 2.0 |
| tilt limit | ±15° on $\phi, \theta$ | flight envelope (not an actuator limit) |

---

## 2. Rotor model and allocation

### 2.1 Geometry

Plus configuration. Rotor positions in $\mathcal B$:

$$
r_1 = (L,0,0),\quad r_2 = (0,L,0),\quad r_3 = (-L,0,0),\quad r_4 = (0,-L,0).
$$

Each rotor produces a force $F_i = f_i\,e_3$ in $\mathcal B$ and a reaction
(drag) torque $\pm c f_i\,e_3$ opposite its spin. Rotors 1, 3 spin one way and
2, 4 the other, so the reaction torques are $+cf_1, -cf_2, +cf_3, -cf_4$.

**Assumption:** thrust and drag torque are proportional to each other
(both $\propto \Omega_i^2$ for rotor speed $\Omega_i$), and rotor dynamics are
instantaneous — the controller commands force directly. Motor time constants
on a Crazyflie are tens of milliseconds; neglecting them is the main modelling
simplification and is listed in §10.

### 2.2 Wrench

Total thrust is the sum of the forces. The moment of $F_i$ about the centre of
mass is $r_i \times F_i = (r_{i,y} f_i,\ -r_{i,x} f_i,\ 0)$. Summing:

$$
\begin{aligned}
T &= f_1 + f_2 + f_3 + f_4 \\
\tau_x &= L\,(f_2 - f_4) \\
\tau_y &= L\,(f_3 - f_1) \\
\tau_z &= c\,(f_1 - f_2 + f_3 - f_4)
\end{aligned}
\qquad\Longleftrightarrow\qquad
\begin{pmatrix}T\\ \tau\end{pmatrix} = M f,\quad
M = \begin{pmatrix} 1&1&1&1\\ 0&L&0&-L\\ -L&0&L&0\\ c&-c&c&-c\end{pmatrix}.
$$

Sign check for $\tau_y$: rotor 1 sits at $+x_b$; pushing it up tips the nose up,
which is a *negative* rotation about $+y_b$ (right-hand rule: $+y_b$ rotates
$z_b$ towards $x_b$, i.e. nose *down*). Hence $-Lf_1$. ✔

### 2.3 Invertibility

$$
\det M = -8\,c\,L^2 \neq 0,
\qquad
M^{-1} = \tfrac14\begin{pmatrix}
1 & 0 & -2/L & 1/c\\
1 & 2/L & 0 & -1/c\\
1 & 0 & 2/L & 1/c\\
1 & -2/L & 0 & -1/c
\end{pmatrix}.
$$

Four rotors produce exactly four independent quantities, and **no
combination produces a force with a component along $x_b$ or $y_b$**. With six
degrees of freedom and four independent inputs, the vehicle is underactuated
by two (§5.3 shows how position is still reachable — through attitude).

Because $M$ is square and invertible, allocation is a fixed linear map inside
the model; no allocation *optimisation* is needed. That changes when $f$
saturates: $M^{-1}$ of a feasible wrench can be infeasible. Here the
controllers act on $f$ directly and the bounds are on $f$, so saturation is
handled where it arises (clipping for LQR, hard QP bounds for MPC, the action
map for PPO).

*Checked by:* `test_s2_allocation_is_invertible_with_the_stated_determinant`,
`test_s2_allocation_matches_the_model` (perturbs each rotor in the real model
and reads the wrench back).

---

## 3. Equations of motion

### 3.1 Attitude: ZYX Euler angles

Body-to-world rotation, yaw then pitch then roll:

$$
R(\eta) = R_z(\psi)\,R_y(\theta)\,R_x(\phi) =
\begin{pmatrix}
c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi\\
s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi\\
-s_\theta & c_\theta s_\phi & c_\theta c_\phi
\end{pmatrix}
$$

with $c_\alpha = \cos\alpha$, $s_\alpha = \sin\alpha$. $R \in SO(3)$:
$R^\top R = I$, $\det R = 1$.

### 3.2 Translation (Newton, in $\mathcal W$)

Thrust acts along $z_b$, which is the third column of $R$:

$$
m\,\dot v = T\,R e_3 - m g\,e_3 + m\,w
\quad\Longrightarrow\quad
\dot v = \frac{T}{m}\begin{pmatrix}
c_\psi s_\theta c_\phi + s_\psi s_\phi\\
s_\psi s_\theta c_\phi - c_\psi s_\phi\\
c_\theta c_\phi
\end{pmatrix} - \begin{pmatrix}0\\0\\g\end{pmatrix} + w.
$$

$w$ is a world-frame disturbance acceleration (wind). It is zero everywhere in
this project; it exists so a wind study needs no model change.

### 3.3 Rotation (Euler, in $\mathcal B$)

For a rigid body with diagonal inertia $I$:
$I\dot\omega + \omega \times (I\omega) = \tau$. The gyroscopic term is

$$
\omega \times I\omega = \big(qr(I_{zz}-I_{yy}),\ pr(I_{xx}-I_{zz}),\ pq(I_{yy}-I_{xx})\big),
$$

so

$$
\dot p = \frac{\tau_x + (I_{yy}-I_{zz})\,qr}{I_{xx}},\qquad
\dot q = \frac{\tau_y + (I_{zz}-I_{xx})\,pr}{I_{yy}},\qquad
\dot r = \frac{\tau_z + (I_{xx}-I_{yy})\,pq}{I_{zz}}.
$$

### 3.4 Euler-angle kinematics, and where it breaks

Body rates are the three Euler rates, each expressed in $\mathcal B$:

$$
\omega = \dot\phi\,e_1 + R_x^\top(\phi)\,\dot\theta\,e_2 + R_x^\top(\phi)R_y^\top(\theta)\,\dot\psi\,e_3
= W(\eta)\,\dot\eta,\qquad
W(\eta) = \begin{pmatrix}
1 & 0 & -s_\theta\\
0 & c_\phi & s_\phi c_\theta\\
0 & -s_\phi & c_\phi c_\theta
\end{pmatrix}.
$$

$\det W = \cos\theta$. Inverting:

$$
\dot\phi = p + (q\,s_\phi + r\,c_\phi)\tan\theta,\qquad
\dot\theta = q\,c_\phi - r\,s_\phi,\qquad
\dot\psi = \frac{q\,s_\phi + r\,c_\phi}{\cos\theta}.
$$

**At $\theta = \pm 90°$, $W$ is singular** and $\tan\theta$, $1/\cos\theta$
diverge — gimbal lock. This is not a defect of the choice of Euler order: $SO(3)$
has no global non-singular three-parameter chart, so every three-number
attitude representation has a singular set somewhere. The project accepts this
and stays away from it: the closed loops here peak below 35°, and the RL
episodes terminate at 60° (§7.5), where $1/\cos\theta = 2$ and the model is still
well conditioned. Aerobatics would need quaternions or a geometric controller
on $SO(3)$.

*Checked by:* `test_s3_rotation_matrix_is_a_rotation_and_matches_the_model`
(random attitudes: $R$ orthonormal, and $\dot v$ from the model equals
$(T/m)Re_3 - ge_3$), `test_s3_euler_kinematics_match_W_inverse` (random
attitudes and rates: $W\dot\eta = \omega$ and $\det W = \cos\theta$).

---

## 4. Hover equilibrium

Set $\dot x = 0$.

- $\dot p = v = 0$.
- $\dot\eta = 0 \Rightarrow \omega = W\dot\eta = 0$.
- $\dot v = 0 \Rightarrow T\,Re_3 = mg\,e_3$. The third column of a rotation is a
  unit vector, so $Re_3 = e_3$, which for $|\phi|,|\theta| < 90°$ means
  $\phi = \theta = 0$, and then $T = mg$.
- $\dot\omega = 0$ with $\omega = 0 \Rightarrow \tau = 0 \Rightarrow f_2 = f_4,\ f_1 = f_3,\ f_1 + f_3 = f_2 + f_4$.

Together:

$$
f_i^\star = \frac{mg}{4} = 0.066218\ \text{N},\qquad \phi^\star = \theta^\star = 0,\qquad
\psi^\star \text{ arbitrary},\qquad p^\star \text{ arbitrary}.
$$

$\psi$ and position do not appear in the balance, so hover is a
**four-parameter family** of equilibria. Every controller here is designed at
the member $p^\star = 0, \psi^\star = 0$; translation invariance (the dynamics
never contain $p$) and yaw invariance of the hover balance make the design valid
at any target.

$f^\star$ is half of $f_{\max}$: at hover each rotor has equal authority up and
down, which is what a thrust-to-weight of 2 buys.

*Checked by:* `test_s4_hover_trim` (cf's numerical trim equals $mg/4$ to 1e-12),
`test_s4_hover_is_a_family_in_yaw`.

---

## 5. Linearization at hover

### 5.1 The Jacobians

Write $\delta x = x - x^\star$, $\delta f = f - f^\star$. Differentiating §3 at
hover, where every $\sin$ of an angle and every rate is zero and $T = mg$:

- Translation. $\partial \dot v_x/\partial\theta = (T/m)\,c_\psi c_\theta c_\phi = g$,
  $\partial \dot v_y/\partial\phi = (T/m)(-c_\psi c_\phi) = -g$; every other angle
  derivative of $\dot v$ carries a sine and vanishes.
  $\partial\dot v_z/\partial f_i = c_\theta c_\phi/m = 1/m$, and
  $\partial \dot v_{x,y}/\partial f_i = 0$ (the thrust direction is vertical at hover).
- Kinematics. Every term of §3.4 except the leading ones is multiplied by a
  sine, a tangent or a rate, so $\dot\phi = p$, $\dot\theta = q$, $\dot\psi = r$
  to first order.
- Rotation. The gyroscopic terms are products of two rates — second order —
  and drop out. $\dot\omega = I^{-1}\tau = I^{-1}M_{2:4}\,\delta f$.

So the only non-zero entries are

$$
\begin{aligned}
\delta\dot x &= \delta v_x, & \delta\dot v_x &= g\,\delta\theta, & \delta\dot\theta &= \delta q, & \delta\dot q &= \tfrac{L}{I_{yy}}(\delta f_3 - \delta f_1),\\
\delta\dot y &= \delta v_y, & \delta\dot v_y &= -g\,\delta\phi, & \delta\dot\phi &= \delta p, & \delta\dot p &= \tfrac{L}{I_{xx}}(\delta f_2 - \delta f_4),\\
\delta\dot z &= \delta v_z, & \delta\dot v_z &= \tfrac1m\textstyle\sum_i \delta f_i, & & & &\\
& & & & \delta\dot\psi &= \delta r, & \delta\dot r &= \tfrac{c}{I_{zz}}(\delta f_1 - \delta f_2 + \delta f_3 - \delta f_4).
\end{aligned}
$$

Numerically: $g = 9.81$, $1/m = 37.04$, $L/I_{xx} = 2534$, $L/I_{yy} = 2522$,
$c/I_{zz} = 779$ (rad/s² per newton). Yaw authority is 3.3× weaker than roll —
it comes from drag, not thrust, acting on the largest inertia.

### 5.2 Structure: four decoupled chains

Group the states:

| Chain | States | Driven by | Order |
|---|---|---|---|
| $x$ | $x \leftarrow v_x \leftarrow \theta \leftarrow q$ | $f_3 - f_1$ | 4 (quadruple integrator) |
| $y$ | $y \leftarrow v_y \leftarrow \phi \leftarrow p$ | $f_2 - f_4$ | 4 |
| $z$ | $z \leftarrow v_z$ | $\sum f_i$ | 2 |
| $\psi$ | $\psi \leftarrow r$ | $f_1 - f_2 + f_3 - f_4$ | 2 |

No entry of $A$ links two chains. All twelve open-loop eigenvalues are at
zero, and $A^4 = 0 \neq A^3$: $A$ is nilpotent of index 4, set by the longest
chain. This is the linear picture of "you cannot translate without tilting
first": $\delta x$ is reached only through $\theta$, three integrations away
from the input.

### 5.3 Controllability

Substitute $\nu = M\,\delta f$ (invertible, §2.3). Then each chain is an
integrator chain driven by its own component of $\nu$ — a chain of integrators
with its input at the bottom is controllable, and the chains are independent, so
$(A, B)$ is controllable: $\operatorname{rank}[B, AB, \dots, A^{11}B] = 12$.

Underactuation and controllability are not in tension. Underactuated says you
cannot command six accelerations independently *at an instant*; controllable says
you can still reach any state *over time*.

### 5.4 Observability from pose only

With $y = (x, y, z, \phi, \theta, \psi)$ — what a position fix plus an attitude
estimate gives — each chain measures its top state and, for the $x$ and $y$
chains, its angle too. Velocities and rates follow by differentiation along the
chain, so $(A, C)$ is observable: rank 12. An estimator needs no velocity or
rate sensor in principle (a real IMU provides rates anyway).

*Checked by:* `test_s5_linearization_matches_casadi_exactly` (to 1e-10),
`test_s5_linearization_matches_finite_differences` (independently of CasADi),
`test_s5_four_decoupled_chains`, `test_s5_open_loop_poles_all_at_zero`,
`test_s5_controllable_and_pose_observable`.

---

## 6. LQR and linear MPC

Both act on the error $e = x - x_{\text{ref}}$, with angle components wrapped to
$[-\pi, \pi)$, and both use the **same** $Q$ and $R$
(`quadrl/problem.py`), chosen by Bryson's rule — each weight is
$1/(\text{largest acceptable deviation})^2$:

| | $x,y,z$ | $v$ | $\phi, \theta$ | $\psi$ | $p,q,r$ | each $f_i$ |
|---|---|---|---|---|---|---|
| acceptable deviation | 0.5 m | 1 m/s | 0.3 rad | 0.5 rad | 3 rad/s | 0.06 N |
| weight | 4 | 1 | 11.1 | 4 | 0.111 | 277.8 |

### 6.1 LQR (continuous, MATLAB `lqr`)

Minimise $J = \int_0^\infty e^\top Q e + \delta f^\top R\,\delta f\,dt$ subject to
§5. The solution is $\delta f = -Ke$ with $K = R^{-1}B^\top P$, where $P \succ 0$
solves the continuous algebraic Riccati equation

$$
A^\top P + PA - PBR^{-1}B^\top P + Q = 0.
$$

Applied as $f = \operatorname{clip}(f^\star - K e,\ 0,\ f_{\max})$ at 500 Hz.
The resulting gain inherits the chain structure:

```
          x       y      z      vx      vy     vz     phi     theta   psi     p       q       r
f1  [-0.0849  0      0.060 -0.0737  0      0.0414  0     -0.2098  0.060  0      -0.0168  0.0118]
f2  [ 0      -0.0849 0.060  0      -0.0737 0.0414  0.2097  0      -0.060  0.0168  0      -0.0118]
f3  [ 0.0849  0      0.060  0.0737  0      0.0414  0      0.2098  0.060  0       0.0168  0.0118]
f4  [ 0       0.0849 0.060  0       0.0737 0.0414 -0.2097  0      -0.060 -0.0168  0      -0.0118]
```

The $z$ column is identical for all four rotors (collective thrust), the $\psi$
and $r$ columns alternate sign (yaw by drag), and the $x$-chain gains appear only
on rotors 1 and 3 with opposite signs. Closed-loop poles: slowest
$-2.32 \pm 0.90j$, fastest $-70.97$ rad/s. Sampling a continuous design at
500 Hz is justified because $|\lambda_{\max}|\,\Delta t = 0.14 \ll 1$.

LQR does not know the tilt limit or the rotor bounds; it is clipped after the
fact. That is the point of comparison with MPC.

### 6.2 Discretization for MPC (zero-order hold)

With the input held over $\Delta = 0.02$ s,
$A_d = e^{A\Delta}$, $B_d = \int_0^\Delta e^{As}\,ds\,B$. Because $A^4 = 0$ (§5.2)
the exponential series **terminates**, so these are exact, not truncated:

$$
A_d = I + A\Delta + \tfrac{1}{2}A^2\Delta^2 + \tfrac{1}{6}A^3\Delta^3,\qquad
B_d = \big(I\Delta + \tfrac{1}{2}A\Delta^2 + \tfrac{1}{6}A^2\Delta^3 + \tfrac{1}{24}A^3\Delta^4\big)B.
$$

Down the $x$ chain this gives the familiar $\Delta^k/k!$ pattern, e.g.
$x \leftarrow f_3$: $g\,(L/I_{yy})\,\Delta^4/24 = 1.65\times10^{-4}$ m/N.

### 6.3 The QP solved every 20 ms

Decision variables: $\delta U = (\delta f_0, \dots, \delta f_{N-1}) \in \mathbb R^{80}$
and slacks $s \in \mathbb R^{240}$, with $N = 20$ (a 0.4 s horizon). Predictions
are affine in $\delta U$: $\delta X = S_x\,\delta x_0 + S_u\,\delta U$.

$$
\begin{aligned}
\min_{\delta U,\ s}\quad & \sum_{k=1}^{N-1}\|\delta x_k - \delta x_s\|_Q^2 + \|\delta x_N - \delta x_s\|_{P_d}^2 + \sum_{k=0}^{N-1}\|\delta f_k - \delta f_s\|_R^2 + \rho\,\mathbf 1^\top s\\
\text{s.t.}\quad & -f^\star \le \delta f_k \le f_{\max} - f^\star && \text{(hard: actuator)}\\
& x_{\text{lo}} - s_k \le x^\star + \delta x_k \le x_{\text{hi}} + s_k,\quad s_k \ge 0 && \text{(soft: tilt envelope)}
\end{aligned}
$$

- **Target** $(\delta x_s, \delta f_s)$ solves
  $\begin{psmallmatrix}A & B\\ C & D\end{psmallmatrix}\begin{psmallmatrix}\delta x_s\\ \delta f_s\end{psmallmatrix} = \begin{psmallmatrix}0\\ \delta y_{\text{ref}}\end{psmallmatrix}$
  by least squares. With full-state output and a hover reference this gives
  $\delta x_s = \delta x_{\text{ref}}$, $\delta f_s = 0$.
- **Terminal weight** $P_d$ is the discrete Riccati solution for $(A_d, B_d, Q, R)$,
  so the truncated horizon is charged the infinite-horizon LQR cost-to-go.
- **Slack penalty** is linear with $\rho = 10^5$. A linear (ℓ1) penalty is
  *exact*: once $\rho$ exceeds the optimal Lagrange multiplier, the slack is zero
  whenever the bound can be met — the QP behaves like a hard constraint when
  feasible and degrades gracefully when not.
- **Receding horizon:** apply $\delta f_0$, discard the rest, resolve next period.
  OSQP, warm-started, at most 200 iterations.

**What the tilt constraint actually guarantees.** It is imposed on the *linear*
prediction at the *sample instants* $k\Delta$. The plant is nonlinear and
evolves continuously between samples, so the true pitch can exceed 15° by a
small amount between samples or through linearization error. That is why the
evaluation reports a *maximum excursion* and not just a count — the count is
nonzero, the excursion is tiny (README).

*Checked by:* `test_s6_lqr_closed_loop_is_stable_and_matches_scipy`,
`test_s6_2_zoh_discretization_follows_the_integrator_chains`,
`test_s6_3_terminal_weight_is_the_discrete_riccati_solution`.

---

## 7. The reinforcement-learning problem

### 7.1 The MDP — and why it is really a POMDP

An episode is $(\mathcal S, \mathcal A, P, r, \gamma, \rho_0)$ with:

- environment state $s = (x,\ x_{\text{ref}},\ m,\ I_{zz})$,
- transition $P$: §3 integrated by RK4 over $\Delta = 0.01$ s (4 substeps) with
  $f$ held — the same `sys.f` LQR and MPC are designed from,
- observation $o$ (§7.2), action $a$ (§7.3), reward $r$ (§7.4),
  discount $\gamma$ (§7.6), start distribution $\rho_0$ (§7.3).

The policy sees $o$, which depends on $x - x_{\text{ref}}$ only. Mass and inertia
are randomized per episode but **not observed**, so the problem is partially
observable and a memoryless policy cannot identify them. It must find one action
map that works acceptably across the whole range. The consequence is predictable
before training: like LQR and MPC (which also assume nominal mass), PPO has no
integral action, so on a heavy vehicle it should hold a small altitude offset.
The README's mismatch row tests this.

### 7.2 Observation

$$
o = \operatorname{clip}\!\big((x - x_{\text{ref}}) \oslash \sigma,\ -10,\ 10\big),
\quad \text{angle components wrapped to } [-\pi,\pi)
$$

$\sigma = (2,2,2,\ 2,2,2,\ 0.5,0.5,1,\ 5,5,5)$ makes each component $O(1)$ — the
network's first layer then needs no learned rescaling. This is the same error
LQR multiplies by $-K$; PPO replaces the linear map $-K$ by a nonlinear one.
Because the dynamics do not depend on position (§4), the error coordinates make
"fly to a waypoint" and "recover to the origin" the same problem.

The function lives in `quadrl/env.py` as `observe` and is imported by the
deployed controller, so the training and run-time scalings cannot differ.

### 7.3 Action, start distribution, domain randomization

**Action** $a \in [-1, 1]^4$, mapped to rotor force about **nominal** hover:

$$
f_i = \operatorname{clip}\!\big(f^\star(1 + a_i),\ 0,\ f_{\max}\big).
$$

$a = 0$ is hover, $a_i = -1$ stops rotor $i$, $a_i = +1$ is $2f^\star = 0.13244$ N
$\approx f_{\max}$, so the action box spans the actuator range almost exactly and
nothing outside it is ever requested. Centring on hover matters for learning: an
untrained network outputs $a \approx 0$, so early episodes start hovering
instead of falling.

**Start distribution** $\rho_0$, uniform and independent:

| | $x, y$ | $z$ | $v$ | $\phi, \theta$ | $\psi$ | $\omega$ |
|---|---|---|---|---|---|---|
| range | ±2 m | ±1 m | ±0.5 m/s | ±0.3 rad (17°) | ±0.5 rad | ±1 rad/s |

The evaluation manoeuvre (start at the origin, target $(2, 0, 1)$) is the start
error $(-2, 0, -1)$ — at the edge of this distribution, deliberately.

**Domain randomization:** at each reset, $m \sim \mathcal U[0.025, 0.035]$ kg and
$I_{zz} \sim \mathcal U[2.2, 3.6]\times10^{-5}$ kg m², the uncertainty ranges
declared in `params.yaml`. Nominal $m = 0.027$ sits near the bottom of its range.

### 7.4 Reward

A non-negative stage cost

$$
c = \underbrace{\|e_p\|}_{\text{position}} + 0.1\|e_v\| + 0.5(\phi^2 + \theta^2) + 0.1\,e_\psi^2 + 0.01\|\omega\|^2 + 0.05\|a\|^2
+ 20\!\!\sum_{\alpha\in\{\phi,\theta\}}\!\!\max(0,\ |\alpha| - 15°)^2
$$

and the reward

$$
r = e^{-c} \in (0, 1].
$$

Why each piece:

- **$e^{-c}$, not $-c$.** With a negative reward and early termination, the
  return-maximising policy is to crash as fast as possible: ending the episode
  stops the stream of negative rewards. A reward in $(0, 1]$ makes every
  surviving step worth more than ending, so crashing is never optimal. The
  reward is also bounded, so returns are bounded by $1/(1-\gamma) = 200$, which
  keeps value targets well scaled. It equals 1 exactly at the target with zero
  action, so the optimum is where we want it.
  *Trade-off:* far from the target the gradient of $e^{-c}$ is small. The value
  function carries the signal there; shaping alternatives are listed in §10.
- **$\|e_p\|$ not $\|e_p\|^2$** — linear near the target keeps the pull towards
  it from vanishing as the error shrinks, which reduces steady-state offset.
- **Tilt-limit term** — zero inside ±15°, quadratic outside, weight 20: a soft
  barrier. It *discourages* violations; it cannot *forbid* them. This is the
  honest difference from MPC's constraint and is measured, not assumed.
- **$\|a\|^2$** — penalises deviation from hover thrust, the analogue of $R$.

### 7.5 Termination and truncation

The episode **terminates** if $|\phi|$ or $|\theta| > 60°$, $\|e_p\| > 5$ m,
$\|\omega\| > 30$ rad/s, or the state is non-finite. The 60° bound keeps the
simulation away from the Euler singularity (§3.4). It **truncates** at 5 s
(500 steps). The two are different: on truncation the value of the final state
is bootstrapped (the drone is still flying), on termination it is zero.
Stable-Baselines3 makes this distinction from Gymnasium's
`terminated`/`truncated` flags.

### 7.6 Discount

$\gamma = 0.995$: the effective horizon $1/(1-\gamma) = 200$ steps $= 2$ s, about
the time the closed loop needs to settle a 2 m step. At $\gamma = 0.99$ (1 s)
the agent would barely see the payoff of moving towards a distant target.

*Checked by* `tests/test_env.py`: the env's integrator equals cf's to 1e-8 over
one period, zero action holds hover exactly, the action map's endpoints,
wrapped observation, $0 < r \le 1$, the tilt term is zero inside the limit,
termination before 90°, randomized parameters stay inside their ranges, and
Gymnasium's own `check_env`.

---

## 8. PPO

### 8.1 Policy and value function

Gaussian policy $\pi_\vartheta(a \mid o) = \mathcal N\big(\mu_\vartheta(o),\ \operatorname{diag}\sigma^2\big)$,
$\mu_\vartheta$ an MLP $12 \to 64 \to 64 \to 4$ with $\tanh$ hidden layers and a
linear output; $\log\sigma$ is a learned, state-independent vector initialised
at $-0.5$ ($\sigma = 0.61$). A separate MLP of the same shape is the value
function $V_\varphi(o)$. Samples are clipped to $[-1,1]$ before reaching the
environment.

### 8.2 Objective

From a batch of rollouts under $\pi_{\text{old}}$, with probability ratio
$\rho_t(\vartheta) = \pi_\vartheta(a_t\mid o_t)/\pi_{\text{old}}(a_t\mid o_t)$:

$$
L^{\text{clip}}(\vartheta) = \mathbb E_t\Big[\min\big(\rho_t \hat A_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big],\qquad \epsilon = 0.2.
$$

The clip removes the incentive to move the policy far from the one that
collected the data — a trust region by construction instead of by a KL
constraint. Advantages come from generalized advantage estimation:

$$
\delta_t = r_t + \gamma V_\varphi(o_{t+1}) - V_\varphi(o_t),\qquad
\hat A_t = \sum_{l\ge0} (\gamma\lambda)^l\,\delta_{t+l},\qquad \lambda = 0.95,
$$

normalised per minibatch. The value net regresses on $\hat A_t + V(o_t)$. The
total loss is $-L^{\text{clip}} + 0.5\,L^{V}$ (no entropy bonus: $\sigma$ is
learned and starts wide enough).

### 8.3 Hyperparameters

| | Value | Reason |
|---|---|---|
| parallel envs × steps | 8 × 1024 = 8192 / update | enough episodes per batch to average over randomized mass |
| minibatch, epochs | 256, 10 | SB3 defaults for continuous control |
| learning rate | 3e-4 | default |
| $\gamma$, $\lambda$ | 0.995, 0.95 | §7.6 |
| clip $\epsilon$ | 0.2 | default |
| network | 64-64 tanh, separate $\pi$ and $V$ | small: the LQR it competes with is a 4×12 matrix |
| $\log\sigma_0$ | −0.5 | default 0 ($\sigma = 1$) spans the whole action range — random enough to crash in 0.25 s |
| total steps | 3·10⁶ | ≈ 366 updates |

### 8.4 Deployment

At run time the policy is **deterministic**: $a = \operatorname{clip}(\mu_\vartheta(o), -1, 1)$.
The weights are copied out of PyTorch once and evaluated in NumPy
(`quadrl/policy.py`), so the reported compute time is the network, not
framework overhead. The NumPy pass equals SB3's `predict(deterministic=True)` to
1e-5 (`test_numpy_forward_pass_matches_stable_baselines`). Wrapped as a
`cf.Controller`, the policy runs through exactly the same `cf.sim` as LQR and MPC.

---

## 9. Evaluation protocol

- **One simulator for all three.** `cf.sim`, RK4 with 0.5 ms internal steps, at a
  2 ms base period. LQR runs every step (500 Hz), PPO every 5th (100 Hz), MPC
  every 10th (50 Hz); each holds its input in between.
- **No inverse crime.** The simulator integrates more finely than any
  controller's internal model: LQR uses the continuous linearization, MPC a
  20 ms ZOH discretization, PPO was trained on 2.5 ms RK4 steps.
- **Scenarios.** (A) the 2 m + 1 m step on the nominal vehicle; (B) the same step
  with $m = 0.035$ kg (+30 %), which no controller is told about; (C) 50 random
  starts drawn from §7.3 on the nominal vehicle — success means within 5 cm of
  the target at $t = 6$ s without crashing.
- **Metrics.** Position RMSE over the run; final position error; peak $|\theta|$;
  steps beyond the tilt limit and the largest excursion past it; median and p99
  wall-clock time per controller call, measured by the simulator around `step()`.
- **Same everything else.** Same seed, initial state, reference, duration and
  integrator; `cf.results_to_table` diffs the run metadata and warns if any
  of that differs between rows.

---

## 10. What is not claimed

- **Rotor dynamics are neglected** (§2.1). Real motors lag by tens of ms, which
  costs phase margin — most for the fastest loop. The LQR's 71 rad/s pole would
  feel it first.
- **No sensor noise, no estimator.** All three controllers get the true state.
  §5.4 shows an estimator is possible; it is not in this project.
- **PPO has no constraint guarantee.** The tilt limit is a reward penalty. MPC's
  guarantee is itself only on the linear prediction at sample instants (§6.3),
  but it is a guarantee of a kind PPO does not have.
- **No integral action anywhere**, so a mass mismatch leaves an altitude offset
  for all three (§7.1). Offset-free MPC or an integrator state would remove it.
- **Euler angles** restrict the claims to |tilt| well below 90°.
- **One training seed** in the headline results. RL results vary across seeds;
  a stronger claim would report several.
- Reward-shaping alternatives not tried: potential-based shaping
  $\gamma\Phi(o') - \Phi(o)$ with $\Phi = -\|e_p\|$, which provably does not change
  the optimal policy and would strengthen the gradient far from the target.
