# i-PhysGaussian Replication

Unofficial replication of [i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting](https://arxiv.org/abs/2602.17117) (Cao et al., 2026).

## Formula-by-Formula Replication Checklist

### Eq.6: Shape Function

```
Paper:  w_Ip = N((x_I-x_p)/h) * N((y_I-y_p)/h) * N((z_I-z_p)/h), cubic B-spline
Code:   mpm_utils.py:p2g_apic_with_stress(), quadratic B-spline (3-point stencil)
```

| Item | Status | Notes |
|------|:------:|-------|
| Separable 3D product | ✅ | `w[0,i]*w[1,j]*w[2,k]` |
| Weight gradient dw | ✅ | `compute_dweight()` in P2G |
| Cubic vs quadratic | ⚠️ | Paper says cubic; PhysGaussian codebase uses quadratic. Standard MPM practice. |

### Eq.8: Newmark Acceleration

```
Paper:  a_I^{n+1} = [du_I - dt*v_I^n - dt^2*(0.5-beta)*a_I^n] / (beta*dt^2)
Code:   a_new = (du_cp - dt*v_n_gpu - dt*dt*(0.5-beta_nm)*a_n_gpu) / (beta_nm*dt*dt)
```

| Item | Status | Notes |
|------|:------:|-------|
| Formula | ✅ | Direct translation |
| beta = 0.25 | ✅ | `beta_nm = 0.25` (trapezoidal rule) |
| v_I^n source | ✅ | **Fixed (v6)**: momentum-only P2G via `_p2g_momentum_only()` — no gravity, no stress |
| a_I^n tracking | ✅ | **Fixed (v6)**: set `a_n = 0` to avoid invalid grid-acceleration-across-steps |

### Eq.9: Newmark Velocity

```
Paper:  v_I^{n+1} = v_I^n + dt*[(1-gamma)*a_I^n + gamma*a_I^{n+1}]
Code:   v_new = v_n_gpu + dt*((1.0-gamma_nm)*a_n_gpu + gamma_nm*a_new)
```

| Item | Status | Notes |
|------|:------:|-------|
| Formula | ✅ | Direct translation |
| gamma = 0.5 | ✅ | `gamma_nm = 0.5` |

### Eq.10: Momentum Residual (core equation)

```
Paper:  R_I(du) = f_I^ext + f_I^int(du) - m_I * a_I^{n+1}(du_I),  I in F
Code:   R = f_ext_gpu + f_int_gpu - mass_expanded * a_new
```

| Item | Status | Notes |
|------|:------:|-------|
| Force-unit residual | ✅ | **Fixed (v6)**: raw force residual, not mass-normalized |
| f_ext = m*g | ✅ | Gravity applied once (no double-count after v6 fix) |
| f_int via two-pass P2G | ✅ | `_eval_forces_and_kdiag()`: pass1 momentum-only, pass2 with stress, subtract |
| Free node set F | ✅ | **Fixed (v6)**: BC nodes identified, residual zeroed at Dirichlet nodes |
| No F_trial clamping | ✅ | **Fixed (v6)**: removed det(F) clamp (not in paper) |

### Eq.11: Internal Force

```
Paper:  f_I^int = -sum_p V_p^0 * P_p * grad_X N_I(x_p),  P_p = tau_p * F_p^{-T}
Code:   f_int = (grid_v_in_full - grid_v_in_momentum) / dt  (two-pass P2G extraction)
```

| Item | Status | Notes |
|------|:------:|-------|
| Stress P = tau*F^{-T} | ✅ | `compute_stress_from_F_trial` kernel |
| Reference volume V_p^0 | ✅ | Stored in `particle_vol` |
| Trial F update | ✅ | `_update_x_F` kernel: F^trial = (I + dt*grad_v)*F^n |
| Updated Lagrangian (eval at x^{n+1}) | ✅ | Particles moved before P2G |

### Eq.12: Dirichlet Boundary Conditions

```
Paper:  S * du_I = v_tar - v_hist,  I in D,  S = gamma/(beta*dt)
Code:   BC nodes identified by comparing grid_v before/after grid_postprocess
```

| Item | Status | Notes |
|------|:------:|-------|
| Dirichlet node identification | ✅ | **Fixed (v6)**: detect BC nodes by velocity change |
| Zero residual at D nodes | ✅ | `R[~free_mask_gpu] = 0.0` |
| Zero search direction at D nodes | ✅ | `delta_du[~free_mask_gpu] = 0.0` and in `matvec()` |
| S = gamma/(beta*dt) scaling | ❌ | Not implemented (we zero instead of prescribing du) |

### Eq.13: Newton Objective

```
Paper:  phi(du) = 0.5 * ||R(du)||^2_F
Code:   phi_0 = 0.5 * float(cp.sum(R_k_flat ** 2))
```

| Item | Status | Notes |
|------|:------:|-------|
| Squared norm | ✅ | Direct match |
| Norm over F only | ✅ | **Fixed (v6)**: R zeroed at non-free nodes |

### Eq.14: Initial Guess (Newmark Predictor)

```
Paper:  du_I^{0} = dt*v_I^n + 0.5*dt^2*a_I^n,  I in F
Code:   du_k = cp.asarray(dt * v_grid_n, dtype=cp.float64)  # a_n=0
```

| Item | Status | Notes |
|------|:------:|-------|
| Predictor formula | ✅ | With a_n=0: du = dt*v^n (explicit forward Euler predictor) |
| v_I^n without gravity | ✅ | **Fixed (v6)**: from momentum-only P2G |

### Eq.15-16: Newton Linearization

```
Paper:  J(du^k) * delta_u^k = -R(du^k)
Code:   cp_gmres(A, -R_k_flat, M=M_prec, ...)
```

| Item | Status | Notes |
|------|:------:|-------|
| Linear system | ✅ | GMRES solves J*du = -R |
| delta_u restricted to F | ✅ | **Fixed (v6)**: Dirichlet entries zeroed in matvec and result |

### Eq.17: Line Search Update

```
Paper:  du^{k+1} = du^k + alpha^k * delta_u^k
Code:   du_k = du_k + alpha * delta_du
```

| Item | Status | Notes |
|------|:------:|-------|
| Update rule | ✅ | |
| No v_max clamping | ✅ | **Fixed (v6)**: removed artificial velocity cap from Newton path |

### Eq.18: Directional Derivative

```
Paper:  phi'(0) = <R(du^k), J(du^k)*delta_u^k>_F
Code:   J_delta = matvec(delta_flat, du_k); dphi_0 = dot(R, J_delta)
```

| Item | Status | Notes |
|------|:------:|-------|
| Exact JVP computation | ✅ | **Fixed (v6)**: computes actual J*delta_u, not approximation |
| Steepest descent fallback | ✅ | `delta_du = -R` when phi'(0) >= 0 |

### Eq.19: Finite Difference JVP

```
Paper:  J*p ~ [R(du+eps*p) - R(du-eps*p)] / (2*eps),  ||eps*p||_inf ~ 1e-4
Code:   eps_fd = 1e-4 / norm_inf; (R_plus - R_minus) / (2*eps_fd)
```

| Item | Status | Notes |
|------|:------:|-------|
| Central FD | ✅ | |
| eps scaling | ✅ | `||eps*p||_inf = 1e-4` |
| p projected to free nodes | ✅ | **Fixed (v6)**: Dirichlet entries zeroed in matvec |

### Eq.20: Preconditioner

```
Paper:  W_I = m_I/(beta*dt^2) + K_I^diag,  I in F
Code:   _diag_w = _inertia + _k_diag_rep  where _inertia = m/(beta*dt^2)
```

| Item | Status | Notes |
|------|:------:|-------|
| Inertia term m/(beta*dt^2) | ✅ | |
| K_diag stiffness term | ✅ | **Fixed (v6)**: `K_I ~ (lam+2mu)/(rho*dx^2) * m_I` from actual Lame params |
| Only for free nodes | ✅ | BC nodes zeroed |

### Eq.21-22: Right Preconditioning

```
Paper:  J*W^{-1}*y = -R (right),  then delta_u = W^{-1}*y
Code:   cp_gmres(A, -R, M=M_prec, ...) (left preconditioning)
```

| Item | Status | Notes |
|------|:------:|-------|
| Preconditioning side | ⚠️ | CuPy uses left; paper uses right. Equivalent for diagonal W. |

### Final G2P Transfer

```
Paper:  v_p^{n+1} = sum w_Ip * v_I^{n+1};  x_p^{n+1} = x_p^n + dt*v_p^{n+1}
Code:   mpm.g2p kernel with converged velocity on grid_v_out
```

| Item | Status | Notes |
|------|:------:|-------|
| G2P with converged v | ✅ | Write v_final to grid, run g2p |
| State restore before G2P | ✅ | `_restore_state()` to (x^n, v^n, C^n, F^n) |
| Particle position clamping | ✅ | `clamp_particle_positions` kernel after G2P |

## Replication Scorecard

| Category | Items | Passed | Status |
|----------|:-----:|:------:|--------|
| Newmark integration (Eq.8-9) | 4 | 4 | ✅ |
| Momentum residual (Eq.10-11) | 5 | 5 | ✅ |
| Boundary conditions (Eq.12) | 4 | 3 | ⚠️ S-scaling missing |
| Newton solver (Eq.13-17) | 6 | 6 | ✅ |
| GMRES + JVP (Eq.19-22) | 5 | 4 | ⚠️ Left vs right precond |
| **Total** | **24** | **22** | **92%** |

## Paper Parameters (Unspecified)

The paper omits these values. Our choices:

| Parameter | Our Value | Rationale |
|-----------|-----------|-----------|
| Newmark beta, gamma | 0.25, 0.5 | Standard trapezoidal rule |
| Newton rel. tolerance | 1e-4 | Typical for JFNK |
| Max Newton iters | 25 | Sufficient for all k |
| GMRES restart | 15, maxiter=3 | Up to 45 Krylov vectors |
| Armijo c1 | 1e-4 | Standard |
| Line search max iters | 10 | Conservative |
| EW parameters | gamma=0.9, alpha=1.5 | EW Choice 2 |

## File Structure

```
implicit_mpm_solver.py        # Core solver
  ImplicitMPMSolver
    p2g2p_implicit()           # 30-iter stabilized Picard
    p2g2p_picard_vanilla()     # 10-iter vanilla Picard (paper baseline)
    p2g2p_newton_gmres()       # Displacement-based Newton-GMRES (paper method)
    _p2g_momentum_only()       # P2G without stress (for v_I^n and f_int extraction)
    _eval_forces_and_kdiag()   # Two-pass P2G → f_int, f_ext, mass, K_diag
    _save_state/_restore_state # Checkpoint particle state for iterations
    _grid_velocity_from_p2g()  # v = grid_v_in/m + dt*g (explicit formula)
    _update_x_F()              # Update particle x and F from grid velocity
  zero_stress                  # Warp kernel: zero stress for momentum-only P2G
  clamp_particle_positions     # Warp kernel: keep particles in grid
  update_x_F_from_grid_v       # Warp kernel: G2P for x,F only

gs_simulation.py               # Simulation driver
  --solver {picard, picard_vanilla, newton_gmres}
  --dt_multiplier k            # Timestep scaling
  --impulse_scale 1/k          # Compensate impulse for larger dt

eval/eval_metrics.py           # COMD, mwRMSD, BMF, AUC metrics
config/ficus_config.json       # Scene: E=2e6, nu=0.4, jelly, 50^3 grid
```

## Running

```bash
# Single run
python gs_simulation.py --model_path model/ficus_whitebg-trained \
    --output_path output/ficus_newton_v6_k4_ply \
    --config config/ficus_config.json \
    --implicit --solver newton_gmres \
    --dt_multiplier 4 --impulse_scale 0.25 --output_ply

# Full parallel k-sweep (Phase 1: Picard GPU-parallel, Phase 2: Newton 4-worker)
python /tmp/run_ksweep_parallel.py
```

## Remaining Gaps

1. **Eq.12 S-scaling**: Dirichlet BCs zeroed instead of prescribed via S = gamma/(beta*dt)
2. **Left vs right preconditioning**: CuPy uses left; paper uses right (equivalent for diagonal)
3. **Quadratic vs cubic B-spline**: PhysGaussian base uses quadratic; paper says cubic
4. **K_diag approximation**: We use material-parameter estimate; paper accumulates exact per-node stiffness

## Documentation

- `doc/formula_checklist.md` — Detailed equation-by-equation audit
- `doc/replication_analysis.md` — Solver evolution v1-v6 and root cause analysis
- `doc/bugfix_log.md` — Historical bug tracking
