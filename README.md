# i-PhysGaussian Replication

Unofficial replication of [i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting](https://arxiv.org/abs/2602.17117) (Cao et al., 2026).

## Paper-to-Code Mapping

### Core Solver: Displacement-Based Newton-GMRES (JFNK)

**Paper Section 2.2 → `implicit_mpm_solver.py:p2g2p_newton_gmres()`**

The paper solves for **grid displacement increment** Δu_I using inexact Newton with GMRES (Jacobian-Free Newton-Krylov). Our implementation follows this exactly:

| Paper | Equation | Code Location | Implementation |
|-------|----------|---------------|----------------|
| Newmark acceleration | Eq.8: `a^{n+1} = (Δu - Δt·v^n - Δt²(½-β)·a^n) / (β·Δt²)` | `eval_residual()` | `a_new = (du - dt*v_grid_n - dt²*(0.5-β)*a_n) / (β*dt²)` |
| Newmark velocity | Eq.9: `v^{n+1} = v^n + Δt·[(1-γ)·a^n + γ·a^{n+1}]` | `eval_residual()` | `v_new = v_grid_n + dt*((1-γ)*a_n + γ*a_new)` |
| Momentum residual | Eq.10: `R_I = f^ext + f^int - m·a^{n+1}` | `eval_residual()` | `R = v_explicit - v_new` (mass-scaled equivalent) |
| Internal force | Eq.11: `f^int = -Σ V_p P·∇N_I` | P2G kernel (`p2g_apic_with_stress`) | Warp GPU kernel computes stress divergence |
| Initial guess | Eq.14: `Δu^(0) = Δt·v^n + ½Δt²·a^n` | Step 4 | `du_k = dt*v_grid_n + 0.5*dt²*a_n` |
| Newton linearization | Eq.15-16: `J·δu = -R` | GMRES block | `sp_gmres(A, -R_k_flat, ...)` |
| Newton update | Eq.17: `Δu^{k+1} = Δu^k + α·δu` | Line search block | `du_k += alpha * delta_du` |

### Newmark Parameters

**Paper Section 2.2**: β, γ parameterize the Newmark family. Paper does not state exact values.

**Our choice**: β=1/4, γ=1/2 (trapezoidal rule / constant average acceleration). This is unconditionally stable and second-order accurate. Set in `p2g2p_newton_gmres()`:
```python
beta_nm  = 0.25   # Newmark β
gamma_nm = 0.5    # Newmark γ
```

### GMRES Inner Solver

**Paper Section 2.3 → `p2g2p_newton_gmres()` GMRES block**

| Paper | Code | Notes |
|-------|------|-------|
| Central FD JVP (Eq.19): `J·p ≈ [R(Δu+εp)-R(Δu-εp)]/(2ε)` | `matvec()` closure | ε chosen so ‖εp‖∞ ≈ 1e-4 |
| Right preconditioner (Eq.20): `W_I = m_I/(β·Δt²)` | `M_prec` LinearOperator | Diagonal mass preconditioner |
| Right-precond system (Eq.21): `J·W⁻¹·y = -R` | `sp_gmres(A, -R, M=M_prec, ...)` | scipy uses left-preconditioning; equivalent for diagonal M |
| Eisenstat-Walker tolerance | `eta_k` variable | EW Choice 2 with safeguards |
| GMRES restart | `restart=15, maxiter=3` | 15 Krylov vectors, up to 3 restart cycles |

### Line Search

**Paper Section 2.2, Eq.17-18 → Armijo block in `p2g2p_newton_gmres()`**

| Paper | Code | Notes |
|-------|------|-------|
| Objective: φ(Δu) = ½‖R(Δu)‖² (Eq.13) | `phi_0 = 0.5 * sum(R²)` | Minimize residual norm |
| Directional derivative (Eq.18) | `dphi_0 ≈ -‖R‖²` | Approximation since J·δu ≈ -R |
| Armijo backtracking | `phi_trial ≤ phi_0 + c1·α·dphi_0` | c1 = 1e-4, up to 10 halvings |
| Steepest descent fallback | `delta_du = -R` if dphi_0 ≥ 0 | Paper specifies this fallback explicitly |

### Picard Baseline Solvers

**Paper compares against PhysGaussian (explicit MPM).**

We provide three Picard variants for comparison:

| Solver | Method | Code | Purpose |
|--------|--------|------|---------|
| `picard` | 30-iter Frozen Lagrangian, 0.7 relaxation, best-iterate | `p2g2p_implicit()` | Stabilized baseline |
| `picard_vanilla` | 10-iter Updated Lagrangian, no relaxation | `p2g2p_picard_vanilla()` | Paper-comparable PhysGaussian baseline |
| `newton_gmres` | Displacement-based Newton-GMRES | `p2g2p_newton_gmres()` | Paper i-PhysGaussian method |

### Updated vs Frozen Lagrangian

**Paper**: Uses Updated Lagrangian — particle positions move with the current velocity estimate, so the mass matrix and stress evaluation change each iteration.

| Solver | Lagrangian | Code Detail |
|--------|-----------|-------------|
| `newton_gmres` | Updated | `_update_x_F()` moves particles; no x^n restore |
| `picard_vanilla` | Updated | Same — matches paper PhysGaussian baseline |
| `picard` | Frozen | `wp.copy(particle_x, _buf_x)` restores x^n each iteration |

### Evaluation Metrics

**Paper Appendix D → `eval/eval_metrics.py`**

| Metric | Paper Definition | Code |
|--------|-----------------|------|
| BMF (Body-hit Mass Fraction) | Fraction of mass at domain boundary; fail if exceedance ratio > 0.5 | COMD > 50cm proxy |
| COMD (Center of Mass Drift) | Euclidean distance between COM of implicit and explicit reference | `eval_metrics.py` |
| mwRMSD (mass-weighted RMSD) | Mass-weighted root mean square displacement vs reference | `eval_metrics.py` |
| k_max | Largest k that passes BMF stability gate | `eval_metrics.py` |
| AUC | Normalized area under k-curve for COMD/mwRMSD | `eval_metrics.py` |

### K-Sweep Configuration

**Paper Tables 3-5: Ficus scene**

| Parameter | Paper | Our Config |
|-----------|-------|------------|
| substep_dt | k × 1e-4 | `--dt_multiplier k` |
| frame_dt | 4.0e-2 | `config/ficus_config.json` |
| frame_num | 125 (full) / 30 (sweep) | `frame_num` in config |
| k values | {1,2,4,6,8,10,12,14,16,18,20} | Same |
| impulse_scale | 1/k | `--impulse_scale` |
| Material | jelly, E=2e6, ν=0.4 | Same |
| Grid | 50³ | Same |

## What the Paper Does NOT Specify

These parameters were chosen by us (reasonable defaults):

1. **Newmark β, γ** → we use β=0.25, γ=0.5 (trapezoidal)
2. **Newton tolerance** → we use 1e-3 (L2 norm of velocity residual)
3. **Max Newton iterations** → we use 25
4. **GMRES restart** → we use 15 Krylov vectors, 3 cycles
5. **Armijo c1** → we use 1e-4
6. **Line search max iterations** → we use 10
7. **Eisenstat-Walker variant** → we use EW Choice 2 with α=1.5, γ=0.9

## File Structure

```
implicit_mpm_solver.py    # Core solver: Picard + Newton-GMRES
  ├── ImplicitMPMSolver   # Extends MPM_Simulator_WARP
  │   ├── p2g2p_implicit()         # 30-iter stabilized Picard
  │   ├── p2g2p_picard_vanilla()   # 10-iter vanilla Picard (paper baseline)
  │   ├── p2g2p_newton_gmres()     # Displacement-based Newton (paper method)
  │   ├── _save_state()            # Save x,v,C,F for iteration
  │   ├── _restore_state()         # Restore to start-of-step
  │   ├── _grid_velocity_from_p2g()# v = grid_v_in/m + dt*g
  │   ├── _update_x_F()            # G2P for x and F only
  │   └── _write_grid_v()          # Write numpy to warp grid
  ├── clamp_particle_positions     # Warp kernel: keep particles in grid
  ├── update_x_F_from_grid_v       # Warp kernel: G2P for x,F only
  └── clamp_F_trial_J              # Warp kernel: clamp det(F) ∈ [0.1, 10]

gs_simulation.py          # Main simulation driver
  └── --solver {picard, picard_vanilla, newton_gmres}

eval/eval_metrics.py      # COMD, mwRMSD, BMF, AUC computation
config/ficus_config.json  # Scene parameters
```

## Running Experiments

```bash
# Single run (Newton, k=4)
python gs_simulation.py --model_path model/ficus_whitebg-trained \
    --output_path output/ficus_newton_v3_k4_ply \
    --config config/ficus_config.json \
    --implicit --solver newton_gmres \
    --dt_multiplier 4 --impulse_scale 0.25 --output_ply

# Full parallel k-sweep
python /tmp/run_ksweep_parallel.py
```

## Known Limitations

1. Newton residual is formulated as velocity difference (`v_explicit - v_newmark`) rather than raw momentum residual. This is mathematically equivalent (scaled by mass) but may have different numerical conditioning.
2. GMRES uses scipy's left-preconditioned variant; paper specifies right-preconditioning. For diagonal preconditioners these are equivalent.
3. Hardware: RTX 5090 (32GB) vs paper's RTX 4090 (24GB). Should not affect results.
