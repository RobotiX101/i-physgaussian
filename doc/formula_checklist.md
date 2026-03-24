# i-PhysGaussian Replication — Formula-by-Formula Checklist

Paper: arXiv 2602.17117, Sections 2.1-2.3

## Equation Audit

### Eq.6: Shape Function w_Ip

```
Paper:  w_Ip = N((x_I - x_p)/h) · N((y_I - y_p)/h) · N((z_I - z_p)/h)
        N(·) = cubic B-spline
```

| Check | Status | Code Location | Notes |
|-------|--------|---------------|-------|
| Cubic B-spline weights | ✅ OK | `mpm_utils.py:p2g_apic_with_stress` L330-345 | Uses quadratic B-spline (0.5*w² terms), NOT cubic. Paper says cubic (Eq.6) but PhysGaussian codebase uses quadratic. |
| Separable 3D product | ✅ OK | `w[0,i]*w[1,j]*w[2,k]` | Correct tensor product |
| Weight gradient ∇w | ✅ OK | `compute_dweight()` | Used for stress divergence in P2G |

**⚠️ DISCREPANCY**: Paper says "cubic B-spline" (Eq.6) but the PhysGaussian codebase (which we extend) uses **quadratic** B-spline (3-point stencil per axis). This matches standard MPM practice. The paper may be imprecise here, or they actually use cubic (4-point stencil). This affects grid bandwidth and convergence order but likely not stability.

---

### Eq.8: Newmark Acceleration

```
Paper:  a_I^{n+1}(Δu_I) = [Δu_I - Δt·v_I^n - Δt²·(1/2 - β)·a_I^n] / (β·Δt²)
Code:   a_new = (du_cp - dt*v_n_gpu - dt*dt*(0.5 - beta_nm)*a_n_gpu) / (beta_nm*dt*dt)
```

| Check | Status | Notes |
|-------|--------|-------|
| Formula match | ✅ OK | Direct translation |
| β = 0.25 | ✅ OK | `beta_nm = 0.25` |
| v_I^n source | ⚠️ WRONG | We use `_grid_velocity_from_p2g(dt)` which includes gravity. Paper v_I^n is pure momentum/mass WITHOUT gravity. |
| a_I^n tracking | ❌ BUG | Stored on GRID between steps. But grid is rebuilt each step (particles move). a_n from step n-1 is on a DIFFERENT grid. Paper doesn't specify how they handle this. |

**🔴 BUG: v_I^n includes gravity**
`_grid_velocity_from_p2g(dt)` computes `v = grid_v_in/m + dt*g`. The `+ dt*g` term means v_I^n already has gravity baked in. But in the Newmark formulation, v_I^n should be the velocity at time t^n (no gravity). Gravity enters through f_I^ext in the residual. This means we're double-counting gravity.

---

### Eq.9: Newmark Velocity

```
Paper:  v_I^{n+1}(Δu_I) = v_I^n + Δt·[(1-γ)·a_I^n + γ·a_I^{n+1}(Δu_I)]
Code:   v_new = v_n_gpu + dt*((1.0 - gamma_nm)*a_n_gpu + gamma_nm*a_new)
```

| Check | Status | Notes |
|-------|--------|-------|
| Formula match | ✅ OK | Direct translation |
| γ = 0.5 | ✅ OK | `gamma_nm = 0.5` |
| Same v_I^n bug | ⚠️ | Inherits gravity double-count from Eq.8 check |

---

### Eq.10: Momentum Residual

```
Paper:  R_I(Δu) = f_I^ext(Δu) + f_I^int(Δu) - m_I · a_I^{n+1}(Δu_I),  I ∈ F
Code:   R = (f_ext_gpu + f_int_gpu) / safe_mass - a_new_expanded
```

| Check | Status | Notes |
|-------|--------|-------|
| f_ext + f_int - m·a = 0 | ⚠️ MODIFIED | We divide by mass (acceleration units). Paper uses force units. |
| Free node set F | ❌ MISSING | We zero R at low-mass nodes. Paper restricts to free nodes F (excluding Dirichlet set D). |
| f_ext computation | ⚠️ BUG | We compute `f_ext = m*g`. But if v_I^n already includes gravity (see Eq.8 bug), then f_ext is double-counted. |

**🔴 BUG: Gravity double-counted**
The gravity term appears in THREE places:
1. `v_grid_n = _grid_velocity_from_p2g(dt)` → includes `+ dt*g`
2. `f_ext[...,d] = mass * g[d]` → explicit gravity force
3. Inside `a_new` via the `v_n_gpu` term (which has gravity baked in)

The paper counts gravity ONCE via f_ext. We need to either:
(a) Remove `dt*g` from v_grid_n (use momentum/mass only), OR
(b) Set f_ext = 0 (since gravity is already in v_n)

---

### Eq.11: Internal Force

```
Paper:  f_I^int(Δu) = -Σ_p V_p^0 · P_p(Δu) · ∇_X N_I(x_p)
        P_p = τ_p · F_p^{-T}
Code:   f_int = (grid_v_in_full - momentum) / dt
        (extracted via two-pass P2G)
```

| Check | Status | Notes |
|-------|--------|-------|
| Force extraction | ⚠️ INDIRECT | We use two-pass P2G subtraction instead of direct force accumulation |
| Stress P = τ·F^{-T} | ✅ OK | Handled by `compute_stress_from_F_trial` kernel |
| Reference volume V_p^0 | ✅ OK | Stored in `particle_vol` |
| Trial F update | ✅ OK | Via `_update_x_F` kernel |
| ∇N_I evaluated at x_p | ⚠️ QUESTION | Our P2G evaluates at UPDATED x_p^{n+1}. Paper Eq.11 writes ∇_X N_I(x_p) — is this x_p^n or x_p^{n+1}? For Updated Lagrangian it should be x_p^{n+1}. |
| F_trial clamping | ❌ NOT IN PAPER | We clamp det(F) to [0.1, 10]. Paper doesn't mention this. Could prevent correct large-deformation behavior. |

---

### Eq.12: Dirichlet BCs

```
Paper:  S · Δu_I = v_I^tar - v_I^hist,  I ∈ D
        S = γ/(β·Δt)
Code:   grid_postprocess kernels zero velocity in constrained regions
```

| Check | Status | Notes |
|-------|--------|-------|
| Dirichlet constraint | ❌ WRONG | Paper enforces du = prescribed value. We zero velocity AFTER the solve. |
| Separate F/D node sets | ❌ MISSING | Paper distinguishes free (F) and Dirichlet (D) nodes. We don't. |
| S = γ/(β·Δt) scaling | ❌ MISSING | Not implemented |
| v_hist tracking | ❌ MISSING | Paper tracks velocity history for BC enforcement |

**🔴 MAJOR GAP**: Our BC treatment is fundamentally different. We apply BCs as a post-processing step on grid velocity, not as hard constraints within the Newton solve. This means:
- Residual R includes contributions from Dirichlet nodes (should be excluded)
- GMRES search directions may violate constraints
- Newton convergence is degraded because BCs are not satisfied exactly

---

### Eq.13: Newton Objective

```
Paper:  φ(Δu) = (1/2) · ||R(Δu)||²_F
Code:   phi_0 = 0.5 * float(cp.sum(R_k_flat ** 2))
```

| Check | Status | Notes |
|-------|--------|-------|
| Squared norm | ✅ OK | Direct match |
| Norm over F only | ❌ | We sum over ALL nodes including Dirichlet. Paper sums over free set F only. |

---

### Eq.14: Initial Guess (Newmark Predictor)

```
Paper:  Δu_I^(0) = Δt·v_I^n + (1/2)·Δt²·a_I^n,  I ∈ F
Code:   du_k = cp.asarray(dt * v_grid_n + 0.5 * dt * dt * a_n, dtype=cp.float64)
```

| Check | Status | Notes |
|-------|--------|-------|
| Formula | ✅ OK | Direct match |
| v_I^n bug | ⚠️ | Same gravity double-count issue |
| a_I^n tracking | ❌ | Grid acceleration from previous step invalid (different grid) |
| Only for I ∈ F | ❌ | Applied to all nodes |

---

### Eq.15-16: Newton Linearization

```
Paper:  J(Δu^(k)) · δu^(k) = -R(Δu^(k))
Code:   cp_gmres(A, -R_k_flat, ...)
```

| Check | Status | Notes |
|-------|--------|-------|
| Linear system | ✅ OK | GMRES solves J·δu = -R |
| δu restricted to F | ❌ | Not restricted. Dirichlet nodes included. |

---

### Eq.17: Line Search Update

```
Paper:  Δu^(k+1) = Δu^(k) + α^(k) · δu^(k)
Code:   du_k = cp.clip(du_k + alpha * delta_du, -v_max*dt, v_max*dt)
```

| Check | Status | Notes |
|-------|--------|-------|
| Update rule | ✅ OK | |
| v_max clamping | ❌ NOT IN PAPER | We clip du to [-v_max*dt, v_max*dt]. Paper doesn't mention this. Creates Jacobian discontinuity. |

---

### Eq.18: Directional Derivative

```
Paper:  φ'(0) = ⟨R(Δu^(k)), J(Δu^(k))·δu^(k)⟩_F
Code:   dphi_0 = -2.0 * phi_0  (approximation: since J·δu ≈ -R, φ'(0) ≈ -||R||²)
```

| Check | Status | Notes |
|-------|--------|-------|
| Exact derivative | ❌ APPROXIMATE | We approximate φ'(0) ≈ -||R||². Paper computes it properly via the JVP. Should use `φ'(0) = dot(R, J·δu)` but that costs an extra JVP evaluation. |
| Steepest descent fallback | ✅ OK | Paper specifies fallback to -R when φ'(0) ≥ 0. We implement this. |

---

### Eq.19: Finite Difference JVP

```
Paper:  J(Δu)·p ≈ [R(Δu + ε·p) - R(Δu - ε·p)] / (2ε)
        ||ε·p||_∞ ~ 10^{-4}
Code:   eps_fd = 1e-4 / norm_inf
        return (R_plus - R_minus) / (2.0 * eps_fd)
```

| Check | Status | Notes |
|-------|--------|-------|
| Central FD | ✅ OK | Correct centered difference |
| ε scaling | ✅ OK | `eps = 1e-4 / ||p||_∞` gives `||ε·p||_∞ = 1e-4` |
| p restricted to F | ❌ | Paper says "projected to free subspace". We don't zero Dirichlet entries. |

---

### Eq.20: Preconditioner

```
Paper:  W_I = m_I/(β·Δt²) + K_I^diag,  I ∈ F
Code:   _diag_w = cp.ones(n) / (beta_nm * dt * dt)
```

| Check | Status | Notes |
|-------|--------|-------|
| Mass term m_I/(β·Δt²) | ❌ WRONG | We use 1/(β·Δt²) (no mass) because residual is mass-normalized. But this makes preconditioner identical for all nodes regardless of mass. |
| K_diag stiffness | ❌ MISSING | Paper: "K_I^diag includes material contribution scaled by (λ+2μ)". We omit this entirely. At large dt, stiffness dominates inertia and our preconditioner becomes ineffective. |
| Only for F | ❌ | Applied to all nodes |

**🔴 CRITICAL MISSING**: K_diag is the per-node stiffness diagonal. For neo-Hookean with Lame parameters λ,μ:
```
K_I^diag ≈ Σ_p V_p^0 · (λ + 2μ) · ||∇N_I(x_p)||²
```
This must be accumulated during P2G alongside the momentum/force computation.

---

### Eq.21-22: Right Preconditioning

```
Paper:  J·W^{-1}·y = -R,  then δu = W^{-1}·y
Code:   cp_gmres(A, -R, M=M_prec, ...)  (left preconditioning)
```

| Check | Status | Notes |
|-------|--------|-------|
| Right vs Left | ⚠️ DIFFERENT | Paper uses RIGHT preconditioning (Eq.21). CuPy/scipy gmres uses LEFT. For diagonal W these give same iterates but different residual norms. |
| Recovery δu = W^{-1}·y | ❌ MISSING | With left preconditioning, GMRES returns δu directly, not y. We don't need Eq.22. But the search direction may differ. |

---

### Final G2P Transfer

```
Paper:  v_p^{n+1} = Σ_I w_Ip · v_I^{n+1}
        x_p^{n+1} = x_p^n + Δt · v_p^{n+1}
        F_p^{n+1} = (I + Δt·∇v^{n+1}) · F_p^n
Code:   mpm.g2p kernel
```

| Check | Status | Notes |
|-------|--------|-------|
| G2P with converged v | ✅ OK | We write converged v to grid_v_out then run g2p |
| State restore before G2P | ✅ OK | `_restore_state()` resets to (x^n, v^n, C^n, F^n) |
| F update in G2P | ⚠️ QUESTION | g2p computes `F^{n+1} = (I + dt·∇v)·F^n`. But we also ran `compute_stress_from_F_trial` after restore, which may modify F. Need to verify g2p reads the restored F^n, not F_trial. |

---

## Summary: Critical Bugs Found

| # | Bug | Impact | Fix Difficulty |
|---|-----|--------|----------------|
| 1 | **Gravity double-counted** in v_I^n (includes dt*g) AND f_ext (m*g) | Newton solves wrong equation; explains k=1 COMD drift | Easy: use v_I^n = momentum/mass (no gravity) |
| 2 | **K_diag missing** from preconditioner | GMRES doesn't converge at large k | Medium: accumulate stiffness during P2G |
| 3 | **Dirichlet BCs not as constraints** in Newton | Residual polluted by constrained nodes | Hard: separate F/D node sets |
| 4 | **Grid a^n invalid** across steps | Wrong Newmark predictor after step 0 | Medium: track on particles or use backward Euler |
| 5 | **v_max clamping** creates Jacobian discontinuity | Newton diverges at k≥2 when velocities approach cap | Medium: raise cap or remove with K_diag fix |
| 6 | **F_trial clamped** to det∈[0.1,10] | Not in paper; may suppress large deformations | Easy: remove or widen range |
| 7 | **φ'(0) approximated** instead of computed | Line search may accept bad steps | Easy: compute exact JVP·δu |

## Priority Fix Order

1. **Fix gravity double-count** (Bug 1) — should immediately improve k=1 accuracy
2. **Implement K_diag** (Bug 2) — needed for k≥2 convergence
3. **Remove F_trial clamping** (Bug 6) — quick, may help
4. **Fix Dirichlet BCs** (Bug 3) — needed for correctness
5. **Fix a^n tracking** (Bug 4) — improves initial guess
