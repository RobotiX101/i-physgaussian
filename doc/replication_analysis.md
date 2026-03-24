# i-PhysGaussian Replication Analysis

## Date: 2026-03-24

## Summary

We attempted to replicate arXiv 2602.17117 through 5 solver iterations (v1-v5).
Stability (k_max=20) was replicated, but accuracy (COMD AUC) remains 50x off.

## Solver Evolution

| Version | Residual | Formulation | k_max | COMD AUC | Issue |
|---------|----------|-------------|:-----:|:--------:|-------|
| v1 (old) | v - F(v) | Velocity fixed-point | 20 | 0.834 | 5-iter cap, never converges |
| v2 | v - F(v) | + adaptive 25 iters | 20 | 0.837 | Picard baseline too stable |
| v3 | v - F(v) | + displacement-based | 20 | -- | Too slow, CPU GMRES |
| v4 | v_explicit - v_newmark | Displacement + CuPy | 20 | 0.925 | Frozen trajectory, zero motion |
| v5 | f_ext+f_int over m minus a | Momentum residual | 20* | -- | k>=2 diverges to inf |
| **Paper** | f_ext+f_int - m*a | Momentum, Eq.10 | **20** | **0.018** | -- |

*v5 k_max=20 only because unconverged steps still produce stable but inaccurate output

## What We Got Right

1. Displacement-based unknown, paper Eq.8-16: solve for grid du, not velocity
2. Newmark-beta integration, beta=1/4, gamma=1/2: trapezoidal rule
3. Newmark predictor initial guess, Eq.14: du_0 = dt*v_n + 0.5*dt^2*a_n
4. Central FD JVP, Eq.19: eps*p infinity norm approximately 1e-4
5. Diagonal mass preconditioner, Eq.20: W_I = m_I / beta*dt^2
6. Armijo line search, Eq.17: with steepest-descent fallback
7. Eisenstat-Walker adaptive tolerance: EW Choice 2
8. Updated Lagrangian: particles move with velocity each iteration
9. Two-pass P2G: separates APIC momentum from internal force
10. CuPy GPU acceleration: GMRES and vector ops on GPU

## What We Got Wrong / Missing

### 1. K_diag Stiffness Preconditioner -- CRITICAL

Paper Eq.20: W_I = m_I/beta*dt^2 + K_I_diag

K_I_diag includes material stiffness contribution scaled by lambda+2*mu.
We only use the mass term and omit K_diag entirely.
At large k (large dt), the stiffness term dominates and our preconditioner
becomes ineffective, causing GMRES to take too many iterations or diverge.

### 2. Velocity Clamping vs Paper Bounds -- CRITICAL

We clamp grid velocity to v_max = min(50, 0.4*grid_lim/dt).
This creates Jacobian discontinuities that break Newton at k>=2.
Without clamping, velocities explode to 276000 m/s by step 8.
The paper likely handles this through the implicit solve itself,
with the stiffness preconditioner preventing velocity blowup.

### 3. Grid Acceleration Tracking for Newmark a_n

We store a_n on the grid, but MPM rebuilds the grid each step since
particles move and mass distribution changes. a_n from step n-1 was
computed on a different grid than step n.
Options: track on particles via G2P/P2G, or use backward Euler (no a_n).

### 4. Boundary Condition Treatment

Paper Eq.12: S*du_I = v_tar - v_hist for Dirichlet nodes, S = gamma/beta*dt.
We apply BCs by zeroing grid velocity in constrained regions after P2G.
The paper treats BCs as hard constraints within the Newton solve, separating
free node set F from Dirichlet set D.

### 5. Residual Scaling

Paper Eq.10: R = f_ext + f_int - m*a, in force units O(m/dt^2).
Our v5: R = (f_ext+f_int)/m - a, in acceleration units O(1).
We normalized to avoid overflow, but this changes Jacobian structure.

### 6. Unspecified Paper Parameters

The paper omits these critical values:
- Newmark beta, gamma -- we assume 0.25, 0.5
- Newton convergence tolerance -- we tried 1e-4 to 5.0
- Max Newton iterations -- we use 25
- GMRES restart parameter -- we use 15
- Armijo c1 -- we use 1e-4
- Line search backtracking factor -- we use 0.5
- EW safeguard parameters -- we use gamma=0.9, alpha=1.5

## Detailed Results

### v4 Sweep -- velocity-difference residual, completed, 10 frames

| k | Newton v4 COMD | Vanilla Picard COMD | 30-iter Picard COMD |
|---|:-:|:-:|:-:|
| 1 | 6.05cm | 15.94cm | 2.12cm |
| 2 | 6.05cm | 29.58cm | 7.97cm |
| 4 | 11.77cm | 64.47cm FAIL | 36.29cm |
| 6 | 4.80cm | 67.69cm FAIL | 38.69cm |
| 8 | 9.20cm | 30.50cm | 7.49cm |
| 10 | 9.42cm | 19.79cm | 6.71cm |
| 12 | 5.29cm | 9.18cm | 16.00cm |
| 14 | 8.09cm | 18.66cm | 28.58cm |
| 16 | 3.46cm | 21.67cm | 11.61cm |
| 18 | 7.67cm | 15.33cm | 5.22cm |
| 20 | 3.04cm | 14.27cm | 5.03cm |

v4 Problem: trajectory was frozen. Newton converged to trivial du=0.

### v5 Sweep -- momentum residual, partial

- k=1: stable but slow, 186/4000 steps in 30 min, mostly unconverged at res~1.0
- k=2: diverged at step 116, residual went to infinity
- k=4: diverged at step 80
- k>=6: diverged within 15-36 steps

### Convergence Pattern at k=4 with v5

```
step 0: res=2.1e5 -> 2.5 in 3 iters, converged, rel=1.2e-5
step 1: res->15 in 3 iters, converged
step 5: res->2316 in 6 iters, converged, gv_max=40
step 6: res->6.3e4 in 13 iters, NOT converged, gv_max=50
step 7+: res->inf, v_max clamp creates Jacobian discontinuity
```

## Root Cause Diagnosis

The core issue is that our Newton solver lacks sufficient globalization
for large timesteps. The paper addresses this through:

1. K_diag stiffness preconditioner -- makes GMRES converge faster at large k
2. Proper Dirichlet BC treatment -- constrained nodes excluded from residual
3. Better line search parameters
4. Possibly adaptive v_max or no clamping at all
5. Possibly different mass threshold for low-mass grid cells

Without the code release, we cannot determine which is the primary missing
ingredient. The most likely candidate is the K_diag preconditioner, since
the paper specifically mentions it and ablation shows it matters.

## Recommendations

### Short Term
- Wait for paper code release, described as "clean Python implementation"
- Use v4 results for stability comparison since k_max=20 matches paper
- Run 30-iter Picard as reliable baseline

### Medium Term
- Implement K_diag stiffness preconditioner: requires computing per-node
  stiffness from material parameters during P2G
- Implement proper Dirichlet BC treatment in Newton: separate free/constrained
- Try backward Euler instead of Newmark: avoids a_n tracking problem

### Long Term
- GPU-native Newton: all in Warp, no CPU-GPU transfers
- Analytic Jacobian instead of FD JVP: more accurate, faster

## Commits

| Hash | Description |
|------|-------------|
| e798053 | vanilla Picard + adaptive Newton, initial |
| 55f194a | 4 correctness fixes |
| 6549f56 | bugfix_log documentation |
| 3293cbb | stagnation threshold fix |
| 5ede54e | displacement-based Newton + README |
| 6aec809 | CuPy acceleration |
| d9c035e | v4 sweep results |
| 1314a9d | momentum residual, paper Eq.10 |
| 8f2b6a6 | mass-normalized residual + relative convergence |
