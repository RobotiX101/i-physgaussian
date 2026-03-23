# i-PhysGaussian Replication — Bugfix Log

## Experiment Settings (Old Sweep, 2026-03-23)

- **Scene:** ficus (171,553 particles, jelly material, E=2e6, nu=0.4)
- **Config:** `config/ficus_config.json`, substep_dt=1e-4, frame_dt=4e-2
- **Reference:** explicit k=1 (126 frames, ~33s total)
- **Solvers tested:**
  - `newton_gmres` (i-PhysGaussian): Newton-GMRES with `max_iters=5` (old hard cap)
  - `picard` (PhysGaussian): 30-iter Picard with 0.7 relaxation + best-iterate
- **K values:** 3,4,6,8,10,12,14,16,18,20 (Newton), 1-20 (Picard)
- **Frames per k:** 30 (+ initial = 31 PLY files)
- **Hardware:** RTX 5090 32GB, seetacloud

## Results (Old Sweep — newton_max_iters=5)

### Table 1: BMF Stability

| Method | k_max stable | Fail% | Paper |
|--------|:---:|:---:|:---:|
| i-PhysGaussian (ours) | 20 | 0.0% | k_max=20, 0.0% |
| PhysGaussian (ours) | 20 | 0.0% | k_max=1, 90.9% |

### Table 7: Normalised AUC

| Method | COMD AUC | mwRMSD AUC | Paper COMD | Paper mwRMSD |
|--------|:---:|:---:|:---:|:---:|
| i-PhysGaussian (ours) | **0.837** | **0.917** | 0.018 | 0.028 |
| PhysGaussian (ours) | 0.763 | 0.849 | 0.598 | 0.851 |

### Per-k COMD (cm)

| k | Newton (ours) | Picard (ours) | Paper Newton | Paper Picard |
|---|:---:|:---:|:---:|:---:|
| 1 | — | 2.1 | ~0 | ~0 |
| 2 | — | 8.0 | ~0 | FAIL |
| 4 | 4.8 | 36.3 | ~0 | FAIL |
| 6 | 38.3 | 38.7 | ~0 | FAIL |
| 8 | 7.9 | 7.5 | ~0 | FAIL |
| 10 | 10.6 | 6.7 | ~0 | FAIL |
| 12 | 14.8 | 16.0 | ~0 | FAIL |
| 14 | 20.7 | 28.6 | ~0 | FAIL |
| 16 | 8.7 | 11.6 | ~0 | FAIL |
| 18 | 7.2 | 5.2 | ~0 | FAIL |
| 20 | 5.3 | 5.0 | ~0 | FAIL |

### Newton Convergence Analysis (old max_iters=5)

| k | Steps | Converged | Avg Residual | Max Residual | Newton Iters |
|---|:---:|:---:|:---:|:---:|:---:|
| 3 | 26 | 0% | 1.32e-02 | 3.78e-02 | all 5 (capped) |
| 4 | 27 | 100% | 0.00e+00 | 0.00e+00 | all 1 |
| 6 | 26 | 0% | 3.19e+01 | 2.92e+02 | all 5 (capped) |
| 8 | 26 | 0% | 2.76e+02 | 5.27e+02 | all 5 (capped) |
| 10 | 26 | 0% | 1.85e+02 | 2.70e+02 | all 5 (capped) |
| 12 | 27 | 100% | 0.00e+00 | 0.00e+00 | all 1 |
| 14 | 26 | 0% | 9.33e+02 | 9.33e+02 | all 5 (capped) |
| 16 | 26 | 0% | 5.34e+02 | 5.34e+02 | all 5 (capped) |

## Identified Problems

### Problem 1: Newton 5-iter hard cap prevents convergence (CRITICAL)

**Symptom:** At k≥6, Newton hits 5-iter cap with residual=30-930 (not converged).
Only k=4,12 converge (res=0 at iter 1, likely trivial steps).

**Root cause:** `newton_max_iters=5` is insufficient. At k=14 initial residual ~933,
with contraction rate ρ≈0.53, convergence needs ~22 iterations:
`933 × 0.53^22 ≈ 6e-5 < 1e-4`.

**Impact:** COMD AUC = 0.837 (paper: 0.018) — 46× worse.
Newton is barely better than Picard because it never converges.

**Fix (commit 55f194a):** `newton_max_iters` changed to 25, with adaptive
stagnation detection (exit if contraction > 0.95 after 3 iterations).

### Problem 2: Picard baseline too stable (30 iters + 0.7 relaxation)

**Symptom:** PhysGaussian shows 0% fail rate (paper: 90.9%).
Our Picard k=20 COMD = 5.0cm (paper: should FAIL at k≥2).

**Root cause:** 30 Picard iterations with 0.7 under-relaxation + best-iterate
strategy + Frozen Lagrangian gives much better convergence than the paper's
vanilla PhysGaussian baseline.

**Impact:** Cannot reproduce the paper's Table 1 showing Newton's advantage
over Picard, because our Picard is artificially stabilized.

**Fix (commit e798053 + 55f194a):**
- Added `picard_vanilla` solver: 10 iters, no relaxation
- Changed from Frozen Lagrangian to Updated Lagrangian (particle x moves
  with v^k each iteration), matching the original PhysGaussian behavior.
  This should cause divergence at large dt (spectral radius > 1).

### Problem 3: Stagnation detection direction inverted

**Symptom:** Newton stagnation check computed `prev_res / res` and checked
`< 0.95`. This detects divergence (res growing), not stagnation (res flat).

**Root cause:** Logic error — the ratio was inverted.

**Fix (commit 55f194a):** Changed to `contraction = res / prev_res`, check
`> 0.95` (less than 5% improvement per iteration → stagnating).

### Problem 4: GMRES maxiter=1 insufficient for large k

**Symptom:** At large k, the Jacobian condition number is high. GMRES with
15 Krylov vectors (1 restart cycle) cannot adequately solve the linear system,
leading to poor Newton search directions.

**Fix (commit 55f194a):** `maxiter` changed from 1 to 3 (up to 45 Krylov vectors).

### Problem 5: Newmark predictor corrupted particle_v

**Symptom:** To scatter particle acceleration a^n onto the grid, the code
temporarily overwrote `particle_v` with `a^n` and called `p2g_apic_with_stress`.
This corrupted the APIC momentum transfer because `p2g_apic` uses `particle_v`
for momentum (`m_I * v_I = Σ w_ip * m_p * (v_p + C_p * dx)`).

**Fix (commit 55f194a):** Replaced with grid-level finite difference of
consecutive explicit velocities: `a_grid = (v_explicit_n - v_explicit_{n-1}) / dt`.
No writes to particle state.

## Next Steps

1. **Re-run k-sweep with fixed code** — all current results are from old code
2. **Add vanilla Picard sweep** — should show failures at k≥2
3. **Verify Newton convergence** — expect res<1e-5 at all k values
4. **Update eval_metrics.py** to include 3-way comparison
