# i-PhysGaussian — Unofficial Replication

> **AutoResearch Project** · Unofficial replication of [i-PhysGaussian](https://arxiv.org/abs/2602.17117) by **GLM-5** and **Claude**

This repository is an unofficial implementation of the **i-PhysGaussian** paper, produced autonomously by an AI research pipeline (AutoResearch) using ZhipuAI GLM-5 and Anthropic Claude. The goal is to replicate the implicit MPM solver from arXiv 2602.17117 on top of [PhysGaussian](https://github.com/XPandora/PhysGaussian).

## What is i-PhysGaussian?

i-PhysGaussian extends PhysGaussian with a **fully implicit MPM time integrator** based on Newton-GMRES (Jacobian-Free Newton-Krylov, JFNK). The key improvement over the original explicit solver is large-step stability: the implicit scheme allows timesteps 3–5× larger than explicit CFL limits while preserving elastic dynamics.

**Paper**: *i-PhysGaussian: Implicit Physics-Integrated 3D Gaussians for Generative Dynamics*  
arXiv: https://arxiv.org/abs/2602.17117

## What This Repo Adds

On top of the original PhysGaussian codebase, this replication adds a paper-faithful implicit solver in `implicit_mpm_solver.py`:

### `ImplicitMPMSolver` class (new file)
A drop-in replacement for the explicit MPM loop with the following methods:

#### `_picard_eval_ul(grid_v, dt, grid_size, v_max)` — Updated-Lagrangian residual
Evaluates `F(v)` using **Updated Lagrangian**: particles are moved to `x^n + dt·v^k` before computing elastic forces, giving correct force evaluation at the deformed configuration.

#### `p2g2p_newton_gmres(step, dt)` — Full paper-faithful JFNK timestep
Implements all key algorithmic components from the paper:

| Feature | Paper Reference | Implementation |
|---------|----------------|----------------|
| Updated-Lagrangian residual | §3.2 | `_picard_eval_ul`: moves particles before stress eval |
| Newmark predictor initial guess | Eq. 14 | `v^(0) = v_explicit + dt·(1−γ)·a^n`, γ=½ |
| Central finite-difference JVP | Eq. 19 | `(I−J_F)·p ≈ [F(v+ε·p)−F(v−ε·p)]/(2ε)` |
| Mass diagonal preconditioner | Eq. 20 | `W_I = m_I/(β·dt²)`, β=¼ (Newmark constant-accel) |
| Eisenstat-Walker adaptive tolerance | EW2 | `η_k = |‖R_k‖−‖R_{k-1}‖| / ‖R_{k-2}‖`, clamped to [1e-4, 0.9] |
| Wolfe line search | Eq. 17 | Armijo (c1=1e-4) + curvature (c2=0.9), 8 halvings |
| Deformation gradient clamping | stability | `det(F) ∈ [0.1, 10]` via `clamp_F_trial_J` kernel |

### `gs_simulation.py` additions
- `--solver {picard,newton_gmres}` flag to switch between the original Picard iteration and Newton-GMRES
- `--dt_multiplier N` flag to run with N× the base timestep (tests large-step stability)

## Usage

```bash
# Standard implicit Picard (original)
python gs_simulation.py --model_path model/ficus_whitebg-trained \
    --output_path output/ficus_picard --config config/ficus_config.json \
    --implicit --output_ply

# Newton-GMRES at 1× timestep (paper-faithful)
python gs_simulation.py --model_path model/ficus_whitebg-trained \
    --output_path output/ficus_newton --config config/ficus_config.json \
    --implicit --solver newton_gmres --output_ply

# Newton-GMRES at 3× timestep (large-step stability test)
python gs_simulation.py --model_path model/ficus_whitebg-trained \
    --output_path output/ficus_newton_dt3 --config config/ficus_config.json \
    --implicit --solver newton_gmres --dt_multiplier 3 --output_ply
```

## Observed Results (Ficus Scene)

Running on ficus with E=2MPa, ν=0.4, jelly material:

| Solver | dt multiplier | GMRES iters/step | Convergence | Elastic oscillation |
|--------|--------------|-----------------|-------------|-------------------|
| Explicit | 1× | — | ✅ (CFL) | ✅ visible rebound |
| Picard implicit | 1× | 30 Picard | ❌ over-damped | ❌ no rebound |
| Newton-GMRES | 1× | ~5 Newton × 15 GMRES | ~5 Newton iters | ✅ rebound |
| Newton-GMRES | 3× | ~5 Newton × 15 GMRES | stable | ✅ rebound, 3× faster |

The Newton-GMRES solver correctly reproduces the elastic oscillation (velocity sign changes) that Picard suppresses due to frozen-Lagrangian over-damping.

## Key Implementation Notes

**Why Updated Lagrangian matters**: The Picard iteration in the original code holds particle positions fixed at `x^n`, causing elastic forces to be evaluated at the un-deformed configuration. This severely under-estimates restoring forces on large timesteps, leading to a barely-oscillating simulation. Updated Lagrangian evaluates forces at `x^n + dt·v^k` — the correct deformed state for each Newton iterate.

**GMRES restart vs maxiter**: `scipy.sparse.linalg.gmres(restart=k, maxiter=1)` = exactly k Krylov vectors per cycle. Using `maxiter=k` instead gives k full restart cycles (k×restart matvecs), which is 15× more expensive.

**Deformation gradient clamping**: At dt≥3× CFL, extreme `v^k` candidates during Newton/GMRES can push `det(F)→0`, causing neo-Hookean stress explosion. The `clamp_F_trial_J` kernel keeps `det(F) ∈ [0.1, 10]` via isotropic rescaling.

## AutoResearch Pipeline

This code was produced by the [AutoResearch](https://autoresearch.tech) autonomous research system:
- **Paper reading & planning**: ZhipuAI GLM-5 analyzed arXiv 2602.17117 and identified the key algorithmic differences from the baseline
- **Implementation**: Anthropic Claude translated the plan into Python/Warp patches, debugged convergence issues, and verified stability
- **Evaluation**: Automated COMD/RMSD metrics compare Newton-GMRES vs Picard vs explicit reference trajectories

The full conversation log and implementation details are tracked in the AutoResearch project.

## Original PhysGaussian

This work builds on [PhysGaussian](https://github.com/XPandora/PhysGaussian) (CVPR 2024). Please cite the original:

```bibtex
@article{xie2023physgaussian,
  title={PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics}, 
  author={Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
  journal={arXiv preprint arXiv:2311.12198},
  year={2023}
}
```

And the i-PhysGaussian paper when available:

```bibtex
@article{iphysgaussian2026,
  title={i-PhysGaussian: Implicit Physics-Integrated 3D Gaussians for Generative Dynamics},
  journal={arXiv preprint arXiv:2602.17117},
  year={2026}
}
```

## Disclaimer

This is an **unofficial** replication produced by an AI system. It has not been reviewed or endorsed by the original i-PhysGaussian authors. The implementation is based solely on the arXiv preprint and may differ from the authors' official release.
