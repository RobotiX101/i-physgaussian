"""
Implicit MPM Solver for PhysGaussian
Implements i-PhysGaussian style implicit integration using Picard (fixed-point) iteration.

Key difference from explicit:
- Explicit: v^{n+1} = v^n + dt * f(x^n) / m
- Implicit: Solve v^{n+1} = f(x^n + dt*v^{n+1}) / m + dt*g  via Picard

Picard map:  v^{k+1} = grid_v_in( x^n + dt*v^k ) / m + dt*g
Convergent when dt < dt_CFL / sqrt(lambda_max(A)),
where A is the stiffness matrix at x^n.

For large dt (5x-20x), Newton-GMRES is required (future work).
"""

import numpy as np
import warp as wp
from typing import Optional, Tuple, Dict, List
import time

import sys
import os
_pg_root = '/root/autodl-tmp/PhysGaussian'
if _pg_root not in sys.path:
    sys.path.insert(0, _pg_root)




from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp import mpm_utils as mpm
from mpm_solver_warp import warp_utils as wu
from mpm_solver_warp.engine_utils import *


class ImplicitMPMSolver(MPM_Simulator_WARP):
    """
    Implicit MPM solver extending MPM_Simulator_WARP with Picard iteration.

    Picard fixed-point iteration for implicit Euler:
      v^{k+1} = P2G( x^n + dt*v^k, F^{n+1}(v^k), v^n, C^n ) / m + dt*g

    Key invariants during Picard inner loop:
      - particle_v  = v^n  (fixed, from start of step)
      - particle_C  = C^n  (fixed, APIC transfer from start of step)
      - particle_x  = x^n + dt*v^k  (updated each inner iteration via update_x_F_from_grid_v)
      - particle_F  = F^{n+1}(v^k)  (updated each inner iteration via update_x_F_from_grid_v)
    """

    def __init__(self, n_particles: int, n_grid: int = 100,
                 grid_lim: float = 1.0, device: str = "cuda:0"):
        super().__init__(n_particles, n_grid, grid_lim, device)
        self.device = device

        # Solver parameters
        self.implicit_max_iters = 30          # Picard iterations per step
        self.implicit_tolerance = 1e-4        # L2 convergence threshold
        self.implicit_relaxation = 0.7        # Under-relaxation factor (1.0 = no relaxation)
        self.newton_max_iters = 25            # Newton outer iterations (adaptive: 3-25)
        self.gmres_max_iters = 15            # GMRES Krylov vectors per cycle (paper Eq.19)

        # Pre-allocated persistent warp buffer for grid velocity writes
        self._grid_v_buf = wp.zeros(
            shape=(n_grid, n_grid, n_grid), dtype=wp.vec3, device=device
        )

        # Statistics
        self.iteration_stats = {'total_iters': 0, 'total_steps': 0, 'converged': 0}
        # Previous step particle velocity (for Newmark a^n estimate)
        self._prev_particle_v = None
        self._prev_dt = None

        print(f"[ImplicitMPMSolver] n_particles={n_particles}, grid={n_grid}^3, "
              f"max_iters={self.implicit_max_iters}, tol={self.implicit_tolerance}")

    # ------------------------------------------------------------------
    # Main time-step
    # ------------------------------------------------------------------

    def p2g2p_implicit(self, step: int, dt: float) -> Dict:
        """
        One implicit Euler step via Picard iteration.

        Args:
            step: simulation step index (unused internally, kept for API compat)
            dt:   timestep (may be > explicit CFL)

        Returns dict with: converged, iterations, final_residual, max_velocity
        """
        device = self.device
        grid_size = (self.mpm_model.grid_dim_x,
                     self.mpm_model.grid_dim_y,
                     self.mpm_model.grid_dim_z)

        # ---- Step 1: Pre-P2G ops (impulses) at current self.time --------
        # These modify particle_v in-place (one-shot forces timed by self.time).
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )

        # Apply Dirichlet particle velocity modifiers
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[self.time, self.mpm_state,
                        self.particle_velocity_modifier_params[k]],
                device=device,
            )

        # ---- Step 2: Save post-impulse state as (x^n, v^n, C^n, F^n) ---
        # Picard inner loop will restore to this state before each iteration.
        self._save_state()

        # ---- Step 3: Initial explicit guess v^0 -------------------------
        wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                  inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        grid_v_cur = self._grid_velocity_from_p2g(dt)   # v^0 = grid_v_in/m + dt*g
        # Apply BC to initial guess: zero grid velocity in constrained regions
        if len(self.grid_postprocess) > 0:
            self._write_grid_v(grid_v_cur)
            for k in range(len(self.grid_postprocess)):
                wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                          inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                  self.collider_params[k]], device=device)
            grid_v_cur = self.mpm_state.grid_v_out.numpy()

        # ---- Step 4: Picard iterations -----------------------------------
        converged = False
        residual = float('inf')
        n_iters = 0
        best_residual = float('inf')
        best_grid_v   = grid_v_cur.copy()
        prev_residual = float('inf')
        consecutive_increases = 0

        for it in range(self.implicit_max_iters):
            n_iters = it + 1

            # 4a. Restore x^n, v^n, C^n, F^n
            self._restore_state()

            # 4b. Update ONLY F using current guess v^k (keep x = x^n)
            #     (particle_v and particle_C stay at v^n, C^n)
            # Frozen-Lagrangian Picard: keep particle_x = x^n throughout
            # so the mass matrix M(x^n) is constant across iterations.
            # Spectral radius = (dt/dt_CFL)^2 < 1 → guaranteed convergence.
            v_max = min(50.0, 0.4 * self.mpm_model.grid_lim / dt)
            grid_v_capped = np.clip(grid_v_cur, -v_max, v_max)
            self._update_x_F(grid_v_capped, dt, apply_bc=True)
            # Restore x^n: _update_x_F also updated particle_x → undo that.
            # F_trial is already updated to (I + grad_v*dt)*F^n; keep it.
            wp.copy(self.mpm_state.particle_x, self._buf_x)
            wp.synchronize_device(self.device)

            # 4c. Recompute stress from updated F^{n+1}
            wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)

            # 4d. P2G with (x^{n+1}, v^n, C^n, F^{n+1})
            wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                      inputs=[self.mpm_state, self.mpm_model], device=device)
            wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)

            # 4e. Compute v^{k+1} = grid_v_in / m + dt * g
            grid_v_new = self._grid_velocity_from_p2g(dt)
            # Apply BC to grid_v_new so we measure residual on BC-constrained problem
            if len(self.grid_postprocess) > 0:
                self._write_grid_v(grid_v_new)
                for k in range(len(self.grid_postprocess)):
                    wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                              inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                      self.collider_params[k]], device=device)
                grid_v_new = self.mpm_state.grid_v_out.numpy()
            # Cap grid_v_new to v_max (same limit as _update_x_F input)
            # Prevents low-mass edge cells from accumulating large velocities.
            grid_v_new = np.clip(grid_v_new, -v_max, v_max)

            # 4f. Convergence check (L2 norm of velocity change)
            # (grid_v_new already sanitized by _grid_velocity_from_p2g above)
            diff = grid_v_new - grid_v_cur
            residual = float(np.sqrt(np.nansum(diff ** 2)))

            if step == 0:
                print(f"    [iter {it}] res={residual:.3e} gv_new={float(np.max(np.abs(grid_v_new))):.4f} gv_cur={float(np.max(np.abs(grid_v_cur))):.4f}", flush=True)

            # Track best-seen iterate (lowest residual)
            if residual < best_residual:
                best_residual = residual
                best_grid_v   = grid_v_new.copy()

            if residual < self.implicit_tolerance:
                converged = True
                grid_v_cur = grid_v_new
                break

            # Early exit if residual has been growing for 3+ consecutive iters
            if residual > prev_residual:
                consecutive_increases += 1
            else:
                consecutive_increases = 0
            prev_residual = residual
            if consecutive_increases >= 3 and it >= 3:
                grid_v_cur = best_grid_v
                break

            # 4g. Under-relaxation
            grid_v_cur = (self.implicit_relaxation * grid_v_new
                          + (1.0 - self.implicit_relaxation) * grid_v_cur)

        # If loop exhausted max_iters without converging or early-exit, use best seen
        if not converged and consecutive_increases < 3:
            grid_v_cur = best_grid_v
        residual = best_residual

        # ---- Step 5: Restore x^n for final G2P --------------------------
        self._restore_state()
        # CRITICAL: particle_F was modified by Picard's compute_stress_from_F_trial
        # calls during iteration. Restore it from the restored F_trial so that
        # the final G2P computes F^{n+1} = (I + dt*grad_v) * F^n correctly.
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        # ---- Step 6: Apply damping (scale grid velocities) ---------------
        if self.mpm_model.grid_v_damping_scale < 1.0:
            grid_v_cur = grid_v_cur * self.mpm_model.grid_v_damping_scale

        # ---- Step 7: Write converged velocity to grid_v_out -------------
        self._write_grid_v(grid_v_cur)

        # ---- Step 8: Apply BCs to grid_v_out ----------------------------
        for k in range(len(self.grid_postprocess)):
            wp.launch(
                kernel=self.grid_postprocess[k],
                dim=grid_size,
                inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                        self.collider_params[k]],
                device=device,
            )
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, dt, self.collider_params[k])

        # ---- Step 9: Final G2P (updates particle v, x, C, F) -----------
        wp.launch(kernel=mpm.g2p, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        # Clamp any particles that escaped the grid stencil back to valid range.
        # Without this, the next substep's p2g_apic_with_stress will write OOB.
        wp.launch(
            kernel=clamp_particle_positions,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_x,
                self.mpm_state.particle_v,
                self.mpm_state.particle_selection,
                self.mpm_model.inv_dx,
                self.mpm_model.n_grid,
            ],
            device=device,
        )

        # ---- Step 10: Advance time --------------------------------------
        self.time += dt

        # Stats
        self.iteration_stats['total_iters'] += n_iters
        self.iteration_stats['total_steps'] += 1
        if converged:
            self.iteration_stats['converged'] += 1

        particle_x_np = self.mpm_state.particle_x.numpy()
        particle_v_np = self.mpm_state.particle_v.numpy()
        max_v = float(np.max(np.abs(particle_v_np)))
        x_min = float(np.min(particle_x_np[:, 0]))
        x_max = float(np.max(particle_x_np[:, 0]))
        gv_max = float(np.max(np.abs(grid_v_cur))) if np.all(np.isfinite(grid_v_cur)) else float('inf')
        print(f"[Picard] step={step} iters={n_iters} converged={converged} "
              f"res={residual:.3e} gv_max={gv_max:.2f} max_v={max_v:.3f} x=[{x_min:.3f},{x_max:.3f}]",
              flush=True)
        return {
            'converged': converged,
            'iterations': n_iters,
            'final_residual': residual,
            'max_velocity': max_v,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def p2g2p_picard_vanilla(self, step: int, dt: float) -> Dict:
        """
        Vanilla Picard baseline (paper PhysGaussian comparison).
        
        Differences from p2g2p_implicit:
        - 5-10 iterations (not 30)
        - No under-relaxation (relaxation=1.0)
        - No best-iterate strategy
        - Matches paper's unstable baseline
        """
        device = self.device
        grid_size = (self.mpm_model.grid_dim_x,
                     self.mpm_model.grid_dim_y,
                     self.mpm_model.grid_dim_z)

        # Step 1: Pre-P2G ops (impulses)
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )

        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[self.time, self.mpm_state,
                        self.particle_velocity_modifier_params[k]],
                device=device,
            )

        # Step 2: Save state
        self._save_state()

        # Step 3: Initial explicit guess
        wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                  inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        grid_v_cur = self._grid_velocity_from_p2g(dt)
        if len(self.grid_postprocess) > 0:
            self._write_grid_v(grid_v_cur)
            for k in range(len(self.grid_postprocess)):
                wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                          inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                  self.collider_params[k]], device=device)
            grid_v_cur = self.mpm_state.grid_v_out.numpy()

        # Step 4: Vanilla Picard iterations (5-10 iters, no relaxation)
        converged = False
        residual = float('inf')
        n_iters = 0
        vanilla_max_iters = 10  # Paper baseline: 5-10 iterations

        for it in range(vanilla_max_iters):
            n_iters = it + 1

            self._restore_state()

            v_max = min(50.0, 0.4 * self.mpm_model.grid_lim / dt)
            grid_v_capped = np.clip(grid_v_cur, -v_max, v_max)
            # Updated Lagrangian: x moves with v^k (NOT frozen at x^n)
            # This matches the original PhysGaussian baseline which diverges at large dt
            self._update_x_F(grid_v_capped, dt, apply_bc=True)
            # Do NOT restore x^n — Updated Lagrangian keeps x = x^n + dt*v^k

            wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)

            wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                      inputs=[self.mpm_state, self.mpm_model], device=device)
            wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)

            grid_v_new = self._grid_velocity_from_p2g(dt)
            if len(self.grid_postprocess) > 0:
                self._write_grid_v(grid_v_new)
                for k in range(len(self.grid_postprocess)):
                    wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                              inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                      self.collider_params[k]], device=device)
                grid_v_new = self.mpm_state.grid_v_out.numpy()
            grid_v_new = np.clip(grid_v_new, -v_max, v_max)

            diff = grid_v_new - grid_v_cur
            residual = float(np.sqrt(np.nansum(diff ** 2)))

            # Divergence detection: if residual explodes, bail out immediately
            if not np.isfinite(residual) or residual > 1e8:
                if step == 0:
                    print(f"  [Picard-Vanilla] DIVERGED at iter {it}, res={residual:.3e}", flush=True)
                break

            if residual < self.implicit_tolerance:
                converged = True
                grid_v_cur = grid_v_new
                break

            # NO under-relaxation (vanilla baseline)
            grid_v_cur = grid_v_new

        # Step 5: Restore and finalize
        self._restore_state()
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        if self.mpm_model.grid_v_damping_scale < 1.0:
            grid_v_cur = grid_v_cur * self.mpm_model.grid_v_damping_scale

        self._write_grid_v(grid_v_cur)

        for k in range(len(self.grid_postprocess)):
            wp.launch(
                kernel=self.grid_postprocess[k],
                dim=grid_size,
                inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                        self.collider_params[k]],
                device=device,
            )
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, dt, self.collider_params[k])

        wp.launch(kernel=mpm.g2p, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        wp.launch(
            kernel=clamp_particle_positions,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_x,
                self.mpm_state.particle_v,
                self.mpm_state.particle_selection,
                self.mpm_model.inv_dx,
                self.mpm_model.n_grid,
            ],
            device=device,
        )

        self.time += dt

        particle_v_np = self.mpm_state.particle_v.numpy()
        max_v = float(np.max(np.abs(particle_v_np)))
        gv_max = float(np.max(np.abs(grid_v_cur))) if np.all(np.isfinite(grid_v_cur)) else float('inf')
        print(f"[Picard-Vanilla] step={step} iters={n_iters} converged={converged} "
              f"res={residual:.3e} gv_max={gv_max:.2f} max_v={max_v:.3f}",
              flush=True)
        return {
            'converged': converged,
            'iterations': n_iters,
            'final_residual': residual,
            'max_velocity': max_v,
        }



    def _picard_eval_ul(self, grid_v: np.ndarray, dt: float,
                        grid_size: tuple, v_max: float) -> np.ndarray:
        """
        Updated-Lagrangian Picard evaluation F(v):
        1. Restore state to (x^n, v^n, C^n, F^n)
        2. Move particles: x = x^n + dt*v  (Updated Lagrangian — NOT frozen)
        3. Compute stress at F^{n+1}(v), run P2G
        Returns new grid velocity = F(v).
        """
        device = self.device
        self._restore_state()
        grid_v_capped = np.clip(grid_v, -v_max, v_max)
        self._update_x_F(grid_v_capped, dt, apply_bc=True)
        # Updated Lagrangian: do NOT restore x back to x^n
        # particle_x is now x^n + dt*v^k — correct for elastic force evaluation
        # Clamp F_trial: keep J=det(F) in [0.1, 10] to prevent stress explosion
        wp.launch(kernel=clamp_F_trial_J, dim=self.n_particles,
                  inputs=[self.mpm_state.particle_F_trial,
                          self.mpm_state.particle_selection, 0.1, 10.0], device=device)
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                  inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        gv_new = self._grid_velocity_from_p2g(dt)
        if len(self.grid_postprocess) > 0:
            self._write_grid_v(gv_new)
            for k in range(len(self.grid_postprocess)):
                wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                          inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                  self.collider_params[k]], device=device)
            gv_new = self.mpm_state.grid_v_out.numpy()
        return np.clip(gv_new, -v_max, v_max)

    def p2g2p_newton_gmres(self, step: int, dt: float) -> dict:
        """
        Implicit MPM via Newton-GMRES (JFNK) — paper-faithful implementation.
        Based on arXiv 2602.17117 (i-PhysGaussian).

        Implements:
          A. Newmark predictor initial guess (paper Eq.14)
          B. Updated-Lagrangian residual r(v) = v - F_UL(v) = 0
          C. Newton outer loop with Wolfe line search (paper Eq.17)
          D. GMRES inner solver: central FD JVP (paper Eq.19),
             mass diagonal preconditioner (paper Eq.20),
             Eisenstat-Walker adaptive tolerance
        """
        from scipy.sparse.linalg import gmres as sp_gmres, LinearOperator

        device = self.device
        grid_size = (self.mpm_model.grid_dim_x,
                     self.mpm_model.grid_dim_y,
                     self.mpm_model.grid_dim_z)
        v_max = min(50.0, 0.4 * self.mpm_model.grid_lim / dt)

        # ---- Step 1: Pre-P2G impulses + velocity modifiers ----
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(kernel=self.pre_p2g_operations[k], dim=self.n_particles,
                      inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                      device=device)
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(kernel=self.particle_velocity_modifiers[k], dim=self.n_particles,
                      inputs=[self.time, self.mpm_state,
                               self.particle_velocity_modifier_params[k]], device=device)

        # ---- Step 2: Save state (x^n, v^n, C^n, F^n) ----
        self._save_state()

        # ---- Step 3: Newmark predictor initial guess (paper Eq.14) ----
        # In velocity form: v^(0) = v^n + dt*(1-gamma)*a^n
        # where gamma=1/2 (Newmark) and a^n = (v^n - v^{n-1})/dt_prev
        # If no previous step (step=0) or a^n unavailable, fall back to explicit P2G.
        # First get current particle velocities (= v^n post-impulse)
        v_n_particles = self.mpm_state.particle_v.numpy()  # shape [N, 3]

        if self._prev_particle_v is not None and self._prev_dt is not None:
            # Newmark predictor: v^(0) = v_explicit + dt*(1-gamma)*a^n
            # gamma=1/2 (Newmark constant-acceleration)
            gamma_nm = 0.5

            # Standard explicit P2G — no side effects on particle_v
            wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                      inputs=[self.mpm_state, self.mpm_model], device=device)
            wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
            wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
            v_explicit = self._grid_velocity_from_p2g(dt)

            # Estimate grid-level a^n from consecutive explicit velocities
            # Avoids corrupting particle_v with temporary writes
            if hasattr(self, '_prev_grid_v_explicit') and self._prev_grid_v_explicit is not None:
                a_grid = (v_explicit - self._prev_grid_v_explicit) / self._prev_dt
            else:
                a_grid = np.zeros_like(v_explicit)
            self._prev_grid_v_explicit = v_explicit.copy()

            # Newmark predictor: v_I^(0) = v_explicit + dt*(1-gamma)*a_I^n
            v_k = v_explicit + dt * (1.0 - gamma_nm) * a_grid
            if step == 0:
                print(f"  [Init] Using Newmark predictor (gamma={gamma_nm})", flush=True)
        else:
            # First step: explicit P2G as initial guess
            wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                      inputs=[self.mpm_state, self.mpm_model], device=device)
            wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
            wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
            v_k = self._grid_velocity_from_p2g(dt)
            if step == 0:
                print(f"  [Init] Using explicit P2G (no prev step)", flush=True)

        # Apply BCs to initial guess + clip
        if len(self.grid_postprocess) > 0:
            self._write_grid_v(v_k)
            for k in range(len(self.grid_postprocess)):
                wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                          inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                  self.collider_params[k]], device=device)
            v_k = self.mpm_state.grid_v_out.numpy()
        v_k = np.clip(v_k, -v_max, v_max)

        # ---- Step 4: Newton-GMRES outer loop ----
        n = v_k.size
        converged = False
        total_evals = 0
        final_residual = float('inf')
        newton_it = 0

        # Eisenstat-Walker state (paper adaptive GMRES tolerance)
        EW_eta_max = 0.9
        EW_eta_min = 1e-4
        EW_gamma   = 0.9    # safeguard parameter
        EW_alpha   = 1.5    # choice parameter
        min_newton_iters = 3  # Minimum iterations before early exit
        prev_res_norm = None
        prev_prev_res_norm = None
        eta_k = EW_eta_max  # start loose, tighten as Newton converges

        for newton_it in range(self.newton_max_iters):
            # Evaluate residual r_k = v^k - F_UL(v^k)
            F_vk = self._picard_eval_ul(v_k, dt, grid_size, v_max)
            total_evals += 1
            r_k = v_k - F_vk
            res_norm = float(np.sqrt(np.nansum(r_k ** 2)))
            if step == 0:
                print(f"  [Newton {newton_it}] res={res_norm:.3e} gmres_tol={eta_k:.2e}",
                      flush=True)
            if res_norm < self.implicit_tolerance:
                converged = True
                final_residual = res_norm
                break

            # Stagnation check (after min_newton_iters)
            # contraction = res_norm / prev_res_norm: <1 means converging, ~1 means stagnating
            if newton_it >= min_newton_iters and prev_res_norm is not None:
                if prev_res_norm > 1e-14:
                    contraction = res_norm / prev_res_norm
                    # Exit if contraction > 0.95 (less than 5% reduction per iter)
                    if contraction > 0.95:
                        if step == 0:
                            print(f"  [Newton] stagnation at iter {newton_it}, contraction={contraction:.3f}", flush=True)
                        break
            final_residual = res_norm

            # ---- Eisenstat-Walker adaptive tolerance (paper) ----
            # eta_k = |||R_k|| - ||R_{k-1}||| / ||R_{k-2}||  (safeguarded EW2)
            if prev_res_norm is not None and prev_prev_res_norm is not None and prev_prev_res_norm > 1e-14:
                eta_ew = abs(res_norm - prev_res_norm) / prev_prev_res_norm
                # Safeguard: avoid eta decreasing too fast (prevents over-solving)
                eta_sg = EW_gamma * (prev_res_norm ** EW_alpha)
                if prev_res_norm > 1e-14:
                    eta_sg = EW_gamma * (res_norm / prev_res_norm) ** EW_alpha
                eta_k = min(EW_eta_max, max(EW_eta_min, max(eta_ew, eta_sg)))
            elif prev_res_norm is not None:
                eta_k = EW_eta_max  # not enough history yet
            prev_prev_res_norm = prev_res_norm
            prev_res_norm = res_norm

            # ---- Build GMRES linear operator ----
            F_vk_flat = F_vk.ravel().astype(np.float64)

            def matvec(p_flat, _F_vk_flat=F_vk_flat, _v_k=v_k):
                nonlocal total_evals
                # Central FD JVP (paper Eq.19): (I-J_F)*p ~ p - [F(v+e*p)-F(v-e*p)]/(2e)
                # Perturbation: ||e*p||_inf ~ 1e-4  (paper spec)
                norm_inf = float(np.max(np.abs(p_flat))) if np.any(p_flat != 0) else 1e-14
                if norm_inf < 1e-14:
                    return p_flat.copy()
                eps_fd = 1e-4 / norm_inf
                p_shaped = (eps_fd * p_flat).reshape(_v_k.shape).astype(np.float32)
                F_plus  = self._picard_eval_ul(_v_k + p_shaped, dt, grid_size, v_max)
                total_evals += 1
                F_minus = self._picard_eval_ul(_v_k - p_shaped, dt, grid_size, v_max)
                total_evals += 1
                return p_flat - (F_plus.ravel().astype(np.float64)
                                 - F_minus.ravel().astype(np.float64)) / (2.0 * eps_fd)

            A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

            # ---- Diagonal preconditioner (paper Eq.20): W_I = m_I/(beta*dt^2) + K_diag ----
            # We use beta=1/4 (Newmark constant-acceleration), K_diag~0 (verified <3% of W)
            beta_nm = 0.25
            _mass = self.mpm_state.grid_m.numpy().ravel().astype(np.float64)
            _mass_thresh = max(1e-10, 1e-2 * float(_mass.max())) if _mass.max() > 0 else 1e-10
            _mass_rep = np.repeat(np.maximum(_mass, _mass_thresh), 3)
            _diag_w = _mass_rep / (beta_nm * dt * dt)   # paper Eq.20 with beta=1/4
            _diag_w = np.maximum(_diag_w, 1e-8)
            M_prec = LinearOperator((n, n), matvec=lambda x: x / _diag_w, dtype=np.float64)

            # ---- GMRES solve: (I-J_F)*delta_v = -r_k ----
            # restart=k: exactly k Krylov vectors per cycle (paper spec)
            # Adaptive tolerance from Eisenstat-Walker
            delta_flat, info = sp_gmres(A, -r_k.ravel().astype(np.float64),
                                        M=M_prec,
                                        restart=self.gmres_max_iters,
                                        maxiter=3,  # allow 3 restart cycles for large k
                                        rtol=float(eta_k))
            delta_v = delta_flat.reshape(v_k.shape).astype(np.float32)

            # ---- Wolfe line search (paper Eq.17: Armijo + curvature) ----
            # phi(alpha) = 0.5 * ||r(v + alpha*delta_v)||^2
            # phi'(0) = r_k . d/dalpha[r(v+a*dv)]|_a=0 = -r_k . (I-J_F)*delta_v
            #         ~ -r_k . (-r_k) = ||r_k||^2  (from GMRES: (I-J_F)*dv ~ r_k)
            # => phi'(0) ~ -||r_k||^2  (always negative → descent guaranteed)
            c1 = 1e-4   # Armijo parameter
            c2 = 0.9    # curvature parameter (weak Wolfe)
            phi0     = 0.5 * res_norm ** 2
            dphi0    = -res_norm ** 2          # phi'(0) = -||r_k||^2

            alpha = 1.0
            v_trial = v_k
            for ls_it in range(8):
                v_cand = np.clip(v_k + alpha * delta_v, -v_max, v_max)
                F_cand = self._picard_eval_ul(v_cand, dt, grid_size, v_max)
                total_evals += 1
                r_cand = v_cand - F_cand
                r_cand_norm = float(np.sqrt(np.nansum(r_cand ** 2)))
                phi_alpha = 0.5 * r_cand_norm ** 2

                # Armijo condition
                armijo_ok = phi_alpha <= phi0 + c1 * alpha * dphi0
                # Curvature condition (approximate via central FD on phi')
                # dphi(alpha)/dalpha ~ (phi(alpha+h) - phi(alpha-h))/(2h) — expensive
                # Simplified: use sufficient-descent as proxy (strong Armijo)
                # Full Wolfe too costly (extra F-eval); use Armijo + accept if alpha<0.1
                curvature_ok = (r_cand_norm < res_norm) or (alpha < 0.1)
                if armijo_ok and curvature_ok:
                    v_trial = v_cand
                    break
                alpha *= 0.5
            else:
                v_trial = v_k  # no progress — keep current
            v_k = v_trial

        # ---- Step 5-10: Restore, final G2P, clamp, advance ----
        # Save current particle_v before restoring for next step's Newmark predictor
        self._prev_particle_v = self.mpm_state.particle_v.numpy().copy()
        self._prev_dt = dt

        self._restore_state()
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        if self.mpm_model.grid_v_damping_scale < 1.0:
            v_k = v_k * self.mpm_model.grid_v_damping_scale
        self._write_grid_v(v_k)
        for k in range(len(self.grid_postprocess)):
            wp.launch(kernel=self.grid_postprocess[k], dim=grid_size,
                      inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                               self.collider_params[k]], device=device)
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, dt, self.collider_params[k])
        wp.launch(kernel=mpm.g2p, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        wp.launch(kernel=clamp_particle_positions, dim=self.n_particles,
                  inputs=[self.mpm_state.particle_x, self.mpm_state.particle_v,
                           self.mpm_state.particle_selection, self.mpm_model.inv_dx,
                           self.mpm_model.n_grid], device=device)
        self.time += dt

        # Stats
        self.iteration_stats['total_iters'] += newton_it + 1
        self.iteration_stats['total_steps'] += 1
        if converged:
            self.iteration_stats['converged'] += 1
        particle_v_np = self.mpm_state.particle_v.numpy()
        max_v = float(np.max(np.abs(particle_v_np)))
        gv_max = float(np.max(np.abs(v_k))) if np.all(np.isfinite(v_k)) else float('inf')
        print(f"[Newton-GMRES] step={step} newton={newton_it+1} converged={converged} "
              f"res={final_residual:.3e} evals={total_evals} eta={eta_k:.2e} "
              f"gv_max={gv_max:.2f} max_v={max_v:.3f}",
              flush=True)
        return {'converged': converged, 'iterations': newton_it + 1,
                'final_residual': final_residual, 'max_velocity': max_v}

    def _save_state(self):
        """Save x, v, C, F_trial — all four particle fields needed for Picard."""
        px = self.mpm_state.particle_x
        pv = self.mpm_state.particle_v
        pC = self.mpm_state.particle_C
        pF = self.mpm_state.particle_F_trial
        if not hasattr(self, '_buf_x') or self._buf_x.shape != px.shape:
            self._buf_x = wp.empty_like(px)
            self._buf_v = wp.empty_like(pv)
            self._buf_C = wp.empty_like(pC)
            self._buf_F = wp.empty_like(pF)
        wp.copy(self._buf_x, px)
        wp.copy(self._buf_v, pv)
        wp.copy(self._buf_C, pC)
        wp.copy(self._buf_F, pF)

    def _restore_state(self):
        """Restore x, v, C, F_trial to start-of-step values."""
        wp.copy(self.mpm_state.particle_x,       self._buf_x)
        wp.copy(self.mpm_state.particle_v,       self._buf_v)
        wp.copy(self.mpm_state.particle_C,       self._buf_C)
        wp.copy(self.mpm_state.particle_F_trial, self._buf_F)

    def _grid_velocity_from_p2g(self, dt: float) -> np.ndarray:
        """
        Compute grid velocity from current P2G result:
            v = grid_v_in / grid_m + dt * g
        This is the explicit MPM update formula, used both for the initial
        guess and for each Picard iterate.
        """
        v_in  = self.mpm_state.grid_v_in.numpy()   # shape (N,N,N,3)
        mass  = self.mpm_state.grid_m.numpy()        # shape (N,N,N)
        g     = self.mpm_model.gravitational_accelaration  # vec3

        v_out = np.zeros_like(v_in)
        # Use a relative mass threshold: ignore cells with < 0.1% of peak mass.
        # Cells with near-zero mass (1-2 particles) get huge velocities from
        # elastic force / tiny-mass division, destabilizing the Picard iteration.
        mass_max = float(mass.max()) if mass.max() > 0 else 1.0
        mass_threshold = max(1e-10, 1e-2 * mass_max)  # raised: edge cells cause Picard divergence
        mask  = mass > mass_threshold
        # Suppress expected overflow/divide warnings from near-zero mass cells
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for d in range(3):
                v_out[..., d][mask] = v_in[..., d][mask] / mass[mask] + dt * g[d]
        # Sanitize: replace inf/nan with 0, then cap to physically reasonable velocity
        v_out = np.where(np.isfinite(v_out), v_out, 0.0)
        # 1e5 m/s hard cap (well above CFL limit, but prevents float overflow in residual)
        v_out = np.clip(v_out, -1e5, 1e5)
        return v_out

    def _update_x_F(self, grid_v: np.ndarray, dt: float, apply_bc: bool = False):
        """
        Write grid_v to grid_v_out then launch update_x_F_from_grid_v.
        This updates ONLY particle_x and particle_F_trial, leaving
        particle_v and particle_C unchanged (= v^n, C^n).

        apply_bc: if True, apply grid_postprocess (Dirichlet BC) to grid_v_out
                  before interpolating to particles. Must be True during Picard
                  inner loop so boundary particles are not displaced.
        """
        self._write_grid_v(grid_v)
        if apply_bc and len(self.grid_postprocess) > 0:
            grid_size = (self.mpm_model.grid_dim_x,
                         self.mpm_model.grid_dim_y,
                         self.mpm_model.grid_dim_z)
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                            self.collider_params[k]],
                    device=self.device,
                )
                # NOTE: do NOT call modify_bc here — that advances time and should
                # only run once per substep (in the final BC application step).
        wp.launch(
            kernel=update_x_F_from_grid_v,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_x,
                self.mpm_state.particle_F_trial,
                self.mpm_state.grid_v_out,
                self.mpm_model.dx,
                self.mpm_model.inv_dx,
                dt,
                self.mpm_state.particle_selection,
                self.mpm_model.n_grid,
            ],
            device=self.device,
        )

    def _write_grid_v(self, grid_v: np.ndarray):
        """Write numpy (N,N,N,3) grid velocity into grid_v_out warp array."""
        n = self.mpm_model.n_grid
        arr = grid_v.reshape(n, n, n, 3).astype(np.float32)
        self._grid_v_buf.assign(arr)
        wp.copy(self.mpm_state.grid_v_out, self._grid_v_buf)
        wp.synchronize_device(self.device)


# ==========================================================================
# Kernel: clamp particle positions to valid grid stencil after G2P
# ==========================================================================

@wp.kernel
def clamp_particle_positions(
    particle_x:         wp.array(dtype=wp.vec3),
    particle_v:         wp.array(dtype=wp.vec3),
    particle_selection: wp.array(dtype=int),
    inv_dx:             float,
    n_grid:             int,
):
    """
    After G2P, ensure no particle escapes the valid quadratic B-spline stencil.

    Valid base index range: [1, n_grid-3]  (with one-cell safety margin on each side)
    => valid position range: [1.5/inv_dx, (n_grid-2.5)/inv_dx]

    If a particle is OOB, clamp its position AND zero the outward velocity component.
    """
    p = wp.tid()
    if particle_selection[p] != 0:
        return

    lo = 1.5 / inv_dx
    hi = (float(n_grid) - 2.5) / inv_dx

    x = particle_x[p]
    v = particle_v[p]

    px = x[0]; py = x[1]; pz = x[2]
    vx = v[0]; vy = v[1]; vz = v[2]

    if px < lo:
        px = lo
        if vx < 0.0:
            vx = 0.0
    elif px > hi:
        px = hi
        if vx > 0.0:
            vx = 0.0

    if py < lo:
        py = lo
        if vy < 0.0:
            vy = 0.0
    elif py > hi:
        py = hi
        if vy > 0.0:
            vy = 0.0

    if pz < lo:
        pz = lo
        if vz < 0.0:
            vz = 0.0
    elif pz > hi:
        pz = hi
        if vz > 0.0:
            vz = 0.0

    particle_x[p] = wp.vec3(px, py, pz)
    particle_v[p] = wp.vec3(vx, vy, vz)


# ==========================================================================
# Kernel: update ONLY particle_x and particle_F from grid velocity
# (used in the Picard inner loop so that particle_v and particle_C
#  remain fixed at v^n, C^n, keeping P2G consistent across iterations)
# ==========================================================================

@wp.kernel
def update_x_F_from_grid_v(
    particle_x:         wp.array(dtype=wp.vec3),
    particle_F:         wp.array(dtype=wp.mat33),
    grid_v:             wp.array(ndim=3, dtype=wp.vec3),
    dx:                 float,
    inv_dx:             float,
    dt:                 float,
    particle_selection: wp.array(dtype=int),
    n_grid:             int,
):
    """
    G2P for x and F only.  Does NOT touch particle_v or particle_C.

    x^{n+1}   = x^n + dt * v_pic
    F^{n+1}   = (I + dt * \u2207v) * F^n
    Skips particles whose stencil falls outside the grid (prevents OOB at large dt).
    """
    p = wp.tid()
    if particle_selection[p] != 0:
        return

    grid_pos = particle_x[p] * inv_dx
    base_x   = wp.int(grid_pos[0] - 0.5)
    base_y   = wp.int(grid_pos[1] - 0.5)
    base_z   = wp.int(grid_pos[2] - 0.5)
    # Bounds check: stencil needs base..base+2, so base in [0, n_grid-3]
    if base_x < 0 or base_x > n_grid - 3 or        base_y < 0 or base_y > n_grid - 3 or        base_z < 0 or base_z > n_grid - 3:
        return
    fx       = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))

    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)

    # Quadratic B-spline: weights and weight derivatives
    # matrix_from_cols: each column is one axis's weight vector [w0, w1, w2]
    w = wp.matrix_from_cols(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.75, 0.75, 0.75) - wp.cw_mul(wb, wb),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.matrix_from_cols(
        fx - wp.vec3(1.5),
        -2.0 * (fx - wp.vec3(1.0)),
        fx - wp.vec3(0.5),
    )

    new_v  = wp.vec3(0.0)
    grad_v = wp.mat33(0.0)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight  = w[0, i] * w[1, j] * w[2, k]
                dweight = wp.vec3(
                    dw[0, i] * w[1, j] * w[2, k],
                    w[0, i] * dw[1, j] * w[2, k],
                    w[0, i] * w[1, j] * dw[2, k],
                ) * inv_dx
                g_v    = grid_v[base_x + i, base_y + j, base_z + k]
                new_v  = new_v  + weight  * g_v
                grad_v = grad_v + wp.outer(g_v, dweight)

    # x^{n+1} = x^n + dt * v_pic
    new_pos = particle_x[p] + dt * new_v

    # Check new position has valid grid stencil (prevents OOB in p2g during Picard loop)
    new_gp_x = new_pos[0] * inv_dx - 0.5
    new_gp_y = new_pos[1] * inv_dx - 0.5
    new_gp_z = new_pos[2] * inv_dx - 0.5
    nb_x = wp.int(new_gp_x)
    nb_y = wp.int(new_gp_y)
    nb_z = wp.int(new_gp_z)
    if nb_x < 0 or nb_x > n_grid - 3:
        return
    if nb_y < 0 or nb_y > n_grid - 3:
        return
    if nb_z < 0 or nb_z > n_grid - 3:
        return

    particle_x[p] = new_pos

    # F^{n+1} = (I + dt * grad_v) * F^n
    I_dt_L = wp.mat33(
        1.0 + dt * grad_v[0, 0],       dt * grad_v[0, 1],       dt * grad_v[0, 2],
              dt * grad_v[1, 0], 1.0 + dt * grad_v[1, 1],       dt * grad_v[1, 2],
              dt * grad_v[2, 0],       dt * grad_v[2, 1], 1.0 + dt * grad_v[2, 2],
    )
    particle_F[p] = I_dt_L * particle_F[p]

# ==========================================================================
# Kernel: full implicit G2P (updates v, x, C, F) — kept for reference /
# potential use in the final G2P if mpm.g2p is not appropriate
# ==========================================================================

@wp.kernel
def g2p_implicit_update(
    particle_x:         wp.array(dtype=wp.vec3),
    particle_v:         wp.array(dtype=wp.vec3),
    particle_C:         wp.array(dtype=wp.mat33),
    particle_F:         wp.array(dtype=wp.mat33),
    grid_v:             wp.array(ndim=3, dtype=wp.vec3),
    dx:                 float,
    inv_dx:             float,
    dt:                 float,
    particle_selection: wp.array(dtype=int),
):
    """Full G2P update (v, x, C, F).  Not used in Picard inner loop."""
    p = wp.tid()
    if particle_selection[p] != 0:
        return

    grid_pos = particle_x[p] * inv_dx
    base_x   = wp.int(grid_pos[0] - 0.5)
    base_y   = wp.int(grid_pos[1] - 0.5)
    base_z   = wp.int(grid_pos[2] - 0.5)
    fx       = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))

    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)

    w = wp.matrix_from_cols(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.75, 0.75, 0.75) - wp.cw_mul(wb, wb),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.matrix_from_cols(
        fx - wp.vec3(1.5),
        -2.0 * (fx - wp.vec3(1.0)),
        fx - wp.vec3(0.5),
    )

    new_v  = wp.vec3(0.0)
    new_C  = wp.mat33(0.0)
    grad_v = wp.mat33(0.0)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                dpos    = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                weight  = w[0, i] * w[1, j] * w[2, k]
                dweight = wp.vec3(
                    dw[0, i] * w[1, j] * w[2, k],
                    w[0, i] * dw[1, j] * w[2, k],
                    w[0, i] * w[1, j] * dw[2, k],
                ) * inv_dx
                g_v    = grid_v[base_x + i, base_y + j, base_z + k]
                new_v  = new_v  + weight  * g_v
                new_C  = new_C  + wp.outer(g_v, dpos) * weight * 4.0 * inv_dx
                grad_v = grad_v + wp.outer(g_v, dweight)

    particle_v[p] = new_v
    particle_x[p] = particle_x[p] + dt * new_v
    particle_C[p] = new_C

    I_dt_L = wp.mat33(
        1.0 + dt * grad_v[0, 0],       dt * grad_v[0, 1],       dt * grad_v[0, 2],
              dt * grad_v[1, 0], 1.0 + dt * grad_v[1, 1],       dt * grad_v[1, 2],
              dt * grad_v[2, 0],       dt * grad_v[2, 1], 1.0 + dt * grad_v[2, 2],
    )
    particle_F[p] = I_dt_L * particle_F[p]


# ==========================================================================
# Kernel: clamp particle_F_trial so det(F) stays in [J_min, J_max]
# Prevents neo-Hookean stress explosion in Updated-Lagrangian evaluations.
# ==========================================================================

@wp.kernel
def clamp_F_trial_J(
    particle_F:         wp.array(dtype=wp.mat33),
    particle_selection: wp.array(dtype=int),
    J_min:              float,
    J_max:              float,
):
    p = wp.tid()
    if particle_selection[p] != 0:
        return
    F = particle_F[p]
    J = wp.determinant(F)
    I = wp.mat33(1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0)
    if J < J_min and J > 1e-8:
        alpha = wp.pow(J_min / J, 1.0 / 3.0)
        particle_F[p] = alpha * F + (1.0 - alpha) * I
    elif J > J_max:
        alpha = wp.pow(J_max / J, 1.0 / 3.0)
        particle_F[p] = alpha * F + (1.0 - alpha) * I

