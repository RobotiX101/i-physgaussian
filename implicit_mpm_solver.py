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
        self.implicit_tolerance = 5.0         # L2 convergence threshold (momentum/force units)
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

    def _p2g_momentum_only(self, dt):
        """P2G without stress — returns pure APIC momentum on grid.
        Temporarily zeros out particle_stress, runs P2G, then restores."""
        device = self.device
        grid_size = (self.mpm_model.grid_dim_x,
                     self.mpm_model.grid_dim_y,
                     self.mpm_model.grid_dim_z)
        # Save stress, zero it
        stress_buf = wp.empty_like(self.mpm_state.particle_stress)
        wp.copy(stress_buf, self.mpm_state.particle_stress)
        wp.launch(kernel=zero_stress, dim=self.n_particles,
                  inputs=[self.mpm_state.particle_stress,
                          self.mpm_state.particle_selection], device=device)
        # P2G with zero stress → pure momentum
        wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                  inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        momentum = self.mpm_state.grid_v_in.numpy().copy()
        mass = self.mpm_state.grid_m.numpy().copy()
        # Restore stress
        wp.copy(self.mpm_state.particle_stress, stress_buf)
        return momentum, mass

    def _eval_forces(self, v_grid, dt):
        """
        Given grid velocity v^{n+1}, update particles and do P2G.
        Returns (f_int, f_ext, mass) on the grid.

        f_ext_I = m_I * g
        f_int_I = (grid_v_in_full - grid_v_in_momentum) / dt
        """
        import cupy as cp
        device = self.device
        grid_size = (self.mpm_model.grid_dim_x,
                     self.mpm_model.grid_dim_y,
                     self.mpm_model.grid_dim_z)
        v_max = min(50.0, 0.4 * self.mpm_model.grid_lim / dt)

        # Convert to numpy for Warp
        if hasattr(v_grid, 'get'):  # cupy
            v_np = cp.asnumpy(cp.clip(v_grid, -v_max, v_max).astype(cp.float32))
        else:
            v_np = np.clip(v_grid, -v_max, v_max).astype(np.float32)

        # Restore to (x^n, v^n, C^n, F^n), then update x and F using v^{n+1}
        self._restore_state()
        self._update_x_F(v_np, dt, apply_bc=True)
        # Updated Lagrangian: particles now at x^{n+1} = x^n + dt*v^{n+1}

        # Clamp F_trial to prevent stress explosion
        wp.launch(kernel=clamp_F_trial_J, dim=self.n_particles,
                  inputs=[self.mpm_state.particle_F_trial,
                          self.mpm_state.particle_selection, 0.1, 10.0], device=device)

        # Compute stress from updated F
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        # Pass 1: P2G momentum only (zero stress)
        momentum, mass = self._p2g_momentum_only(dt)

        # Pass 2: P2G with stress (full)
        wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                  inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        grid_v_in_full = self.mpm_state.grid_v_in.numpy().copy()

        # Extract internal force: grid_v_in = momentum + dt * f_int
        # So f_int = (grid_v_in_full - momentum) / dt
        f_int = (grid_v_in_full - momentum) / dt

        # External force: f_ext_I = m_I * g
        g = self.mpm_model.gravitational_accelaration
        f_ext = np.zeros_like(f_int)
        for d in range(3):
            f_ext[..., d] = mass * g[d]

        return f_int, f_ext, mass

    def p2g2p_newton_gmres(self, step: int, dt: float) -> dict:
        """
        Implicit MPM via Newton-GMRES with proper momentum residual.
        Paper-faithful (arXiv 2602.17117, Eq.8-16).

        Residual: R_I(Δu) = f_I^ext + f_I^int(Δu) - m_I · a_I^{n+1}(Δu)

        Newmark-β: β=1/4, γ=1/2 (trapezoidal rule).
        """
        import cupy as cp
        from cupyx.scipy.sparse.linalg import gmres as cp_gmres
        from cupyx.scipy.sparse.linalg import LinearOperator as CpLinearOperator

        device = self.device
        grid_size = (self.mpm_model.grid_dim_x,
                     self.mpm_model.grid_dim_y,
                     self.mpm_model.grid_dim_z)

        # ---- Step 1: Pre-P2G impulses ----
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(kernel=self.pre_p2g_operations[k], dim=self.n_particles,
                      inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                      device=device)
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(kernel=self.particle_velocity_modifiers[k], dim=self.n_particles,
                      inputs=[self.time, self.mpm_state,
                              self.particle_velocity_modifier_params[k]], device=device)

        # ---- Step 2: Save state ----
        self._save_state()

        # ---- Step 3: Setup ----
        beta_nm  = 0.25
        gamma_nm = 0.5
        v_max = min(50.0, 0.4 * self.mpm_model.grid_lim / dt)

        # Get v^n and mass from explicit P2G (at x^n, F^n)
        wp.launch(kernel=mpm.zero_grid, dim=grid_size,
                  inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        wp.launch(kernel=mpm.p2g_apic_with_stress, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        v_grid_n = self._grid_velocity_from_p2g(dt)  # v^n on grid

        grid_mass_np = self.mpm_state.grid_m.numpy()
        mass_max = float(grid_mass_np.max()) if grid_mass_np.max() > 0 else 1.0
        mass_thresh = max(1e-10, 1e-2 * mass_max)

        # Previous acceleration for Newmark predictor
        if hasattr(self, '_prev_grid_a') and self._prev_grid_a is not None:
            a_n = self._prev_grid_a
        else:
            a_n = np.zeros_like(v_grid_n)

        # ---- Step 4: Initial guess (Eq.14) ----
        # Δu^(0) = Δt·v^n + ½·Δt²·a^n
        du_k = cp.asarray(dt * v_grid_n + 0.5 * dt * dt * a_n, dtype=cp.float64)

        # BCs on initial guess
        if len(self.grid_postprocess) > 0:
            v_tmp = cp.asnumpy((du_k / dt).astype(cp.float32))
            self._write_grid_v(v_tmp)
            for k_bc in range(len(self.grid_postprocess)):
                wp.launch(kernel=self.grid_postprocess[k_bc], dim=grid_size,
                          inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                                  self.collider_params[k_bc]], device=device)
            du_k = cp.asarray(self.mpm_state.grid_v_out.numpy(), dtype=cp.float64) * dt
        du_k = cp.clip(du_k, -v_max * dt, v_max * dt)

        # Move constants to GPU
        v_n_gpu = cp.asarray(v_grid_n, dtype=cp.float64)
        a_n_gpu = cp.asarray(a_n, dtype=cp.float64)
        mass_gpu = cp.asarray(grid_mass_np, dtype=cp.float64)

        # ---- Helper: momentum residual R(Δu) ----
        def eval_residual(du_cp):
            """
            R_I = f_I^ext + f_I^int(Δu) - m_I · a_I^{n+1}(Δu)

            1. Newmark: a^{n+1}, v^{n+1} from Δu
            2. _eval_forces: update particles, two-pass P2G, extract f_int
            3. R = f_ext + f_int - m*a
            """
            # Newmark relations (Eq.8-9) on GPU
            a_new = (du_cp - dt * v_n_gpu - dt*dt*(0.5 - beta_nm) * a_n_gpu) / (beta_nm * dt * dt)
            v_new = v_n_gpu + dt * ((1.0 - gamma_nm) * a_n_gpu + gamma_nm * a_new)

            # Get forces from two-pass P2G (on CPU/Warp)
            v_new_np = cp.asnumpy(cp.clip(v_new, -v_max, v_max).astype(cp.float32))
            f_int_np, f_ext_np, mass_np = self._eval_forces(v_new_np, dt)

            # Momentum residual: R = f_ext + f_int - m * a^{n+1}
            f_int_gpu = cp.asarray(f_int_np, dtype=cp.float64)
            f_ext_gpu = cp.asarray(f_ext_np, dtype=cp.float64)
            mass_local = cp.asarray(mass_np, dtype=cp.float64)

            # Expand mass to match force shape (..., 3)
            a_new_expanded = a_new  # already (N,N,N,3)
            mass_expanded = cp.expand_dims(mass_local, -1)  # (N,N,N,1)

            R = f_ext_gpu + f_int_gpu - mass_expanded * a_new_expanded
            # Zero out residual at low-mass nodes (not in free set F)
            low_mass = mass_local < mass_thresh
            R[low_mass] = 0.0

            return R, v_new, a_new

        # ---- Step 5: Newton outer loop ----
        n = du_k.size
        converged = False
        total_evals = 0
        final_residual = float('inf')

        EW_eta_max = 0.9
        EW_eta_min = 1e-4
        min_newton_iters = 5
        prev_res_norm = None
        prev_prev_res_norm = None
        eta_k = EW_eta_max

        for newton_it in range(self.newton_max_iters):
            R_k, v_new_k, a_new_k = eval_residual(du_k)
            total_evals += 1
            res_norm = float(cp.sqrt(cp.sum(R_k ** 2)))

            if step == 0:
                print(f"  [Newton {newton_it}] res={res_norm:.3e} eta={eta_k:.2e}", flush=True)

            if res_norm < self.implicit_tolerance:
                converged = True
                final_residual = res_norm
                break

            # Stagnation check
            if newton_it >= min_newton_iters and prev_res_norm is not None:
                if prev_res_norm > 1e-14:
                    contraction = res_norm / prev_res_norm
                    if contraction > 0.995:
                        if step == 0:
                            print(f"  [Newton] stagnation iter {newton_it}, c={contraction:.3f}", flush=True)
                        break
            final_residual = res_norm

            # Eisenstat-Walker
            if prev_res_norm is not None and prev_prev_res_norm is not None and prev_prev_res_norm > 1e-14:
                eta_ew = abs(res_norm - prev_res_norm) / prev_prev_res_norm
                eta_sg = 0.9 * (res_norm / prev_res_norm) ** 1.5 if prev_res_norm > 1e-14 else EW_eta_max
                eta_k = min(EW_eta_max, max(EW_eta_min, max(eta_ew, eta_sg)))
            elif prev_res_norm is not None:
                eta_k = EW_eta_max
            prev_prev_res_norm = prev_res_norm
            prev_res_norm = res_norm

            # ---- GMRES: J·δu = -R (Eq.15-16) ----
            R_k_flat = R_k.ravel()

            def matvec(p_flat, _du_k=du_k):
                nonlocal total_evals
                # Central FD JVP (Eq.19): J·p ≈ [R(Δu+εp) - R(Δu-εp)] / (2ε)
                norm_inf = float(cp.max(cp.abs(p_flat)))
                if norm_inf < 1e-14:
                    return p_flat.copy()
                eps_fd = 1e-4 / norm_inf
                p_shaped = (eps_fd * p_flat).reshape(_du_k.shape)
                R_plus, _, _ = eval_residual(_du_k + p_shaped)
                total_evals += 1
                R_minus, _, _ = eval_residual(_du_k - p_shaped)
                total_evals += 1
                return (R_plus.ravel() - R_minus.ravel()) / (2.0 * eps_fd)

            A = CpLinearOperator((n, n), matvec=matvec, dtype=cp.float64)

            # Preconditioner (Eq.20): W_I = m_I / (β·Δt²)
            _mass_flat = cp.maximum(mass_gpu.ravel(), mass_thresh)
            _mass_rep = cp.repeat(_mass_flat, 3)
            _diag_w = _mass_rep / (beta_nm * dt * dt)
            _diag_w = cp.maximum(_diag_w, 1e-8)
            M_prec = CpLinearOperator((n, n), matvec=lambda x: x / _diag_w, dtype=cp.float64)

            delta_flat, info = cp_gmres(A, -R_k_flat, M=M_prec,
                                        restart=self.gmres_max_iters,
                                        maxiter=3, atol=0, rtol=float(eta_k))
            delta_du = delta_flat.reshape(du_k.shape)

            # ---- Armijo line search (Eq.17) ----
            phi_0 = 0.5 * float(cp.sum(R_k_flat ** 2))
            dphi_0 = -2.0 * phi_0

            if dphi_0 >= 0:
                # Steepest descent fallback (paper specifies this)
                delta_du = -R_k
                dphi_0 = -2.0 * phi_0

            alpha = 1.0
            c1 = 1e-4
            for ls_it in range(10):
                du_trial = cp.clip(du_k + alpha * delta_du, -v_max * dt, v_max * dt)
                R_trial, _, _ = eval_residual(du_trial)
                total_evals += 1
                phi_trial = 0.5 * float(cp.sum(R_trial.ravel() ** 2))
                if phi_trial <= phi_0 + c1 * alpha * dphi_0:
                    break
                alpha *= 0.5

            du_k = cp.clip(du_k + alpha * delta_du, -v_max * dt, v_max * dt)

        # ---- Step 6: Final state ----
        a_final = (du_k - dt * v_n_gpu - dt*dt*(0.5 - beta_nm) * a_n_gpu) / (beta_nm * dt * dt)
        v_final = v_n_gpu + dt * ((1.0 - gamma_nm) * a_n_gpu + gamma_nm * a_final)
        v_final = cp.clip(v_final, -v_max, v_max)
        v_final_np = cp.asnumpy(v_final.astype(cp.float32))

        # Store a^{n+1} for next step's Newmark predictor
        self._prev_grid_a = cp.asnumpy(a_final)
        self._prev_particle_v = self.mpm_state.particle_v.numpy().copy()
        self._prev_dt = dt

        # Restore and final G2P with converged velocity
        self._restore_state()
        wp.launch(kernel=mpm.compute_stress_from_F_trial, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        if self.mpm_model.grid_v_damping_scale < 1.0:
            v_final_np = v_final_np * self.mpm_model.grid_v_damping_scale
        self._write_grid_v(v_final_np)
        for k_bc in range(len(self.grid_postprocess)):
            wp.launch(kernel=self.grid_postprocess[k_bc], dim=grid_size,
                      inputs=[self.time, dt, self.mpm_state, self.mpm_model,
                              self.collider_params[k_bc]], device=device)
            if self.modify_bc[k_bc] is not None:
                self.modify_bc[k_bc](self.time, dt, self.collider_params[k_bc])
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
        gv_max = float(np.max(np.abs(v_final_np))) if np.all(np.isfinite(v_final_np)) else float('inf')
        print(f"[Newton-GMRES] step={step} newton={newton_it+1} converged={converged} "
              f"res={final_residual:.3e} evals={total_evals} "
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




@wp.kernel
def zero_stress(
    particle_stress: wp.array(dtype=wp.mat33),
    particle_selection: wp.array(dtype=int),
):
    """Zero out particle stress for momentum-only P2G."""
    p = wp.tid()
    if particle_selection[p] == 0:
        particle_stress[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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

