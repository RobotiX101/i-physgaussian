"""
Quick test of Implicit MPM Solver
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'mpm_solver_warp')

import numpy as np
import warp as wp
wp.init()

from implicit_mpm_solver import ImplicitMPMSolver

# 创建求解器
n_particles = 500
n_grid = 32
grid_lim = 1.0
dx = grid_lim / n_grid  # 网格间距

solver = ImplicitMPMSolver(n_particles=n_particles, n_grid=n_grid, grid_lim=grid_lim)

# 初始化粒子（立方体）
np.random.seed(42)
particle_x_np = np.zeros((n_particles, 3), dtype=np.float32)
particle_x_np[:, 0] = np.random.uniform(0.3, 0.7, n_particles)
particle_x_np[:, 1] = np.random.uniform(0.3, 0.7, n_particles)
particle_x_np[:, 2] = np.random.uniform(0.3, 0.7, n_particles)

particle_v_np = np.zeros((n_particles, 3), dtype=np.float32)
particle_v_np[:, 0] = 0.1  # 初始 x 方向速度

# 估计每个粒子的体积（假设均匀分布）
# 总体积 ≈ 0.4^3 = 0.064，每个粒子体积 ≈ 0.064 / n_particles
domain_volume = 0.4 ** 3
particle_vol = domain_volume / n_particles
particle_vol_arr = np.full(n_particles, particle_vol, dtype=np.float32)

# 设置 Warp 数组
solver.mpm_state.particle_x = wp.from_numpy(particle_x_np, dtype=wp.vec3, device="cuda:0")
solver.mpm_state.particle_v = wp.from_numpy(particle_v_np, dtype=wp.vec3, device="cuda:0")
solver.mpm_state.particle_vol = wp.from_numpy(particle_vol_arr, dtype=float, device="cuda:0")

# 设置材料参数
E_val = 1e5  # 中等刚度
nu_val = 0.4
solver.mpm_model.material = 0  # jelly (弹性)

E_arr = np.full(n_particles, E_val, dtype=np.float32)
nu_arr = np.full(n_particles, nu_val, dtype=np.float32)
solver.mpm_model.E = wp.from_numpy(E_arr, dtype=float, device="cuda:0")
solver.mpm_model.nu = wp.from_numpy(nu_arr, dtype=float, device="cuda:0")

# 设置重力
solver.mpm_model.gravitational_accelaration = wp.vec3(0.0, -9.8, 0.0)

# finalize_mu_lam 会计算 mu 和 lam
solver.finalize_mu_lam()

# 添加边界盒（地面在 y=0）
solver.add_bounding_box()

# 计算密度和质量
density = 1000.0  # kg/m^3
total_mass = density * domain_volume
particle_mass = total_mass / n_particles
print(f"Setup:")
print(f"  Particles: {n_particles}")
print(f"  Grid: {n_grid}^3, dx = {dx:.4f}")
print(f"  E = {E_val:.0e}, nu = {nu_val}")
print(f"  Particle volume = {particle_vol:.2e}")
print(f"  Particle mass = {particle_mass:.2e}")
print(f"  Initial max velocity: {np.max(np.abs(particle_v_np)):.4f}")
print()

# 测试显式方法（小时间步）
print("="*60)
print("Test 1: Explicit (dt=1e-4, 5 steps)")
print("="*60)
dt_explicit = 1e-4
max_v_explicit = []
for step in range(5):
    solver.p2g2p(step, dt_explicit)
    v_np = solver.export_particle_v_to_torch().cpu().numpy()
    max_v = np.max(np.linalg.norm(v_np, axis=1))
    max_v_explicit.append(max_v)
    x_np = solver.export_particle_x_to_torch().cpu().numpy()
    y_min = np.min(x_np[:, 1])
    print(f"  Step {step}: max_v = {max_v:.6f}, y_min = {y_min:.4f}")

print()

# 重置粒子
solver.mpm_state.particle_x = wp.from_numpy(particle_x_np, dtype=wp.vec3, device="cuda:0")
solver.mpm_state.particle_v = wp.from_numpy(particle_v_np, dtype=wp.vec3, device="cuda:0")
solver.finalize_mu_lam()

# 测试隐式方法（10x 大时间步）
print("="*60)
print("Test 2: Implicit (dt=1e-3, 10x larger, 5 steps)")
print("="*60)
dt_implicit = 1e-3
max_v_implicit = []
for step in range(5):
    try:
        solver.p2g2p_implicit(step, dt_implicit)
        v_np = solver.export_particle_v_to_torch().cpu().numpy()
        max_v = np.max(np.linalg.norm(v_np, axis=1))
        max_v_implicit.append(max_v)
        x_np = solver.export_particle_x_to_torch().cpu().numpy()
        y_min = np.min(x_np[:, 1])
        converged = solver.convergence_history[-1] if solver.convergence_history else False
        print(f"  Step {step}: max_v = {max_v:.6f}, y_min = {y_min:.4f}, converged = {converged}")
    except Exception as e:
        print(f"  Step {step}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        break

print()
print("="*60)
print("Summary")
print("="*60)
print(f"Explicit max_v (dt=1e-4): {max(max_v_explicit):.6f}")
if max_v_implicit:
    print(f"Implicit max_v (dt=1e-3): {max(max_v_implicit):.6f}")
    if max(max_v_implicit) < 10 * max(max_v_explicit):
        print("✅ Implicit solver is STABLE at 10x dt!")
    else:
        print("⚠️  Implicit solver may need tuning")
else:
    print("Implicit test FAILED")
