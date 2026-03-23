import numpy as np
import os
import glob
from plyfile import PlyData

BASE = "/root/autodl-tmp/PhysGaussian/output"

def load_trajectory(run_dir, max_frames=None):
    ply_dir = os.path.join(run_dir, "simulation_ply")
    files = sorted(glob.glob(os.path.join(ply_dir, "sim_*.ply")))
    if max_frames:
        files = files[:max_frames + 1]
    if not files:
        return None
    frames = []
    for f in files:
        d = PlyData.read(f)
        v = d["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
        frames.append(pts)
    return np.array(frames)

def compute_com_trajectory(traj):
    return traj.mean(axis=1)

def compute_comd_per_frame(ref, sim):
    n = min(len(ref), len(sim))
    com_ref = ref[:n].mean(axis=1)
    com_sim = sim[:n].mean(axis=1)
    return np.sqrt(((com_ref - com_sim)**2).sum(axis=1))

def compute_rmsd_per_frame(ref, sim):
    n = min(len(ref), len(sim))
    rmsds = []
    for i in range(n):
        diff = ref[i] - sim[i]
        rmsds.append(np.sqrt((diff**2).sum(axis=1).mean()))
    return np.array(rmsds)

print("=" * 75)
print("i-PhysGaussian Replication -- Complete Evaluation")
print("Ficus: E=2MPa, jelly, impulse=[-0.18,0,0], 125 frames")
print("=" * 75)

runs = {
    "explicit_dt1":  ("Explicit",     "1x",  "ficus_explicit_ply"),
    "picard_dt1":    ("Picard",       "1x",  "ficus_dt1_ply"),
    "picard_dt3":    ("Picard",       "3x",  "ficus_dt3_ply"),
    "picard_dt5":    ("Picard",       "5x",  "ficus_dt5_ply"),
    "newton_dt3":    ("Newton-GMRES", "3x",  "ficus_newton_dt3_v5_ply"),
}

print("\nLoading trajectories...")
trajs = {}
for key, (method, dt, dirname) in runs.items():
    path = os.path.join(BASE, dirname)
    if not os.path.exists(path):
        print("  %s: NOT FOUND" % key)
        continue
    t = load_trajectory(path)
    if t is None:
        print("  %s: empty" % key)
        continue
    trajs[key] = t
    print("  %-20s %3d frames, %d particles" % (key+":", len(t), t.shape[1]))

print()
print("STABILITY TABLE")
print("-" * 55)
stability_rows = [
    ("Explicit",     "1x",   "explicit_dt1",  True),
    ("Picard",       "1x",   "picard_dt1",    True),
    ("Picard",       "3x",   "picard_dt3",    True),
    ("Picard",       "5x",   "picard_dt5",    True),
    ("Explicit",     "3x",   None,            False),
    ("Explicit",     "5x",   None,            False),
    ("Newton-GMRES", "3x",   "newton_dt3",    None),
]
for (method, dt, key, stable) in stability_rows:
    label = "%s dt=%s" % (method, dt)
    if key and key in trajs:
        n = len(trajs[key])
        status = "STABLE (%d/126)" % n if n == 126 else "PARTIAL (%d/126)" % n
    elif stable is False:
        status = "CRASH (CUDA error 700)"
    elif stable is None:
        status = "RUNNING (overnight)..."
    else:
        status = "not loaded"
    print("  %-22s  %s" % (label, status))
print("-" * 55)

ref = trajs.get("explicit_dt1")
if ref is None:
    print("\nNo explicit reference.")
else:
    print()
    print("ACCURACY vs EXPLICIT dt=1x (units = cm)")
    print("-" * 75)
    print("  %-22s %4s %6s %10s %9s %10s %9s" % (
        "Method", "dt", "N", "COMD_mean", "COMD_max", "RMSD_mean", "RMSD_max"))
    print("-" * 75)
    for key, (method, dt, dirname) in runs.items():
        if key == "explicit_dt1" or key not in trajs:
            continue
        sim = trajs[key]
        n = min(len(ref), len(sim))
        comd = compute_comd_per_frame(ref[:n], sim[:n])
        rmsd  = compute_rmsd_per_frame(ref[:n], sim[:n])
        label = "%s dt=%s" % (method, dt)
        print("  %-22s %4s %6d %9.3fcm %8.3fcm %9.3fcm %8.3fcm" % (
            label, dt, n,
            comd.mean()*100, comd.max()*100,
            rmsd.mean()*100,  rmsd.max()*100))
    print("-" * 75)

    print()
    print("COM TRAJECTORY (x-axis, direction of impulse)")
    print("-" * 65)
    ref_com = compute_com_trajectory(ref)
    ref_com_x = ref_com[:, 0]
    x0 = ref_com_x[0]
    print("  %-28s x0=%.4f  max_disp=%+.2fcm  final=%+.2fcm" % (
        "Explicit dt=1x", x0,
        (ref_com_x - x0).max()*100,
        (ref_com_x[-1] - x0)*100))
    for key, (method, dt, dirname) in runs.items():
        if key == "explicit_dt1" or key not in trajs:
            continue
        sim = trajs[key]
        com_x = compute_com_trajectory(sim)[:, 0]
        s0 = com_x[0]
        label = "%s dt=%s" % (method, dt)
        print("  %-28s x0=%.4f  max_disp=%+.2fcm  final=%+.2fcm" % (
            label, s0,
            (com_x - s0).max()*100,
            (com_x[-1] - s0)*100))

    print()
    print("OSCILLATION ANALYSIS (elastic rebound -- sign changes in COM velocity)")
    print("-" * 65)
    for key, (method, dt, dirname) in runs.items():
        if key not in trajs:
            continue
        sim = trajs[key]
        com_x = compute_com_trajectory(sim)[:, 0]
        dcom = np.diff(com_x)
        sc = int(np.sum(np.diff(np.sign(dcom)) != 0))
        label = "Explicit dt=1x" if key == "explicit_dt1" else "%s dt=%s" % (method, dt)
        osc = "YES -- elastic rebound" if sc >= 3 else "NO  -- over-damped"
        print("  %-28s sign_changes=%3d  -> %s" % (label, sc, osc))

print()
print("=" * 75)
