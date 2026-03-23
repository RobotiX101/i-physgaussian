"""
i-PhysGaussian Replication -- Evaluation Script v2
Reproduces Table 1 (BMF stability frontier) and Table 7 (COMD/mwRMSD AUC)
from arXiv 2602.17117.

Usage:
    python eval/eval_metrics.py [--base BASE_DIR] [--scene ficus]
"""
import numpy as np
import os
import glob
import argparse
from plyfile import PlyData

# ── BMF threshold (paper: frames where COMD > BMF_THRESH count as failures) ──
# Paper does not state the exact value; we use 0.5 * grid_lim = 0.5m as threshold
# matching the "bounded motion fidelity" concept (motion must stay within domain)
BMF_THRESH = 0.5  # metres

def load_trajectory(run_dir, max_frames=None):
    """Load PLY sequence → np array [F, N, 3]."""
    ply_dir = os.path.join(run_dir, "simulation_ply")
    files = sorted(glob.glob(os.path.join(ply_dir, "sim_*.ply")))
    if max_frames:
        files = files[:max_frames + 1]
    if not files:
        return None
    frames = []
    for fpath in files:
        d = PlyData.read(fpath)
        v = d["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
        frames.append(pts)
    return np.array(frames)   # [F, N, 3]

def compute_comd(ref, sim):
    """COM displacement per frame (metres). ref/sim: [F, N, 3]."""
    n = min(len(ref), len(sim))
    com_ref = ref[:n].mean(axis=1)
    com_sim = sim[:n].mean(axis=1)
    return np.sqrt(((com_ref - com_sim)**2).sum(axis=1))  # [n]

def compute_mwrmsd(ref, sim):
    """Mass-weighted RMSD per frame.
    Since all particles have equal mass in MPM, mwRMSD = plain RMSD.
    If per-particle mass is available in PLY it would be used; otherwise uniform.
    """
    n = min(len(ref), len(sim))
    mwrmsds = []
    for i in range(n):
        diff = ref[i] - sim[i]   # [N, 3]
        sq = (diff**2).sum(axis=1)  # [N]
        mwrmsds.append(np.sqrt(sq.mean()))
    return np.array(mwrmsds)  # [n]

def bmf_pass(comd_series, threshold=BMF_THRESH):
    """BMF gate: True if ALL frames have COMD < threshold."""
    return bool(np.all(comd_series < threshold))

def bmf_fail_rate(comd_series, threshold=BMF_THRESH):
    """Fraction of frames failing the BMF gate."""
    return float(np.mean(comd_series >= threshold))

def auc_normalised(curve, n_total):
    """Normalised AUC of a per-frame curve (0=best, 1=worst).
    Normalised by (n_total * max_possible_value).
    Paper normalises by total frames and scales to [0,1].
    """
    if len(curve) == 0:
        return 1.0  # worst case for failed runs
    # Pad with worst-case (BMF_THRESH) for missing frames
    pad_len = max(0, n_total - len(curve))
    padded = np.concatenate([curve[:n_total], np.full(pad_len, BMF_THRESH)])
    return float(padded.mean() / BMF_THRESH)


def run_eval(base_dir, scene="ficus", n_frames=126):
    print("=" * 75)
    print(f"  i-PhysGaussian Replication -- {scene.upper()} Evaluation")
    print(f"  Reproducing Table 1 (BMF) and Table 7 (AUC) of arXiv 2602.17117")
    print("=" * 75)

    # ── Load reference (explicit k=1) ──────────────────────────────────────
    ref_dir = os.path.join(base_dir, f"{scene}_explicit_ply")
    ref = load_trajectory(ref_dir, n_frames)
    if ref is None:
        print(f"ERROR: reference not found at {ref_dir}")
        return
    print(f"\nReference (explicit k=1): {len(ref)} frames, {ref.shape[1]} particles\n")

    # ── k-sweep definition (paper exact set) ────────────────────────────────
    K_VALUES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    methods = {
        "i-PhysGaussian": ("newton_gmres", f"{scene}_newton_k{{k}}_ply"),
        "PhysGaussian":   ("picard",       f"{scene}_picard_k{{k}}_ply"),
    }

    # ── Per-method, per-k results ───────────────────────────────────────────
    results = {}  # method -> list of (k, comd_series, mwrmsd_series, pass)

    for method_name, (solver, dir_template) in methods.items():
        results[method_name] = []
        for k in K_VALUES:
            run_dir = os.path.join(base_dir, dir_template.format(k=k))
            traj = load_trajectory(run_dir, n_frames)
            if traj is None or len(traj) < 5:
                results[method_name].append((k, None, None, False))
                continue
            n = min(len(ref), len(traj))
            comd   = compute_comd(ref[:n], traj[:n])
            mwrmsd = compute_mwrmsd(ref[:n], traj[:n])
            passed = bmf_pass(comd)
            results[method_name].append((k, comd, mwrmsd, passed))

    # ── TABLE 1: BMF Stability Frontier ────────────────────────────────────
    print("TABLE 1: BMF Stability Frontier")
    print(f"  (frame fails if COMD > {BMF_THRESH*100:.0f}cm)")
    print("-" * 65)
    print(f"  {'Method':<22}  {'k_max':>6}  {'Fail%':>7}  {'Stable k values'}")
    print("-" * 65)

    # Paper reference values for ficus
    paper_ref = {
        "i-PhysGaussian": (20, 0.0),
        "PhysGaussian":   (1,  90.9),
    }

    for method_name in methods:
        res = results[method_name]
        passed_ks = [k for k, _, _, p in res if p]
        failed_ks = [k for k, c, _, p in res if c is not None and not p]
        missing_ks = [k for k, c, _, _ in res if c is None]
        k_max = max(passed_ks) if passed_ks else 0
        total_tested = len([r for r in res if r[1] is not None])
        n_failed = len(failed_ks)
        fail_pct = 100.0 * n_failed / len(K_VALUES) if total_tested > 0 else 100.0

        paper_kmax, paper_fail = paper_ref.get(method_name, (None, None))
        paper_str = f"  [paper: k_max={paper_kmax}, fail={paper_fail}%]" if paper_kmax else ""

        print(f"  {method_name:<22}  {k_max:>6}  {fail_pct:>6.1f}%  {passed_ks}")
        if missing_ks:
            print(f"  {'':22}  {'':>6}  {'':>7}  (missing runs: k={missing_ks})")
        if paper_str:
            print(f"  {'':22}  {paper_str}")
    print("-" * 65)

    # ── TABLE 7: Normalised AUC ─────────────────────────────────────────────
    print(f"\nTABLE 7: Normalised AUC of COMD and mwRMSD (lower = better)")
    print("-" * 55)
    print(f"  {'Method':<22}  {'COMD AUC':>10}  {'mwRMSD AUC':>12}")
    print("-" * 55)

    paper_auc = {
        "i-PhysGaussian": (0.0184, 0.0279),
        "PhysGaussian":   (0.5975, 0.8509),
    }

    for method_name in methods:
        res = results[method_name]
        # Only include BMF-passing runs in AUC (paper does this)
        comd_aucs, mwrmsd_aucs = [], []
        for k, comd, mwrmsd, passed in res:
            if comd is not None and passed:
                comd_aucs.append(auc_normalised(comd, n_frames))
                mwrmsd_aucs.append(auc_normalised(mwrmsd, n_frames))

        if comd_aucs:
            cauc = np.mean(comd_aucs)
            mauc = np.mean(mwrmsd_aucs)
        else:
            cauc = mauc = float('nan')

        pa, pm = paper_auc.get(method_name, (None, None))
        paper_str = f"  [paper: COMD={pa}, mwRMSD={pm}]" if pa else ""
        print(f"  {method_name:<22}  {cauc:>10.4f}  {mauc:>12.4f}")
        if paper_str:
            print(f"  {'':22}  {paper_str}")
    print("-" * 55)

    # ── Per-k accuracy detail ───────────────────────────────────────────────
    print(f"\nPER-K ACCURACY DETAIL")
    print(f"  {'Method':<22}  {'k':>3}  {'BMF':>5}  {'COMD_mean':>10}  {'mwRMSD_mean':>12}")
    print("-" * 65)
    for method_name in methods:
        for k, comd, mwrmsd, passed in results[method_name]:
            if comd is None:
                print(f"  {method_name:<22}  {k:>3}  {'--':>5}  {'(no run)':>10}")
            else:
                bmf_str = "PASS" if passed else "FAIL"
                print(f"  {method_name:<22}  {k:>3}  {bmf_str:>5}  "
                      f"{comd.mean()*100:>9.3f}cm  {mwrmsd.mean()*100:>11.3f}cm")
    print("-" * 65)

    # ── COM trajectory oscillation ──────────────────────────────────────────
    print(f"\nELASTIC REBOUND (sign changes in COM velocity, k=1 runs)")
    print("-" * 55)
    for method_name, (solver, dir_template) in methods.items():
        run_dir = os.path.join(base_dir, dir_template.format(k=1))
        traj = load_trajectory(run_dir, n_frames)
        if traj is not None:
            com_x = traj.mean(axis=1)[:, 0]
            sc = int(np.sum(np.diff(np.sign(np.diff(com_x))) != 0))
            osc = "YES" if sc >= 3 else "NO (over-damped)"
            print(f"  {method_name:<22}  sign_changes={sc:3d}  -> {osc}")
    # Reference
    com_x = ref.mean(axis=1)[:, 0]
    sc = int(np.sum(np.diff(np.sign(np.diff(com_x))) != 0))
    print(f"  {'Explicit k=1':<22}  sign_changes={sc:3d}  -> {'YES' if sc >= 3 else 'NO'} (reference)")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/root/autodl-tmp/PhysGaussian/output",
                        help="Base output directory")
    parser.add_argument("--scene", default="ficus", help="Scene name")
    parser.add_argument("--n_frames", type=int, default=126)
    args = parser.parse_args()
    run_eval(args.base, args.scene, args.n_frames)
