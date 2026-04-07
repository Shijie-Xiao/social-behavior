"""
Visualization for MouseSRNN — following the original socialAttention style.

Reference:
  - visualize.py: trajectories with solid (GT) + dashed (pred)
  - attn_visualize.py: attention circles at target positions

Saves ONE figure per sample, clean and readable.

Usage:
    python visualize_mice.py --results ../save/mice/run01/results_val_mean.pkl
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MICE = 3
N_KPS = 4
N_NODES = 12
KP_NAMES = ["nose", "neck", "center_back", "tail_base"]
MOUSE_COLORS = ["#2ecc71", "#3498db", "#e74c3c"]


def _aggregate_mouse_attention(attn):
    """(12, 11) node attention → (3, 3) mouse attention."""
    mat = np.zeros((N_MICE, N_MICE))
    cnt = np.zeros((N_MICE, N_MICE))
    for i in range(N_NODES):
        mi = i // N_KPS
        k = 0
        for j in range(N_NODES):
            if j == i:
                continue
            mj = j // N_KPS
            mat[mi, mj] += attn[i, k]
            cnt[mi, mj] += 1
            k += 1
    mask = cnt > 0
    mat[mask] /= cnt[mask]
    return mat


def plot_one(gt, pred, obs_length, attn_list, arena_scale, save_path,
             sample_id=0):
    """
    Plot one sample following original socialAttention style.

    gt   : (T, 12, 2) normalised
    pred : (pred_len, 12, 2) normalised
    attn_list : list of (12, 11) per pred step, or None
    """
    T = gt.shape[0]
    pred_len = T - obs_length
    s = arena_scale
    cb_idx = 2  # center_back

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for m in range(N_MICE):
        c = MOUSE_COLORS[m]
        cb = m * N_KPS + cb_idx

        # True trajectory (solid + circle markers, like original)
        true_xy = gt[:, cb] * s
        ax.plot(true_xy[:obs_length, 0], true_xy[:obs_length, 1],
                color=c, ls="-", lw=2, marker="o", ms=4,
                markevery=list(range(obs_length - 1)),
                label=f"Mouse {m} obs" if m == 0 else None)
        ax.plot(true_xy[obs_length - 1:, 0], true_xy[obs_length - 1:, 1],
                color=c, ls="-", lw=2, marker="o", ms=4,
                markevery=list(range(1, pred_len + 1)))

        # Blue diamond at last observed position (like original)
        ax.scatter(true_xy[obs_length - 1, 0], true_xy[obs_length - 1, 1],
                   color="b", marker="D", s=80, zorder=5)

        # Predicted trajectory (dashed + x markers)
        p_len = min(pred_len, pred.shape[0])
        pred_xy = pred[:p_len, cb] * s
        ax.plot(pred_xy[:, 0], pred_xy[:, 1],
                color=c, ls="--", lw=2, marker="x", ms=6,
                markevery=list(range(p_len)))

    # Attention circles (like original attn_visualize.py)
    if attn_list is not None and len(attn_list) > 0:
        attn_raw = attn_list[0]  # first prediction step
        mouse_attn = _aggregate_mouse_attention(attn_raw)

        # Get center_back position at obs boundary for each mouse
        centers = {}
        for m in range(N_MICE):
            cb = m * N_KPS + cb_idx
            centers[m] = gt[obs_length - 1, cb] * s

        for m_src in range(N_MICE):
            for m_dst in range(N_MICE):
                if m_src == m_dst:
                    continue
                w = mouse_attn[m_src, m_dst]
                # radius proportional to weight (like original: weight * scale)
                radius = w * s * 0.08
                if radius < 2:
                    radius = 2
                circle = plt.Circle(
                    (centers[m_dst][0], centers[m_dst][1]),
                    radius, fill=False,
                    color="b", lw=2, alpha=0.8)
                ax.add_artist(circle)

    ax.set_aspect("equal")
    ax.set_title(f"Sequence {sample_id}", fontsize=12)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_one_with_attn(gt, pred, obs_length, attn_list, arena_scale,
                       save_path, sample_id=0):
    """
    Per-mouse attention plot: for each source mouse, show its trajectory
    and circles on other mice showing how much it attends to them.

    Saves one figure per source mouse (like original attn_visualize.py).
    """
    if attn_list is None or len(attn_list) == 0:
        return

    T = gt.shape[0]
    pred_len = T - obs_length
    s = arena_scale
    cb_idx = 2
    attn_raw = attn_list[0]
    mouse_attn = _aggregate_mouse_attention(attn_raw)

    for m_src in range(N_MICE):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        c_src = MOUSE_COLORS[m_src]

        # Plot source mouse trajectory
        cb = m_src * N_KPS + cb_idx
        true_xy = gt[:, cb] * s
        ax.plot(true_xy[:obs_length, 0], true_xy[:obs_length, 1],
                color="r", ls="-", lw=2, marker="o", ms=4,
                markevery=list(range(obs_length - 1)))
        ax.scatter(true_xy[obs_length - 1, 0], true_xy[obs_length - 1, 1],
                   color="b", marker="D", s=80, zorder=5)

        # Plot other mice with attention circles
        for m_dst in range(N_MICE):
            if m_dst == m_src:
                continue
            c_other = MOUSE_COLORS[m_dst]
            cb_o = m_dst * N_KPS + cb_idx
            true_xy_o = gt[:, cb_o] * s

            ax.plot(true_xy_o[:obs_length, 0], true_xy_o[:obs_length, 1],
                    color=c_other, ls="-", lw=1.5, marker="o", ms=3,
                    markevery=list(range(obs_length - 1)))
            ax.scatter(true_xy_o[obs_length - 1, 0],
                       true_xy_o[obs_length - 1, 1],
                       color="b", marker="D", s=60, zorder=5)

            # Attention circle
            w = mouse_attn[m_src, m_dst]
            radius = w * s * 0.08
            if radius < 2:
                radius = 2
            circle = plt.Circle(
                (true_xy_o[obs_length - 1, 0], true_xy_o[obs_length - 1, 1]),
                radius, fill=False, color="b", lw=2)
            ax.add_artist(circle)

        ax.set_aspect("equal")
        ax.set_title(f"Seq {sample_id}, Mouse {m_src} attention", fontsize=11)

        base, ext = os.path.splitext(save_path)
        path = f"{base}_mouse{m_src}{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize MouseSRNN (socialAttention style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--obs_length", type=int, default=10)
    parser.add_argument("--arena_px", type=float, default=450)
    parser.add_argument("--n_examples", type=int, default=8,
                        help="Number of samples to plot")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.results, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} result batches")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.results), "plots")
    traj_dir = os.path.join(out_dir, "trajectories")
    attn_dir = os.path.join(out_dir, "attention")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(attn_dir, exist_ok=True)

    # Collect individual samples
    samples = []
    for r in results:
        B = r["gt"].shape[0]
        for i in range(B):
            attn_i = [a[i] for a in r["attn"]] if r.get("attn") else None
            samples.append({
                "gt": r["gt"][i],
                "pred": r["pred"][i],
                "attn": attn_i,
            })
    print(f"Total samples available: {len(samples)}")

    n = min(args.n_examples, len(samples))

    # Pick diverse samples (sort by activity, sample evenly)
    activities = []
    for s in samples:
        disp = np.linalg.norm(np.diff(s["gt"], axis=0), axis=-1)
        activities.append(disp.mean())
    sorted_idx = np.argsort(activities)
    pick_idx = np.linspace(0, len(sorted_idx) - 1, n, dtype=int)

    for rank, pidx in enumerate(pick_idx):
        idx = sorted_idx[pidx]
        s = samples[idx]
        act = activities[idx]

        # Trajectory plot
        path = os.path.join(traj_dir, f"sequence{rank}.png")
        plot_one(s["gt"], s["pred"], args.obs_length, s["attn"],
                 args.arena_px, path, sample_id=rank)
        print(f"  {path}  (act={act:.4f})")

        # Per-mouse attention plot
        if s["attn"]:
            path_attn = os.path.join(attn_dir, f"sequence{rank}.png")
            plot_one_with_attn(s["gt"], s["pred"], args.obs_length,
                               s["attn"], args.arena_px,
                               path_attn, sample_id=rank)

    print(f"\nTrajectory plots: {traj_dir}/")
    print(f"Attention plots:  {attn_dir}/")
    print(f"Total: {n} samples × (1 traj + 3 attn) = {n * 4} figures")


if __name__ == "__main__":
    main()
