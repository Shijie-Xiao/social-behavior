"""
Compare trajectory predictions from two inference paths:
  A) visualize_attn.run_model()  — pre-computed GT edges (potentially leaky)
  B) model.predict()             — on-the-fly edges from predicted positions

For each sample, produces a side-by-side trajectory plot and prints ADE.
"""
import torch
import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MouseSRNN, build_edges_from_nodes
from visualize_attn import run_model  # path A

ARENA_PX = 450
N_MICE = 3
N_KPS = 4
N_NODES = 12
OBS_LEN = 10
CB_LOCAL = 2  # center_back is keypoint index 2
MC = ["#2ecc71", "#3498db", "#e74c3c"]


def predict_correct(net, gt, device):
    """Path B: use model.predict() — edges computed from predicted positions."""
    batch = torch.tensor(gt[np.newaxis], dtype=torch.float32).to(device)
    pred_pos, pred_params, attn_inter, attn_intra = net.predict(
        batch, OBS_LEN, mode="mean", n_samples=1)
    return pred_pos[0].cpu().numpy()


def predict_visualize(net, gt, device):
    """Path A: use visualize_attn.run_model() — pre-computed GT edges."""
    batch = torch.tensor(gt[np.newaxis], dtype=torch.float32).to(device)
    pred, a_inter, a_intra = run_model(net, batch, OBS_LEN)
    return pred[0] if pred is not None else None


def cb_ade(pred, gt_future, scale):
    cb_idx = [m * N_KPS + CB_LOCAL for m in range(N_MICE)]
    return np.linalg.norm(
        pred[:, cb_idx] - gt_future[:, cb_idx], axis=-1).mean() * scale


def plot_comparison(gt, pred_A, pred_B, save_path, title=""):
    s = ARENA_PX
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for ax_idx, (ax, pred, label) in enumerate(zip(
        axes,
        [pred_A, pred_B, None],
        ["A: visualize run_model (GT edges)", "B: model.predict (correct)", "Overlay"]
    )):
        ax.set_title(label, fontsize=11, fontweight="bold")
        for m in range(N_MICE):
            c = MC[m]
            cb = m * N_KPS + CB_LOCAL
            true_xy = gt[:, cb] * s

            ax.plot(true_xy[:OBS_LEN, 0], true_xy[:OBS_LEN, 1], color=c,
                    ls="-", lw=2, marker="o", ms=4,
                    markevery=list(range(OBS_LEN - 1)), zorder=3,
                    label=f"M{m} obs" if m == 0 else None)
            ax.plot(true_xy[OBS_LEN - 1:, 0], true_xy[OBS_LEN - 1:, 1],
                    color=c, ls="-", lw=1.2, marker="o", ms=2, alpha=0.4,
                    zorder=2)
            ax.scatter(true_xy[OBS_LEN - 1, 0], true_xy[OBS_LEN - 1, 1],
                       color="b", marker="D", s=70, zorder=6)

            if ax_idx == 0 and pred_A is not None:
                pred_xy = pred_A[:, cb] * s
                full = np.vstack([true_xy[OBS_LEN - 1:OBS_LEN], pred_xy])
                ax.plot(full[:, 0], full[:, 1], color=c, ls="--", lw=2,
                        marker="x", ms=5,
                        markevery=list(range(1, len(full))), zorder=4)
            elif ax_idx == 1 and pred_B is not None:
                pred_xy = pred_B[:, cb] * s
                full = np.vstack([true_xy[OBS_LEN - 1:OBS_LEN], pred_xy])
                ax.plot(full[:, 0], full[:, 1], color=c, ls="--", lw=2,
                        marker="x", ms=5,
                        markevery=list(range(1, len(full))), zorder=4)
            elif ax_idx == 2:
                if pred_A is not None:
                    pa = pred_A[:, cb] * s
                    fa = np.vstack([true_xy[OBS_LEN - 1:OBS_LEN], pa])
                    ax.plot(fa[:, 0], fa[:, 1], color=c, ls="--", lw=2,
                            marker="x", ms=5, alpha=0.5,
                            markevery=list(range(1, len(fa))), zorder=4)
                if pred_B is not None:
                    pb = pred_B[:, cb] * s
                    fb = np.vstack([true_xy[OBS_LEN - 1:OBS_LEN], pb])
                    ax.plot(fb[:, 0], fb[:, 1], color=c, ls="-.", lw=2.5,
                            marker="+", ms=7,
                            markevery=list(range(1, len(fb))), zorder=5)

        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    if pred_A is not None and pred_B is not None:
        ade_a = cb_ade(pred_A, gt[OBS_LEN:], ARENA_PX)
        ade_b = cb_ade(pred_B, gt[OBS_LEN:], ARENA_PX)
        fig.suptitle(f"{title}    cbADE  A(GT-edges)={ade_a:.1f}px  vs  B(correct)={ade_b:.1f}px",
                     fontsize=13, fontweight="bold")

    axes[2].plot([], [], "k--", lw=2, label="A: GT edges (dashed ×)")
    axes[2].plot([], [], "k-.", lw=2.5, label="B: correct (dash-dot +)")
    axes[2].legend(loc="upper left", fontsize=9)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, nargs="*", default=None)
    parser.add_argument("--auto_select", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    net = MouseSRNN(saved_args).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    raw = np.load(args.data)
    test_data = raw["test_data"].reshape(-1, 20, N_NODES, 2)
    test_activity = raw["test_activity"]
    test_chase = raw["test_chase"]
    print(f"Test set: {len(test_data)} windows")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), "plots", "compare_AB")
    os.makedirs(out_dir, exist_ok=True)

    if args.sample_idx is not None:
        samples = [(idx, "manual") for idx in args.sample_idx]
    elif args.auto_select:
        cb_idx = [m * N_KPS + CB_LOCAL for m in range(N_MICE)]
        ade_list = []
        with torch.no_grad():
            for i in range(0, len(test_data), 64):
                batch = torch.tensor(
                    test_data[i:i + 64], dtype=torch.float32).to(device)
                pred_pos, _, _, _ = net.predict(batch, OBS_LEN, mode="mean")
                gt_p = test_data[i:i + 64, OBS_LEN:]
                pred_np = pred_pos.cpu().numpy()
                for j in range(pred_np.shape[0]):
                    a = np.linalg.norm(
                        pred_np[j, :, cb_idx] - gt_p[j, :, cb_idx],
                        axis=-1).mean() * ARENA_PX
                    ade_list.append((i + j, a))
        ade_list.sort(key=lambda x: x[1])
        ade_dict = dict(ade_list)

        seen = set()
        samples = []
        for idx, _ in ade_list[:6]:
            if idx not in seen:
                seen.add(idx); samples.append((idx, "best"))
        act_sorted = np.argsort(test_activity)[::-1]
        cnt = 0
        for idx in act_sorted:
            if cnt >= 3: break
            if idx not in seen and ade_dict.get(int(idx), 999) < 15:
                seen.add(int(idx)); samples.append((int(idx), "active"))
                cnt += 1
        for idx in np.where(test_chase == 1)[0]:
            if int(idx) not in seen:
                seen.add(int(idx)); samples.append((int(idx), "chase"))
        print(f"Auto-selected {len(samples)} samples")
    else:
        idx = int(np.argsort(test_activity)[int(0.95 * len(test_activity))])
        samples = [(idx, "default")]

    print(f"\nProcessing {len(samples)} samples...")
    with torch.no_grad():
        for rank, (idx, tag) in enumerate(samples):
            gt = test_data[idx]
            pred_A = predict_visualize(net, gt, device)
            pred_B = predict_correct(net, gt, device)

            ade_a = cb_ade(pred_A, gt[OBS_LEN:], ARENA_PX) if pred_A is not None else -1
            ade_b = cb_ade(pred_B, gt[OBS_LEN:], ARENA_PX) if pred_B is not None else -1

            label = f"{rank:02d}_{tag}"
            chase = " chase" if test_chase[idx] else ""
            print(f"  [{rank:2d}] {tag:7s} idx={idx:4d}  "
                  f"ADE_A={ade_a:5.1f}px  ADE_B={ade_b:5.1f}px  "
                  f"diff={ade_b - ade_a:+.1f}px{chase}")

            save_path = os.path.join(out_dir, f"{label}_ade_A{ade_a:.0f}_B{ade_b:.0f}.png")
            plot_comparison(gt, pred_A, pred_B, save_path,
                            title=f"Sample {idx} ({tag}){chase}")

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
