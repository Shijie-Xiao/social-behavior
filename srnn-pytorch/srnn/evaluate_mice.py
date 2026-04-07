"""
Evaluation pipeline for MouseSRNN on mouse trajectory prediction.

Implements the full evaluation workflow from the socialAttention paper,
adapted for our 12-node mouse graph:
  1. Load trained model
  2. Autoregressive prediction on test set
  3. Compute metrics: ADE, FDE, NLL, body-structure error, baselines
  4. Save results for visualization

Usage:
    python evaluate_mice.py --checkpoint ../save/mice/test_debug/best_model.tar
    python evaluate_mice.py --checkpoint ../save/mice/test_debug/best_model.tar --n_samples 20
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from model_mice import MouseSRNN, build_edges_from_nodes, N_NODES
from mouse_dataset import get_mouse_dataloaders
from criterion_mice import gaussian_2d_nll

_HAS_WANDB = True
try:
    import wandb
except ImportError:
    _HAS_WANDB = False

N_MICE = 3
N_KPS = 4
KP_NAMES = ["nose", "neck", "center_back", "tail_base"]
MOUSE_SKELETON = [(0, 1), (1, 2), (2, 3)]  # nose-neck-back-tail


# ─── Metrics ─────────────────────────────────────────────────────────

def ade(pred, gt):
    """
    Average Displacement Error.
    pred, gt: (B, pred_len, 12, 2)
    Returns: (B,) — mean L2 over pred frames and nodes
    """
    return torch.norm(pred - gt, dim=-1).mean(dim=(1, 2))


def fde(pred, gt):
    """
    Final Displacement Error.
    Returns: (B,) — mean L2 at last pred frame over nodes
    """
    return torch.norm(pred[:, -1] - gt[:, -1], dim=-1).mean(dim=1)


def ade_per_mouse(pred, gt):
    """
    ADE broken down by mouse.
    Returns: (B, 3)
    """
    B, T, N, _ = pred.shape
    err = torch.norm(pred - gt, dim=-1)  # (B, T, 12)
    err_3d = err.reshape(B, T, N_MICE, N_KPS)
    return err_3d.mean(dim=(1, 3))  # (B, 3)


def body_structure_error(pred, gt):
    """
    Mean absolute error in intra-mouse pairwise keypoint distances.
    Measures whether the model preserves body geometry.
    Returns: (B,)
    """
    B, T, N, _ = pred.shape
    pred_m = pred.reshape(B, T, N_MICE, N_KPS, 2)
    gt_m = gt.reshape(B, T, N_MICE, N_KPS, 2)

    total_err = torch.zeros(B, device=pred.device)
    count = 0
    for i in range(N_KPS):
        for j in range(i + 1, N_KPS):
            d_pred = torch.norm(pred_m[:, :, :, i] - pred_m[:, :, :, j], dim=-1)
            d_gt = torch.norm(gt_m[:, :, :, i] - gt_m[:, :, :, j], dim=-1)
            total_err += (d_pred - d_gt).abs().mean(dim=(1, 2))
            count += 1
    return total_err / count


def inter_mouse_distance_error(pred, gt):
    """
    Error in distances between mouse centers (center_back keypoint).
    Returns: (B,)
    """
    cb_idx = KP_NAMES.index("center_back")
    B, T, N, _ = pred.shape
    pred_m = pred.reshape(B, T, N_MICE, N_KPS, 2)
    gt_m = gt.reshape(B, T, N_MICE, N_KPS, 2)

    pred_centers = pred_m[:, :, :, cb_idx]  # (B, T, 3, 2)
    gt_centers = gt_m[:, :, :, cb_idx]

    total_err = torch.zeros(B, device=pred.device)
    count = 0
    for i in range(N_MICE):
        for j in range(i + 1, N_MICE):
            d_pred = torch.norm(pred_centers[:, :, i] - pred_centers[:, :, j], dim=-1)
            d_gt = torch.norm(gt_centers[:, :, i] - gt_centers[:, :, j], dim=-1)
            total_err += (d_pred - d_gt).abs().mean(dim=1)
            count += 1
    return total_err / count


# ─── Baselines ───────────────────────────────────────────────────────

def baseline_static(nodes, obs_length):
    """Repeat last observed position for all pred frames."""
    last_obs = nodes[:, obs_length - 1: obs_length]  # (B, 1, 12, 2)
    pred_len = nodes.shape[1] - obs_length
    return last_obs.expand(-1, pred_len, -1, -1)


def baseline_linear(nodes, obs_length):
    """Linear extrapolation from last two observed positions."""
    pos_t1 = nodes[:, obs_length - 2]  # (B, 12, 2)
    pos_t0 = nodes[:, obs_length - 1]  # (B, 12, 2)
    vel = pos_t0 - pos_t1
    pred_len = nodes.shape[1] - obs_length
    preds = []
    for step in range(1, pred_len + 1):
        preds.append(pos_t0 + vel * step)
    return torch.stack(preds, dim=1)


def baseline_constant_velocity(nodes, obs_length, lookback=3):
    """Extrapolate using average velocity over last `lookback` frames."""
    start = max(0, obs_length - lookback)
    vel = (nodes[:, obs_length - 1] - nodes[:, start]) / (obs_length - 1 - start)
    pred_len = nodes.shape[1] - obs_length
    preds = []
    pos = nodes[:, obs_length - 1]
    for _ in range(pred_len):
        pos = pos + vel
        preds.append(pos)
    return torch.stack(preds, dim=1)


# ─── Main evaluation ────────────────────────────────────────────────

@torch.no_grad()
def evaluate(net, loader, obs_length, pred_length, device, mode="mean",
             n_samples=1, max_batches=None):
    """
    Run full evaluation on a DataLoader.

    Returns
    -------
    metrics : dict of aggregated scalar metrics
    results : list of per-batch result dicts (for visualization)
    """
    net.eval()
    all_metrics = {
        "ade": [], "fde": [], "nll": [],
        "body_err": [], "inter_dist_err": [],
        "ade_m0": [], "ade_m1": [], "ade_m2": [],
        "bl_static_ade": [], "bl_static_fde": [],
        "bl_linear_ade": [], "bl_linear_fde": [],
        "bl_cv_ade": [], "bl_cv_fde": [],
    }
    results = []

    for batch_idx, (nodes_b, activity_b, lights_b, chase_b) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        nodes_b = nodes_b.to(device)
        gt_pred = nodes_b[:, obs_length:]  # (B, pred_len, 12, 2)

        # Model prediction
        pred_pos, pred_params, attn_list = net.predict(
            nodes_b, obs_length, mode=mode, n_samples=n_samples
        )

        if mode == "sample" and n_samples > 1:
            # Best-of-N: pick sample with lowest ADE per instance
            ade_per_sample = torch.norm(
                pred_pos - gt_pred.unsqueeze(0), dim=-1
            ).mean(dim=(2, 3))  # (S, B)
            best_idx = ade_per_sample.argmin(dim=0)  # (B,)
            B = nodes_b.size(0)
            pred_best = pred_pos[best_idx, torch.arange(B)]
        else:
            pred_best = pred_pos

        # Metrics
        all_metrics["ade"].append(ade(pred_best, gt_pred))
        all_metrics["fde"].append(fde(pred_best, gt_pred))
        all_metrics["body_err"].append(body_structure_error(pred_best, gt_pred))
        all_metrics["inter_dist_err"].append(
            inter_mouse_distance_error(pred_best, gt_pred))

        # Per-mouse ADE
        ade_m = ade_per_mouse(pred_best, gt_pred)  # (B, 3)
        all_metrics["ade_m0"].append(ade_m[:, 0])
        all_metrics["ade_m1"].append(ade_m[:, 1])
        all_metrics["ade_m2"].append(ade_m[:, 2])

        # NLL (teacher-forcing)
        e_t, e_s = build_edges_from_nodes(nodes_b)
        outputs_tf = net(nodes_b, e_t, e_s)
        nll = gaussian_2d_nll(outputs_tf, nodes_b, pred_length)
        all_metrics["nll"].append(nll)

        # Baselines
        bl_s = baseline_static(nodes_b, obs_length)
        bl_l = baseline_linear(nodes_b, obs_length)
        bl_cv = baseline_constant_velocity(nodes_b, obs_length)

        all_metrics["bl_static_ade"].append(ade(bl_s, gt_pred))
        all_metrics["bl_static_fde"].append(fde(bl_s, gt_pred))
        all_metrics["bl_linear_ade"].append(ade(bl_l, gt_pred))
        all_metrics["bl_linear_fde"].append(fde(bl_l, gt_pred))
        all_metrics["bl_cv_ade"].append(ade(bl_cv, gt_pred))
        all_metrics["bl_cv_fde"].append(fde(bl_cv, gt_pred))

        # Store a few examples for visualization
        if batch_idx < 5:
            attn_obs = [a.cpu().numpy() for a in attn_list[obs_length - 1:]]
            results.append({
                "gt": nodes_b.cpu().numpy(),
                "pred": pred_best.cpu().numpy(),
                "pred_params": pred_params.cpu().numpy(),
                "activity": activity_b.numpy(),
                "lights": lights_b.numpy(),
                "chase": chase_b.numpy(),
                "attn": attn_obs,
                "bl_static": bl_s.cpu().numpy(),
                "bl_linear": bl_l.cpu().numpy(),
            })

    # Aggregate
    metrics = {}
    for k, v in all_metrics.items():
        vals = torch.cat(v)
        metrics[k] = float(vals.mean())
        metrics[f"{k}_std"] = float(vals.std())

    return metrics, results


def print_metrics(metrics, arena_px=450):
    """Pretty-print evaluation metrics with physical units."""
    scale = arena_px  # to convert normalised coords → pixels
    cm_per_px = 50.0 / arena_px

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")

    print(f"\n  ▸ Trajectory Metrics (normalised / pixels / cm):")
    for name, key in [("ADE", "ade"), ("FDE", "fde")]:
        v = metrics[key]
        print(f"    {name:>4s}: {v:.6f}  /  {v*scale:.2f} px  /  {v*scale*cm_per_px:.2f} cm")

    print(f"\n  ▸ NLL (teacher forcing): {metrics['nll']:.4f}")

    print(f"\n  ▸ Structure Preservation:")
    print(f"    Body structure error:     {metrics['body_err']:.6f}  "
          f"({metrics['body_err']*scale:.2f} px)")
    print(f"    Inter-mouse dist error:   {metrics['inter_dist_err']:.6f}  "
          f"({metrics['inter_dist_err']*scale:.2f} px)")

    print(f"\n  ▸ Per-Mouse ADE:")
    for i in range(3):
        v = metrics[f"ade_m{i}"]
        print(f"    Mouse {i}: {v:.6f}  ({v*scale:.2f} px)")

    print(f"\n  ▸ Baseline Comparison:")
    header = f"    {'Method':>18s}  {'ADE':>10s}  {'FDE':>10s}  {'ADE(px)':>10s}  {'FDE(px)':>10s}"
    print(header)
    print("    " + "-" * 64)
    for name, prefix in [
        ("MouseSRNN", ""),
        ("Static", "bl_static_"),
        ("Linear", "bl_linear_"),
        ("Const Velocity", "bl_cv_"),
    ]:
        a_key = f"{prefix}ade" if prefix else "ade"
        f_key = f"{prefix}fde" if prefix else "fde"
        a = metrics[a_key]
        f = metrics[f_key]
        marker = " ← ours" if prefix == "" else ""
        print(f"    {name:>18s}  {a:>10.6f}  {f:>10.6f}  "
              f"{a*scale:>10.2f}  {f*scale:>10.2f}{marker}")

    # Improvement over best baseline
    bl_ades = [metrics["bl_static_ade"], metrics["bl_linear_ade"], metrics["bl_cv_ade"]]
    best_bl_ade = min(bl_ades)
    our_ade = metrics["ade"]
    if best_bl_ade > 0:
        improvement = (1 - our_ade / best_bl_ade) * 100
        print(f"\n  ▸ ADE improvement over best baseline: {improvement:+.1f}%")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MouseSRNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.tar)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz data (auto-detected from checkpoint if None)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--mode", type=str, default="mean",
                        choices=["mean", "sample"],
                        help="'mean' for deterministic, 'sample' for stochastic")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of stochastic samples (only for mode=sample)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--arena_px", type=float, default=450,
                        help="Arena diameter in pixels (for physical unit conversion)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit number of batches (for quick testing)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir (default: same as checkpoint dir)")

    # Wandb
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="MS_mice")
    parser.add_argument("--wandb_entity", type=str,
                        default="sxiao73-georgia-institute-of-technology")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Resume a wandb run by ID to attach eval results")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    obs_length = saved_args.obs_length
    pred_length = saved_args.pred_length

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    net = MouseSRNN(saved_args).to(device)
    net.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']['total']:.4f})")

    # Data
    data_path = args.data or saved_args.data
    print(f"Data: {data_path}")
    loaders = get_mouse_dataloaders(
        data_path, obs_length=obs_length,
        batch_size=args.batch_size, num_workers=0,
    )
    loader = loaders[args.split]
    print(f"Evaluating on {args.split}: {len(loader.dataset)} windows")

    # Wandb init
    use_wandb = _HAS_WANDB and not args.no_wandb
    if use_wandb:
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="eval",
            config={**vars(saved_args), "eval_split": args.split,
                    "eval_mode": args.mode},
            tags=["eval", args.split, args.mode],
        )
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        else:
            exp_tag = getattr(saved_args, "exp_tag", "eval")
            init_kwargs["name"] = f"eval_{exp_tag}_{args.split}_{args.mode}"
        wandb.init(**init_kwargs)
        print(f"wandb: {wandb.run.url}")

    # Evaluate
    print(f"\nRunning evaluation (mode={args.mode})...")
    t0 = time.time()
    metrics, results = evaluate(
        net, loader, obs_length, pred_length, device,
        mode=args.mode,
        n_samples=args.n_samples if args.mode == "sample" else 1,
        max_batches=args.max_batches,
    )
    elapsed = time.time() - t0
    print(f"Evaluation completed in {elapsed:.1f}s")

    # Print
    print_metrics(metrics, arena_px=args.arena_px)

    # Save locally
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, f"metrics_{args.split}_{args.mode}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    results_path = os.path.join(out_dir, f"results_{args.split}_{args.mode}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to: {results_path}")

    # Wandb: log metrics + upload results artifact
    if use_wandb:
        scale = args.arena_px
        wandb.summary.update({
            f"eval/{args.split}_ade_px": metrics["ade"] * scale,
            f"eval/{args.split}_fde_px": metrics["fde"] * scale,
            f"eval/{args.split}_nll": metrics["nll"],
            f"eval/{args.split}_body_err_px": metrics["body_err"] * scale,
            f"eval/{args.split}_bl_static_ade_px": metrics["bl_static_ade"] * scale,
            f"eval/{args.split}_bl_linear_ade_px": metrics["bl_linear_ade"] * scale,
        })

        # Build a comparison table
        table = wandb.Table(columns=["Method", "ADE (px)", "FDE (px)"])
        for name, prefix in [("MouseSRNN", ""), ("Static", "bl_static_"),
                              ("Linear", "bl_linear_"), ("ConstVel", "bl_cv_")]:
            a = metrics[f"{prefix}ade"] * scale
            f_ = metrics[f"{prefix}fde"] * scale
            table.add_data(name, round(a, 2), round(f_, 2))
        wandb.log({"eval/baseline_comparison": table})

        art = wandb.Artifact(
            f"eval-{args.split}-{args.mode}", type="eval_results",
            metadata=metrics,
        )
        art.add_file(metrics_path)
        wandb.log_artifact(art)

        wandb.finish()
        print("wandb evaluation run finished")


if __name__ == "__main__":
    main()
