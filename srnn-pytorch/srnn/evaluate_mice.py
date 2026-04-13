"""
Evaluation pipeline for MouseSRNN on mouse trajectory prediction.

m'nmnSupports variable n_keypoints (1/2/3/4) and graph_type (full/inter).
All cross-config comparisons use the center_back keypoint (3 nodes).

Usage:
    python evaluate_mice.py --checkpoint ../save/mice/exp/best_model.tar
    python evaluate_mice.py --checkpoint ../save/mice/exp/best_model.tar --n_samples 20
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from model_mice import MouseSRNN, build_edges_from_nodes
from mouse_dataset import (
    get_mouse_dataloaders, get_center_back_node_indices, N_MICE, KP_PRESETS,
)
from criterion_mice import gaussian_2d_nll

_HAS_WANDB = True
try:
    import wandb
except ImportError:
    _HAS_WANDB = False


ALL_KP_NAMES = ["nose", "neck", "center_back", "tail_base"]


# ─── Metrics ─────────────────────────────────────────────────────────

def ade(pred, gt):
    """Average Displacement Error over all nodes. (B, T, N, 2) → (B,)"""
    return torch.norm(pred - gt, dim=-1).mean(dim=(1, 2))


def fde(pred, gt):
    """Final Displacement Error over all nodes. → (B,)"""
    return torch.norm(pred[:, -1] - gt[:, -1], dim=-1).mean(dim=1)


def ade_fde_on_indices(pred, gt, node_indices):
    """ADE and FDE on a subset of node indices. → (ade_B, fde_B)"""
    p = pred[:, :, node_indices, :]
    g = gt[:, :, node_indices, :]
    err = torch.norm(p - g, dim=-1)
    return err.mean(dim=(1, 2)), err[:, -1].mean(dim=1)


def ade_per_mouse(pred, gt, n_kps):
    """ADE broken down per mouse. → (B, N_MICE)"""
    B, T, N, _ = pred.shape
    err = torch.norm(pred - gt, dim=-1)  # (B, T, N)
    err_m = err.reshape(B, T, N_MICE, n_kps)
    return err_m.mean(dim=(1, 3))  # (B, N_MICE)


def body_structure_error(pred, gt, n_kps):
    """
    Mean absolute error in intra-mouse pairwise keypoint distances.
    Returns (B,). Returns zeros if n_kps <= 1.
    """
    if n_kps <= 1:
        return torch.zeros(pred.size(0), device=pred.device)
    B, T, N, _ = pred.shape
    pred_m = pred.reshape(B, T, N_MICE, n_kps, 2)
    gt_m = gt.reshape(B, T, N_MICE, n_kps, 2)

    total_err = torch.zeros(B, device=pred.device)
    count = 0
    for i in range(n_kps):
        for j in range(i + 1, n_kps):
            d_pred = torch.norm(
                pred_m[:, :, :, i] - pred_m[:, :, :, j], dim=-1)
            d_gt = torch.norm(
                gt_m[:, :, :, i] - gt_m[:, :, :, j], dim=-1)
            total_err += (d_pred - d_gt).abs().mean(dim=(1, 2))
            count += 1
    return total_err / count


def inter_mouse_distance_error(pred, gt, cb_indices):
    """
    Error in pairwise distances between mouse centers (center_back).
    cb_indices: list of 3 node indices corresponding to center_back.
    → (B,)
    """
    pred_cb = pred[:, :, cb_indices, :]  # (B, T, 3, 2)
    gt_cb = gt[:, :, cb_indices, :]

    total_err = torch.zeros(pred.size(0), device=pred.device)
    count = 0
    for i in range(N_MICE):
        for j in range(i + 1, N_MICE):
            d_pred = torch.norm(
                pred_cb[:, :, i] - pred_cb[:, :, j], dim=-1)
            d_gt = torch.norm(
                gt_cb[:, :, i] - gt_cb[:, :, j], dim=-1)
            total_err += (d_pred - d_gt).abs().mean(dim=1)
            count += 1
    return total_err / count


# ─── Baselines ───────────────────────────────────────────────────────

def baseline_static(nodes, obs_length):
    last_obs = nodes[:, obs_length - 1: obs_length]
    pred_len = nodes.shape[1] - obs_length
    return last_obs.expand(-1, pred_len, -1, -1)


def baseline_linear(nodes, obs_length):
    pos_t1 = nodes[:, obs_length - 2]
    pos_t0 = nodes[:, obs_length - 1]
    vel = pos_t0 - pos_t1
    pred_len = nodes.shape[1] - obs_length
    preds = [pos_t0 + vel * step for step in range(1, pred_len + 1)]
    return torch.stack(preds, dim=1)


def baseline_constant_velocity(nodes, obs_length, lookback=3):
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
def evaluate(net, loader, obs_length, pred_length, device, n_kps,
             mode="mean", n_samples=1, max_batches=None):
    """
    Run full evaluation.

    Returns
    -------
    metrics : dict  — aggregated scalars
    results : list  — per-batch dicts for visualization
    """
    net.eval()
    cb_idx = get_center_back_node_indices(n_kps)

    all_metrics = {
        "ade": [], "fde": [], "nll": [],
        "cb_ade": [], "cb_fde": [],
        "body_err": [], "inter_dist_err": [],
        "bl_static_ade": [], "bl_static_fde": [],
        "bl_linear_ade": [], "bl_linear_fde": [],
        "bl_cv_ade": [], "bl_cv_fde": [],
        "bl_cb_static_ade": [], "bl_cb_linear_ade": [],
        "bl_cb_cv_ade": [],
    }
    for i in range(N_MICE):
        all_metrics[f"ade_m{i}"] = []
    results = []

    for batch_idx, (nodes_b, activity_b, lights_b, chase_b) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        nodes_b = nodes_b.to(device)
        gt_pred = nodes_b[:, obs_length:]

        predict_out = net.predict(
            nodes_b, obs_length, mode=mode, n_samples=n_samples)
        pred_pos, pred_params = predict_out[0], predict_out[1]
        attn_inter_list = predict_out[2]

        if mode == "sample" and n_samples > 1:
            ade_per_sample = torch.norm(
                pred_pos - gt_pred.unsqueeze(0), dim=-1
            ).mean(dim=(2, 3))
            best_idx = ade_per_sample.argmin(dim=0)
            B = nodes_b.size(0)
            pred_best = pred_pos[best_idx, torch.arange(B)]
        else:
            pred_best = pred_pos

        # All-node ADE/FDE
        all_metrics["ade"].append(ade(pred_best, gt_pred))
        all_metrics["fde"].append(fde(pred_best, gt_pred))

        # Center-back ADE/FDE (cross-config comparable)
        cb_a, cb_f = ade_fde_on_indices(pred_best, gt_pred, cb_idx)
        all_metrics["cb_ade"].append(cb_a)
        all_metrics["cb_fde"].append(cb_f)

        # Structure metrics
        all_metrics["body_err"].append(
            body_structure_error(pred_best, gt_pred, n_kps))
        all_metrics["inter_dist_err"].append(
            inter_mouse_distance_error(pred_best, gt_pred, cb_idx))

        # Per-mouse ADE
        ade_m = ade_per_mouse(pred_best, gt_pred, n_kps)
        for i in range(N_MICE):
            all_metrics[f"ade_m{i}"].append(ade_m[:, i])

        # NLL (teacher-forcing)
        e_t, e_s = build_edges_from_nodes(
            nodes_b, net._spatial_src, net._spatial_dst)
        outputs_tf = net(nodes_b, e_t, e_s)
        if isinstance(outputs_tf, tuple):
            outputs_tf = outputs_tf[0]
        nll = gaussian_2d_nll(outputs_tf, nodes_b, pred_length)
        all_metrics["nll"].append(nll)

        # Baselines — all nodes
        bl_s = baseline_static(nodes_b, obs_length)
        bl_l = baseline_linear(nodes_b, obs_length)
        bl_cv = baseline_constant_velocity(nodes_b, obs_length)

        all_metrics["bl_static_ade"].append(ade(bl_s, gt_pred))
        all_metrics["bl_static_fde"].append(fde(bl_s, gt_pred))
        all_metrics["bl_linear_ade"].append(ade(bl_l, gt_pred))
        all_metrics["bl_linear_fde"].append(fde(bl_l, gt_pred))
        all_metrics["bl_cv_ade"].append(ade(bl_cv, gt_pred))
        all_metrics["bl_cv_fde"].append(fde(bl_cv, gt_pred))

        # Baselines — center_back only
        cb_s_a, _ = ade_fde_on_indices(bl_s, gt_pred, cb_idx)
        cb_l_a, _ = ade_fde_on_indices(bl_l, gt_pred, cb_idx)
        cb_cv_a, _ = ade_fde_on_indices(bl_cv, gt_pred, cb_idx)
        all_metrics["bl_cb_static_ade"].append(cb_s_a)
        all_metrics["bl_cb_linear_ade"].append(cb_l_a)
        all_metrics["bl_cb_cv_ade"].append(cb_cv_a)

        if batch_idx < 5:
            attn_obs = [a.cpu().numpy() for a in attn_inter_list[obs_length - 1:]]
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

    metrics = {}
    for k, v in all_metrics.items():
        vals = torch.cat(v)
        metrics[k] = float(vals.mean())
        metrics[f"{k}_std"] = float(vals.std())

    return metrics, results


def print_metrics(metrics, arena_px=450, n_kps=4):
    scale = arena_px
    cm_per_px = 50.0 / arena_px

    kp_list = KP_PRESETS[n_kps]
    kp_names = [ALL_KP_NAMES[i] for i in kp_list]

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS  (n_kps={n_kps}, keypoints={kp_names})")
    print(f"{'='*70}")

    print(f"\n  ▸ All-Node Trajectory (over {N_MICE * n_kps} nodes):")
    for name, key in [("ADE", "ade"), ("FDE", "fde")]:
        v = metrics[key]
        print(f"    {name:>4s}: {v:.6f}  /  {v*scale:.2f} px  "
              f"/  {v*scale*cm_per_px:.2f} cm")

    print(f"\n  ▸ Center-Back Only (3 nodes, cross-config comparable):")
    for name, key in [("cbADE", "cb_ade"), ("cbFDE", "cb_fde")]:
        v = metrics[key]
        print(f"    {name:>6s}: {v:.6f}  /  {v*scale:.2f} px  "
              f"/  {v*scale*cm_per_px:.2f} cm")

    print(f"\n  ▸ NLL (teacher forcing): {metrics['nll']:.4f}")

    if n_kps > 1:
        print(f"\n  ▸ Structure Preservation:")
        print(f"    Body structure error:     {metrics['body_err']:.6f}  "
              f"({metrics['body_err']*scale:.2f} px)")
    print(f"    Inter-mouse dist error:   {metrics['inter_dist_err']:.6f}  "
          f"({metrics['inter_dist_err']*scale:.2f} px)")

    print(f"\n  ▸ Per-Mouse ADE:")
    for i in range(N_MICE):
        v = metrics[f"ade_m{i}"]
        print(f"    Mouse {i}: {v:.6f}  ({v*scale:.2f} px)")

    # Baseline comparison table: both all-node and center-back
    print(f"\n  ▸ Baseline Comparison (all nodes):")
    header = f"    {'Method':>18s}  {'ADE(px)':>10s}  {'FDE(px)':>10s}"
    print(header)
    print("    " + "-" * 42)
    for name, prefix in [("MouseSRNN", ""), ("Static", "bl_static_"),
                          ("Linear", "bl_linear_"), ("ConstVel", "bl_cv_")]:
        a_key = f"{prefix}ade" if prefix else "ade"
        f_key = f"{prefix}fde" if prefix else "fde"
        a = metrics[a_key] * scale
        f_ = metrics[f_key] * scale
        marker = " ←" if prefix == "" else ""
        print(f"    {name:>18s}  {a:>10.2f}  {f_:>10.2f}{marker}")

    print(f"\n  ▸ Baseline Comparison (center_back only — COMPARABLE):")
    header = f"    {'Method':>18s}  {'cbADE(px)':>10s}"
    print(header)
    print("    " + "-" * 30)
    for name, prefix in [("MouseSRNN", "cb_"), ("Static", "bl_cb_static_"),
                          ("Linear", "bl_cb_linear_"), ("ConstVel", "bl_cb_cv_")]:
        a = metrics[f"{prefix}ade"] * scale
        marker = " ←" if prefix == "cb_" else ""
        print(f"    {name:>18s}  {a:>10.2f}{marker}")

    bl_cb_ades = [metrics["bl_cb_static_ade"], metrics["bl_cb_linear_ade"],
                  metrics["bl_cb_cv_ade"]]
    best_bl = min(bl_cb_ades)
    our_cb = metrics["cb_ade"]
    if best_bl > 0:
        improvement = (1 - our_cb / best_bl) * 100
        print(f"\n  ▸ center_back ADE improvement over best baseline: "
              f"{improvement:+.1f}%")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MouseSRNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.tar)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz data (auto from checkpoint)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--mode", type=str, default="mean",
                        choices=["mean", "sample"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--arena_px", type=float, default=450)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MS_mice")
    parser.add_argument("--wandb_entity", type=str,
                        default="sxiao73-georgia-institute-of-technology")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    args = parser.parse_args()

    # Load checkpoint & recover training config
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    obs_length = saved_args.obs_length
    pred_length = saved_args.pred_length
    n_kps = getattr(saved_args, "n_keypoints", 4)
    graph_type = getattr(saved_args, "graph_type", "full")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: n_keypoints={n_kps}, graph_type={graph_type}, "
          f"n_nodes={N_MICE * n_kps}")

    # Model
    net = MouseSRNN(saved_args).to(device)
    net.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']['total']:.4f})")

    # Data — must match n_keypoints used during training
    data_path = args.data or saved_args.data
    print(f"Data: {data_path}")
    loaders = get_mouse_dataloaders(
        data_path, obs_length=obs_length,
        batch_size=args.batch_size, num_workers=0,
        n_keypoints=n_kps,
    )
    loader = loaders[args.split]
    print(f"Evaluating on {args.split}: {len(loader.dataset)} windows")

    # Wandb
    use_wandb = _HAS_WANDB and not args.no_wandb
    if use_wandb:
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="eval",
            config={**vars(saved_args), "eval_split": args.split,
                    "eval_mode": args.mode, "n_keypoints": n_kps,
                    "graph_type": graph_type},
            tags=["eval", args.split, args.mode,
                  f"kp{n_kps}", graph_type],
        )
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        else:
            exp_tag = getattr(saved_args, "exp_tag", "eval")
            init_kwargs["name"] = (
                f"eval_{exp_tag}_kp{n_kps}_{graph_type}"
                f"_{args.split}_{args.mode}")
        wandb.init(**init_kwargs)
        print(f"wandb: {wandb.run.url}")

    # Run evaluation
    print(f"\nRunning evaluation (mode={args.mode}, n_kps={n_kps})...")
    t0 = time.time()
    metrics, results = evaluate(
        net, loader, obs_length, pred_length, device, n_kps=n_kps,
        mode=args.mode,
        n_samples=args.n_samples if args.mode == "sample" else 1,
        max_batches=args.max_batches,
    )
    elapsed = time.time() - t0
    print(f"Evaluation completed in {elapsed:.1f}s")

    print_metrics(metrics, arena_px=args.arena_px, n_kps=n_kps)

    # Save
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(
        out_dir, f"metrics_{args.split}_{args.mode}_kp{n_kps}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    results_path = os.path.join(
        out_dir, f"results_{args.split}_{args.mode}_kp{n_kps}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to: {results_path}")

    if use_wandb:
        scale = args.arena_px
        wandb.summary.update({
            f"eval/{args.split}_ade_px": metrics["ade"] * scale,
            f"eval/{args.split}_fde_px": metrics["fde"] * scale,
            f"eval/{args.split}_cb_ade_px": metrics["cb_ade"] * scale,
            f"eval/{args.split}_cb_fde_px": metrics["cb_fde"] * scale,
            f"eval/{args.split}_nll": metrics["nll"],
            f"eval/{args.split}_body_err_px": metrics["body_err"] * scale,
            f"eval/{args.split}_inter_dist_err_px":
                metrics["inter_dist_err"] * scale,
            f"eval/{args.split}_bl_cb_static_ade_px":
                metrics["bl_cb_static_ade"] * scale,
        })

        table = wandb.Table(
            columns=["Method", "All-ADE(px)", "All-FDE(px)", "cbADE(px)"])
        for name, prefix, cb_prefix in [
            ("MouseSRNN", "", "cb_"),
            ("Static", "bl_static_", "bl_cb_static_"),
            ("Linear", "bl_linear_", "bl_cb_linear_"),
            ("ConstVel", "bl_cv_", "bl_cb_cv_"),
        ]:
            a = metrics[f"{prefix}ade"] * scale
            f_ = metrics[f"{prefix}fde"] * scale
            cb = metrics[f"{cb_prefix}ade"] * scale
            table.add_data(name, round(a, 2), round(f_, 2), round(cb, 2))
        wandb.log({"eval/baseline_comparison": table})

        art = wandb.Artifact(
            f"eval-kp{n_kps}-{graph_type}-{args.split}-{args.mode}",
            type="eval_results", metadata=metrics)
        art.add_file(metrics_path)
        wandb.log_artifact(art)

        wandb.finish()
        print("wandb evaluation run finished")


if __name__ == "__main__":
    main()
