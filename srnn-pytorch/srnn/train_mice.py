"""
Training script for MouseSRNN on preprocessed mouse trajectory data.

Features:
  - Fully vectorized MouseSRNN (batch tensor ops, no Python loops)
  - Combined loss: activity-weighted Gaussian NLL + relative distance MSE
  - Multi-GPU support via DataParallel (auto-detected)
  - AMP mixed-precision training for speed
  - Weights & Biases (wandb) integration
  - Periodic autoregressive evaluation (ADE/FDE/baselines)

Usage:
    python train_mice.py --exp_tag run01
    python train_mice.py --data ../data/mice/dataset_r1_w20_s10_fs4.npz --num_epochs 100
    python train_mice.py --no_wandb --no_amp   # disable wandb and AMP
"""

import argparse
import json
import os
import pickle
import time

import torch
import torch.nn as nn

from model_mice import MouseSRNN, build_edges_from_nodes
from mouse_dataset import get_mouse_dataloaders, N_NODES
from criterion_mice import combined_loss

_HAS_WANDB = True
try:
    import wandb
except ImportError:
    _HAS_WANDB = False


def _ts():
    return time.strftime("%H:%M:%S")


def _get_raw_model(net):
    """Unwrap DataParallel to get the raw MouseSRNN."""
    if isinstance(net, nn.DataParallel):
        return net.module
    return net


# ─── Training / Validation ──────────────────────────────────────────

def train_one_epoch(net, loader, optimizer, args, device, scheduler=None,
                    use_wandb=False, global_step=0, scaler=None):
    net.train()
    epoch_losses = {"total": 0, "nll": 0, "dist": 0, "w_sum": 0, "n": 0}
    use_amp = scaler is not None

    for batch_idx, (nodes_b, activity_b, _, _) in enumerate(loader):
        nodes_b = nodes_b.to(device, non_blocking=True)
        activity_b = activity_b.to(device, non_blocking=True)

        edges_temp, edges_spat = build_edges_from_nodes(nodes_b)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(nodes_b, edges_temp, edges_spat)
            loss, ld = combined_loss(
                outputs, nodes_b, activity_b,
                pred_length=args.pred_length,
                lambda_dist=args.lambda_dist,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), args.grad_clip)
            optimizer.step()

        bs = nodes_b.size(0)
        epoch_losses["total"] += ld["total"] * bs
        epoch_losses["nll"] += ld["nll"] * bs
        epoch_losses["dist"] += ld["dist"] * bs
        epoch_losses["w_sum"] += ld["w_mean"] * bs
        epoch_losses["n"] += bs

        if use_wandb and batch_idx % 20 == 0:
            wandb.log({
                "batch/loss": ld["total"],
                "batch/nll": ld["nll"],
                "batch/dist": ld["dist"],
                "batch/grad_norm": grad_norm.item()
                    if hasattr(grad_norm, "item") else grad_norm,
            }, step=global_step + batch_idx)

        if batch_idx % 100 == 0:
            print(f"  [{_ts()}] batch {batch_idx:>4d}/{len(loader)} | "
                  f"loss={ld['total']:.4f} nll={ld['nll']:.4f} "
                  f"dist={ld['dist']:.6f}", flush=True)

    if scheduler is not None:
        scheduler.step()

    n = epoch_losses["n"]
    return {
        "total": epoch_losses["total"] / n,
        "nll": epoch_losses["nll"] / n,
        "dist": epoch_losses["dist"] / n,
        "w_mean": epoch_losses["w_sum"] / n,
    }


@torch.no_grad()
def validate(net, loader, args, device):
    net.eval()
    epoch_losses = {"total": 0, "nll": 0, "dist": 0, "n": 0}

    for nodes_b, activity_b, _, _ in loader:
        nodes_b = nodes_b.to(device, non_blocking=True)
        activity_b = activity_b.to(device, non_blocking=True)

        edges_temp, edges_spat = build_edges_from_nodes(nodes_b)
        outputs = net(nodes_b, edges_temp, edges_spat)

        _, ld = combined_loss(
            outputs, nodes_b, activity_b,
            pred_length=args.pred_length,
            lambda_dist=args.lambda_dist,
        )

        bs = nodes_b.size(0)
        epoch_losses["total"] += ld["total"] * bs
        epoch_losses["nll"] += ld["nll"] * bs
        epoch_losses["dist"] += ld["dist"] * bs
        epoch_losses["n"] += bs

    n = epoch_losses["n"]
    return {k: v / n for k, v in epoch_losses.items() if k != "n"}


# ─── Autoregressive Evaluation (ADE/FDE) ────────────────────────────

@torch.no_grad()
def eval_ade_fde(net, loader, obs_length, pred_length, device,
                 max_batches=5, arena_px=450):
    raw_net = _get_raw_model(net)
    raw_net.eval()
    ade_list, fde_list = [], []
    bl_static_ade, bl_linear_ade = [], []

    for batch_idx, (nodes_b, activity_b, _, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        nodes_b = nodes_b.to(device)
        gt_pred = nodes_b[:, obs_length:]

        pred_pos, _, _ = raw_net.predict(nodes_b, obs_length, mode="mean")

        err = torch.norm(pred_pos - gt_pred, dim=-1)
        ade_list.append(err.mean(dim=(1, 2)))
        fde_list.append(err[:, -1].mean(dim=1))

        bl_s = nodes_b[:, obs_length - 1: obs_length].expand(
            -1, pred_pos.size(1), -1, -1)
        bl_err = torch.norm(bl_s - gt_pred, dim=-1)
        bl_static_ade.append(bl_err.mean(dim=(1, 2)))

        vel = nodes_b[:, obs_length - 1] - nodes_b[:, obs_length - 2]
        bl_l = torch.stack([nodes_b[:, obs_length - 1] + vel * s
                            for s in range(1, pred_pos.size(1) + 1)], dim=1)
        bl_l_err = torch.norm(bl_l - gt_pred, dim=-1)
        bl_linear_ade.append(bl_l_err.mean(dim=(1, 2)))

    scale = arena_px
    results = {
        "ade_px": torch.cat(ade_list).mean().item() * scale,
        "fde_px": torch.cat(fde_list).mean().item() * scale,
        "bl_static_ade_px": torch.cat(bl_static_ade).mean().item() * scale,
        "bl_linear_ade_px": torch.cat(bl_linear_ade).mean().item() * scale,
    }
    results["improvement_vs_static"] = (
        1 - results["ade_px"] / max(results["bl_static_ade_px"], 1e-6)
    ) * 100
    return results


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MouseSRNN on mouse trajectory data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data", type=str,
                        default="../data/mice/dataset_combined_w20_s10_fs4.npz")
    parser.add_argument("--obs_length", type=int, default=10)
    parser.add_argument("--pred_length", type=int, default=10)

    # Model
    parser.add_argument("--human_node_rnn_size", type=int, default=128)
    parser.add_argument("--human_human_edge_rnn_size", type=int, default=256)
    parser.add_argument("--human_node_input_size", type=int, default=2)
    parser.add_argument("--human_human_edge_input_size", type=int, default=2)
    parser.add_argument("--human_node_output_size", type=int, default=5)
    parser.add_argument("--human_node_embedding_size", type=int, default=64)
    parser.add_argument("--human_human_edge_embedding_size", type=int, default=64)
    parser.add_argument("--attention_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="L2 weight decay for AdamW (original paper=5e-5)")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup epochs (0 to disable)")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--lambda_dist", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--arena_px", type=float, default=450)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP mixed precision")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batches", type=int, default=5)

    # Output
    parser.add_argument("--exp_tag", type=str, default="mice_c")
    parser.add_argument("--save_every", type=int, default=10)

    # Wandb
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MS_mice")
    parser.add_argument("--wandb_entity", type=str,
                        default="sxiao73-georgia-institute-of-technology")

    args = parser.parse_args()
    args.seq_length = args.obs_length + args.pred_length

    if not os.path.isabs(args.data):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data = os.path.join(script_dir, args.data)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    log_dir = os.path.join(base_dir, "log", "mice", args.exp_tag)
    save_dir = os.path.join(base_dir, "save", "mice", args.exp_tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        pickle.dump(args, f)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Wandb ──
    use_wandb = _HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.exp_tag, config=vars(args),
            tags=["mouseSRNN", f"obs{args.obs_length}",
                  f"pred{args.pred_length}"],
            save_code=True,
        )
        wandb.run.log_code(
            root=base_dir,
            include_fn=lambda p: p.endswith(".py"),
        )
        print(f"[{_ts()}] wandb: {wandb.run.url}", flush=True)
    elif not _HAS_WANDB and not args.no_wandb:
        print(f"[{_ts()}] WARNING: wandb not installed", flush=True)

    # ── Device & Multi-GPU ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"[{_ts()}] Device: {device}, GPUs: {n_gpu}", flush=True)
    for i in range(n_gpu):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

    # Scale batch size by number of GPUs
    effective_batch = args.batch_size * max(n_gpu, 1)
    print(f"  Effective batch size: {args.batch_size} x {max(n_gpu,1)} = "
          f"{effective_batch}", flush=True)

    # ── Data ──
    print(f"[{_ts()}] Loading data from {args.data}", flush=True)
    t0 = time.time()
    loaders = get_mouse_dataloaders(
        args.data, obs_length=args.obs_length,
        batch_size=effective_batch, num_workers=args.num_workers,
    )
    print(f"[{_ts()}] Data loaded in {time.time()-t0:.1f}s", flush=True)
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)
    print(f"  Train: {n_train} windows, {len(loaders['train'])} batches",
          flush=True)
    print(f"  Val:   {n_val} windows, {len(loaders['val'])} batches",
          flush=True)
    if use_wandb:
        wandb.config.update({"n_train": n_train, "n_val": n_val,
                              "n_gpu": n_gpu, "effective_batch": effective_batch},
                             allow_val_change=True)

    # ── Model ──
    net = MouseSRNN(args).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[{_ts()}] MouseSRNN parameters: {n_params:,}", flush=True)

    if n_gpu > 1:
        net = nn.DataParallel(net)
        print(f"  Wrapped in DataParallel ({n_gpu} GPUs)", flush=True)

    if use_wandb:
        wandb.config.update({"n_params": n_params}, allow_val_change=True)
        wandb.watch(_get_raw_model(net), log="gradients", log_freq=200)

    # ── AMP scaler ──
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"  AMP mixed precision: {'ON' if use_amp else 'OFF'}", flush=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)

    # LR schedule: optional linear warmup → cosine annealing
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - warmup_epochs, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-5)

    # CSV log
    log_file = open(os.path.join(log_dir, "log_curve.csv"), "w")
    log_file.write("epoch,train_total,train_nll,train_dist,"
                   "val_total,val_nll,val_dist,lr,"
                   "ade_px,fde_px,bl_static_ade_px,bl_linear_ade_px\n")

    best_val = float("inf")
    best_epoch = 0
    best_ade = float("inf")

    print(f"\n{'='*60}")
    print(f"  Training: {args.num_epochs} epochs, batch={effective_batch}")
    print(f"  AdamW weight_decay={args.weight_decay}")
    print(f"  LR warmup: {warmup_epochs} epochs → cosine to 1e-5")
    print(f"  Eval every {args.eval_every} epochs")
    print(f"{'='*60}\n", flush=True)

    for epoch in range(args.num_epochs):
        t0 = time.time()
        print(f"[{_ts()}] === Epoch {epoch}/{args.num_epochs} ===", flush=True)

        global_step = epoch * len(loaders["train"])

        train_loss = train_one_epoch(
            net, loaders["train"], optimizer, args, device, scheduler,
            use_wandb=use_wandb, global_step=global_step, scaler=scaler,
        )

        val_loss = validate(net, loaders["val"], args, device)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        eval_metrics = {}
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.num_epochs - 1)
        if do_eval:
            print(f"  [{_ts()}] Running ADE/FDE evaluation...", flush=True)
            t_eval = time.time()
            eval_metrics = eval_ade_fde(
                net, loaders["val"], args.obs_length, args.pred_length,
                device, max_batches=args.eval_batches, arena_px=args.arena_px,
            )
            print(f"  [{_ts()}] ADE={eval_metrics['ade_px']:.2f}px  "
                  f"FDE={eval_metrics['fde_px']:.2f}px  "
                  f"Static={eval_metrics['bl_static_ade_px']:.2f}px  "
                  f"Linear={eval_metrics['bl_linear_ade_px']:.2f}px  "
                  f"Δ={eval_metrics['improvement_vs_static']:+.1f}%  "
                  f"({time.time()-t_eval:.1f}s)", flush=True)

            if eval_metrics["ade_px"] < best_ade:
                best_ade = eval_metrics["ade_px"]

        print(f"[{_ts()}] Epoch {epoch:>3d}/{args.num_epochs} ({elapsed:.0f}s) | "
              f"Train: {train_loss['total']:.4f} "
              f"(nll={train_loss['nll']:.4f}, dist={train_loss['dist']:.6f}) | "
              f"Val: {val_loss['total']:.4f} "
              f"(nll={val_loss['nll']:.4f}, dist={val_loss['dist']:.6f}) | "
              f"lr={lr:.6f}", flush=True)

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/total": train_loss["total"],
                "train/nll": train_loss["nll"],
                "train/dist": train_loss["dist"],
                "train/w_mean": train_loss["w_mean"],
                "val/total": val_loss["total"],
                "val/nll": val_loss["nll"],
                "val/dist": val_loss["dist"],
                "lr": lr, "epoch_time_s": elapsed,
            }
            if eval_metrics:
                log_dict.update({
                    "eval/ade_px": eval_metrics["ade_px"],
                    "eval/fde_px": eval_metrics["fde_px"],
                    "eval/bl_static_ade_px": eval_metrics["bl_static_ade_px"],
                    "eval/bl_linear_ade_px": eval_metrics["bl_linear_ade_px"],
                    "eval/improvement_vs_static_pct": eval_metrics["improvement_vs_static"],
                    "eval/best_ade_px": best_ade,
                })
            wandb.log(log_dict, step=global_step + len(loaders["train"]))

        ade_str = f"{eval_metrics.get('ade_px', '')}"
        fde_str = f"{eval_metrics.get('fde_px', '')}"
        bls_str = f"{eval_metrics.get('bl_static_ade_px', '')}"
        bll_str = f"{eval_metrics.get('bl_linear_ade_px', '')}"
        log_file.write(
            f"{epoch},{train_loss['total']:.6f},{train_loss['nll']:.6f},"
            f"{train_loss['dist']:.6f},{val_loss['total']:.6f},"
            f"{val_loss['nll']:.6f},{val_loss['dist']:.6f},{lr:.8f},"
            f"{ade_str},{fde_str},{bls_str},{bll_str}\n")
        log_file.flush()

        # Checkpointing — always save raw model (unwrap DP)
        raw_sd = _get_raw_model(net).state_dict()
        ckpt = {
            "epoch": epoch, "state_dict": raw_sd,
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss, "args": vars(args),
        }

        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            best_epoch = epoch
            torch.save(ckpt, os.path.join(save_dir, "best_model.tar"))

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(save_dir, f"model_epoch{epoch}.tar"))

    log_file.close()

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best epoch: {best_epoch}, Best val loss: {best_val:.4f}")
    print(f"  Best ADE: {best_ade:.2f} px")
    print(f"  Model saved to: {save_dir}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_val_loss"] = best_val
        wandb.summary["best_ade_px"] = best_ade

        artifact = wandb.Artifact(
            name=f"model-{args.exp_tag}", type="model",
            description=f"MouseSRNN best (epoch {best_epoch}, "
                        f"val={best_val:.4f})",
            metadata=vars(args),
        )
        best_path = os.path.join(save_dir, "best_model.tar")
        if os.path.exists(best_path):
            artifact.add_file(best_path)
        artifact.add_file(os.path.join(log_dir, "config.json"))
        artifact.add_file(os.path.join(log_dir, "log_curve.csv"))
        wandb.log_artifact(artifact)
        print(f"  wandb artifact uploaded: model-{args.exp_tag}")
        wandb.finish()
        print(f"  wandb run finished")


if __name__ == "__main__":
    main()
