"""
Attention & trajectory visualization for MouseSRNN.

Per sample generates:
  - 12 intra figures (one per node → same-mouse keypoints)
  - 12 inter figures (one per node → other-mouse keypoints)
  - 12×12 full attention heatmap
  - 3×3 mouse-level mean attention heatmap
  - CB trajectory prediction (original Social Attention style)

Usage:
  # Single sample (auto-pick high activity):
  python visualize_attn.py --checkpoint ../checkpoints/best_model.tar \
    --data ../data/mice/dataset_r1_w20_s10_fs4.npz

  # Multiple specific samples:
  python visualize_attn.py --checkpoint ... --data ... --sample_idx 100 200 300

  # Auto-select best + active + chase:
  python visualize_attn.py --checkpoint ... --data ... --auto_select \
    --n_best 6 --n_active 3
"""
import torch
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MouseSRNN, build_edges_from_nodes

ARENA_PX = 450
N_MICE = 3
N_KPS = 4
N_NODES = 12
OBS_LEN = 10
CB_LOCAL = 2
KP = ["nose", "ear", "cb", "tail"]
MC = ["#2ecc71", "#3498db", "#e74c3c"]
NODE_LABELS = [f"M{n // N_KPS}_{KP[n % N_KPS]}" for n in range(N_NODES)]


def node_label(n):
    return f"M{n // N_KPS}_{KP[n % N_KPS]}"


# ─── Model inference ─────────────────────────────────────────────────

def run_model(net, nodes_tensor, obs_length):
    B, T, N, _ = nodes_tensor.shape
    dev = nodes_tensor.device
    ht = torch.zeros(B, N, net.er, device=dev)
    ct = torch.zeros(B, N, net.er, device=dev)
    hs = torch.zeros(B, net.n_spatial, net.er, device=dev)
    cs = torch.zeros(B, net.n_spatial, net.er, device=dev)
    hn = torch.zeros(B, N, net.nr, device=dev)
    cn = torch.zeros(B, N, net.nr, device=dev)
    et, es = build_edges_from_nodes(
        nodes_tensor, net._spatial_src, net._spatial_dst)

    preds, a_inter, a_intra = [], [], []
    for t in range(T):
        inp = nodes_tensor[:, t] if t < obs_length else nxt
        te = net.temporal_edge_enc(et[:, min(t, T - 1)])
        hf, cf = net.temporal_edge_rnn(
            te.reshape(B * N, -1),
            (ht.reshape(B * N, -1), ct.reshape(B * N, -1)))
        ht, ct = hf.reshape(B, N, -1), cf.reshape(B, N, -1)

        se = net.spatial_edge_enc(net._encode_spatial(es[:, min(t, T - 1)]))
        hf2, cf2 = net.spatial_edge_rnn(
            se.reshape(B * net.n_spatial, -1),
            (hs.reshape(B * net.n_spatial, -1),
             cs.reshape(B * net.n_spatial, -1)))
        hs, cs = hf2.reshape(B, net.n_spatial, -1), cf2.reshape(B, net.n_spatial, -1)

        hi, he, wia, wie, _ = net._attend(ht, hs, B, N, dev)
        ni = net.node_enc(inp)
        ei = net.edge_attn_enc(torch.cat([ht, hi, he], dim=-1))
        ri = torch.cat([ni, ei], dim=-1)
        hf3, cf3 = net.node_rnn(
            ri.reshape(B * N, -1),
            (hn.reshape(B * N, -1), cn.reshape(B * N, -1)))
        hn, cn = hf3.reshape(B, N, -1), cf3.reshape(B, N, -1)

        o = net.output_linear(hn)
        if net.residual:
            o = o.clone()
            o[..., :2] = o[..., :2] + inp
        nxt = o[..., :2]
        if t >= obs_length:
            preds.append(nxt.cpu().numpy())
        a_inter.append(wie.cpu().numpy())
        if wia is not None:
            a_intra.append(wia.cpu().numpy())

    pred = np.stack(preds, axis=1) if preds else None
    return pred, a_inter, a_intra


# ─── Attention matrix builders ───────────────────────────────────────

def build_full_12x12(w_inter, w_intra):
    mat = np.zeros((N_NODES, N_NODES))
    for src in range(N_NODES):
        sm, sk = src // N_KPS, src % N_KPS
        others = sorted(m for m in range(N_MICE) if m != sm)
        for oi, md in enumerate(others):
            for dk in range(N_KPS):
                mat[src, md * N_KPS + dk] = w_inter[src, oi * N_KPS + dk]
        if w_intra is not None:
            dst_kps = [k for k in range(N_KPS) if k != sk]
            for ki, dk in enumerate(dst_kps):
                if ki < w_intra.shape[1]:
                    mat[src, sm * N_KPS + dk] = w_intra[src, ki]
    return mat


def build_mouse_level(full_mat):
    mmat = np.zeros((N_MICE, N_MICE))
    for ms in range(N_MICE):
        for md in range(N_MICE):
            vals = []
            for sk in range(N_KPS):
                for dk in range(N_KPS):
                    if ms != md or sk != dk:
                        vals.append(full_mat[ms * N_KPS + sk, md * N_KPS + dk])
            mmat[ms, md] = np.mean(vals) if vals else 0
    return mmat


# ─── Plot functions ──────────────────────────────────────────────────

def plot_intra_from_node(gt, src_node, w_intra, save_path):
    s = ARENA_PX
    pos = gt[OBS_LEN - 1] * s
    m = src_node // N_KPS
    sk = src_node % N_KPS
    cm = MC[m]

    pts = [pos[m * N_KPS + k] for k in range(N_KPS)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mg = 35
    xl, xh = min(xs) - mg, max(xs) + mg
    yl, yh = min(ys) - mg, max(ys) + mg
    if xh - xl < 70: cx = (xl + xh) / 2; xl, xh = cx - 35, cx + 35
    if yh - yl < 70: cy = (yl + yh) / 2; yl, yh = cy - 35, cy + 35

    fig, ax = plt.subplots(figsize=(6, 6))
    for b1, b2 in [(0, 1), (1, 2), (2, 3)]:
        p1, p2 = pos[m * N_KPS + b1], pos[m * N_KPS + b2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ddd", lw=2, zorder=0)

    weights = {}
    if w_intra is not None:
        for ki, dk in enumerate(k for k in range(N_KPS) if k != sk):
            if ki < w_intra.shape[1]:
                weights[dk] = float(w_intra[src_node, ki])

    for kp in range(N_KPS):
        p = pos[m * N_KPS + kp]
        if kp == sk:
            ax.scatter(p[0], p[1], color=cm, s=80, zorder=10,
                       edgecolors="black", linewidths=2)
        else:
            w = weights.get(kp, 0.0)
            ax.scatter(p[0], p[1], color=cm, s=40, zorder=8,
                       edgecolors="black", linewidths=0.8)
            r = max(w * s * 0.06, 2)
            ax.add_artist(plt.Circle((p[0], p[1]), r,
                                     fill=False, color="b", lw=2, zorder=5))
        ax.text(p[0], p[1] - 5, KP[kp], fontsize=8,
                ha="center", va="top", color=cm)

    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh)
    ax.set_aspect("equal")
    ax.set_title(f"Intra: {node_label(src_node)}", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15); ax.tick_params(labelsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_inter_from_node(gt, src_node, w_inter, save_path):
    s = ARENA_PX
    pos = gt[OBS_LEN - 1] * s
    ms = src_node // N_KPS
    sk = src_node % N_KPS
    others = sorted(m for m in range(N_MICE) if m != ms)

    wts = {}
    for oi, md in enumerate(others):
        for dk in range(N_KPS):
            wts[(md, dk)] = float(w_inter[src_node, oi * N_KPS + dk])

    pts = [pos[n] for n in range(N_NODES)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mg = 30
    xl, xh = min(xs) - mg, max(xs) + mg
    yl, yh = min(ys) - mg, max(ys) + mg
    xr, yr = xh - xl, yh - yl
    fw = 7 if xr >= yr else max(4, 7 * xr / yr)
    fh = 7 if yr >= xr else max(4, 7 * yr / xr)

    fig, ax = plt.subplots(figsize=(fw, fh))
    for mo in range(N_MICE):
        for b1, b2 in [(0, 1), (1, 2), (2, 3)]:
            p1, p2 = pos[mo * N_KPS + b1], pos[mo * N_KPS + b2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=MC[mo], lw=1.5, alpha=0.3, zorder=0)
    for k in range(N_KPS):
        p = pos[ms * N_KPS + k]
        if k == sk:
            ax.scatter(p[0], p[1], color=MC[ms], s=60, zorder=10,
                       edgecolors="black", linewidths=2)
        else:
            ax.scatter(p[0], p[1], color="#bbb", s=20, zorder=2)
    for md in others:
        for dk in range(N_KPS):
            p = pos[md * N_KPS + dk]
            w = wts[(md, dk)]
            ax.scatter(p[0], p[1], color=MC[md], s=35, zorder=8,
                       edgecolors="black", linewidths=0.5)
            r = max(w * s * 0.06, 2)
            ax.add_artist(plt.Circle((p[0], p[1]), r,
                                     fill=False, color="b", lw=2, zorder=5))
            ax.text(p[0], p[1] - max(r, 4) - 2, KP[dk],
                    fontsize=6, ha="center", va="top", color=MC[md])

    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh)
    ax.set_aspect("equal")
    ax.set_title(f"Inter: {node_label(src_node)}", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15); ax.tick_params(labelsize=8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_heatmap_12x12(full_mat, save_path, title=""):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(full_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for i in range(N_NODES):
        for j in range(N_NODES):
            v = full_mat[i, j]
            if i // N_KPS == j // N_KPS and i % N_KPS == j % N_KPS:
                ax.text(j, i, "-", ha="center", va="center",
                        fontsize=6, color="gray")
            elif v > 0.001:
                c = "white" if v > 0.4 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=c)
    ax.set_xticks(range(N_NODES))
    ax.set_xticklabels(NODE_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(N_NODES))
    ax.set_yticklabels(NODE_LABELS, fontsize=7)
    for sep in [4, 8]:
        ax.axhline(y=sep - 0.5, color="white", lw=2)
        ax.axvline(x=sep - 0.5, color="white", lw=2)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_ylabel("Source →"); ax.set_xlabel("→ Target")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_mouse(mmat, save_path, title=""):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mmat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    for i in range(N_MICE):
        for j in range(N_MICE):
            v = mmat[i, j]
            c = "white" if v > 0.25 else "black"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=10, color=c, fontweight="bold")
    ax.set_xticks(range(N_MICE))
    ax.set_xticklabels([f"Mouse {i}" for i in range(N_MICE)], fontsize=9)
    ax.set_yticks(range(N_MICE))
    ax.set_yticklabels([f"Mouse {i}" for i in range(N_MICE)], fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(gt, pred, save_path, title=""):
    s = ARENA_PX
    fig, ax = plt.subplots(figsize=(7, 7))
    for m in range(N_MICE):
        c = MC[m]
        cb = m * N_KPS + CB_LOCAL
        true_xy = gt[:, cb] * s
        ax.plot(true_xy[:OBS_LEN, 0], true_xy[:OBS_LEN, 1], color=c,
                ls="-", lw=2, marker="o", ms=4,
                markevery=list(range(OBS_LEN - 1)), zorder=3)
        ax.plot(true_xy[OBS_LEN - 1:, 0], true_xy[OBS_LEN - 1:, 1],
                color=c, ls="-", lw=1.2, marker="o", ms=2, alpha=0.4, zorder=2)
        ax.scatter(true_xy[OBS_LEN - 1, 0], true_xy[OBS_LEN - 1, 1],
                   color="b", marker="D", s=70, zorder=6)
        if pred is not None:
            pred_xy = pred[:, cb] * s
            full = np.vstack([true_xy[OBS_LEN - 1:OBS_LEN], pred_xy])
            ax.plot(full[:, 0], full[:, 1], color=c, ls="--", lw=2,
                    marker="x", ms=5,
                    markevery=list(range(1, len(full))), zorder=4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ─── Process one sample ──────────────────────────────────────────────

def process_sample(net, gt, idx, out_dir, tag, device):
    s = ARENA_PX
    batch = torch.tensor(gt[np.newaxis], dtype=torch.float32).to(device)
    pred, a_inter, a_intra = run_model(net, batch, OBS_LEN)
    pred_s = pred[0] if pred is not None else None
    w_inter = a_inter[OBS_LEN - 1][0]
    w_intra = a_intra[OBS_LEN - 1][0] if a_intra else None

    cb_idx = [m * N_KPS + CB_LOCAL for m in range(N_MICE)]
    cb_ade = (np.linalg.norm(
        pred_s[:, cb_idx] - gt[OBS_LEN:, cb_idx], axis=-1).mean() * s
        if pred_s is not None else 0)

    sdir = os.path.join(out_dir, f"{tag}_ade{cb_ade:.0f}")
    intra_dir = os.path.join(sdir, "intra")
    inter_dir = os.path.join(sdir, "inter")
    os.makedirs(intra_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    # 12 intra
    for src in range(N_NODES):
        plot_intra_from_node(gt, src, w_intra,
                             os.path.join(intra_dir, f"{node_label(src)}.png"))
    # 12 inter
    for src in range(N_NODES):
        plot_inter_from_node(gt, src, w_inter,
                             os.path.join(inter_dir, f"{node_label(src)}.png"))
    # 12×12 heatmap
    full_mat = build_full_12x12(w_inter, w_intra)
    plot_heatmap_12x12(full_mat,
                       os.path.join(sdir, "heatmap_12x12.png"),
                       f"Attention 12×12 — cbADE={cb_ade:.1f}px, {tag}")
    # 3×3 mouse heatmap
    mmat = build_mouse_level(full_mat)
    plot_heatmap_mouse(mmat,
                       os.path.join(sdir, "heatmap_mouse.png"),
                       f"Mouse-Level — {tag}")
    # Trajectory
    plot_trajectory(gt, pred_s,
                    os.path.join(sdir, "trajectory.png"),
                    f"Trajectory — cbADE={cb_ade:.1f}px, {tag}")

    return cb_ade


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention & trajectory visualization for MouseSRNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, nargs="*", default=None,
                        help="Specific test indices")
    parser.add_argument("--auto_select", action="store_true",
                        help="Auto-select best/active/chase samples")
    parser.add_argument("--n_best", type=int, default=6)
    parser.add_argument("--n_active", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--arena_px", type=float, default=450)
    args = parser.parse_args()

    global ARENA_PX
    ARENA_PX = args.arena_px

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    net = MouseSRNN(saved_args).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    raw = np.load(args.data)
    test_data = raw["test_data"].reshape(-1, 20, N_NODES, 2)
    test_act = raw["test_activity"]
    test_chase = raw["test_chase"]

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), "plots", "attn_viz")
    os.makedirs(out_dir, exist_ok=True)

    # Determine sample list
    if args.sample_idx is not None:
        samples = [(idx, "manual") for idx in args.sample_idx]
    elif args.auto_select:
        print("Computing CB ADE for auto-selection...")
        cb_idx = [m * N_KPS + CB_LOCAL for m in range(N_MICE)]
        ade_list = []
        with torch.no_grad():
            for i in range(0, len(test_data), 64):
                batch = torch.tensor(
                    test_data[i:i + 64], dtype=torch.float32).to(device)
                pred, _, _ = run_model(net, batch, OBS_LEN)
                if pred is not None:
                    gt_p = test_data[i:i + 64, OBS_LEN:]
                    for j in range(pred.shape[0]):
                        ade = np.linalg.norm(
                            pred[j, :, cb_idx] - gt_p[j, :, cb_idx],
                            axis=-1).mean() * ARENA_PX
                        ade_list.append((i + j, ade))
        ade_list.sort(key=lambda x: x[1])
        ade_dict = dict(ade_list)

        seen = set()
        samples = []
        for idx, _ in ade_list[:args.n_best]:
            if idx not in seen:
                seen.add(idx); samples.append((idx, "best"))
        act_sorted = np.argsort(test_act)[::-1]
        cnt = 0
        for idx in act_sorted:
            if cnt >= args.n_active:
                break
            if idx not in seen and ade_dict.get(int(idx), 999) < 15:
                seen.add(int(idx)); samples.append((int(idx), "active"))
                cnt += 1
        for idx in np.where(test_chase == 1)[0]:
            if int(idx) not in seen:
                seen.add(int(idx)); samples.append((int(idx), "chase"))
        print(f"Selected {len(samples)} samples")
    else:
        idx = int(np.argsort(test_act)[int(0.95 * len(test_act))])
        samples = [(idx, "default")]

    # Process each sample
    with torch.no_grad():
        for rank, (idx, tag) in enumerate(samples):
            label = f"{rank:02d}_{tag}"
            ade = process_sample(net, test_data[idx], idx, out_dir,
                                 label, device)
            chase = "chase" if test_chase[idx] else ""
            print(f"  [{rank:2d}] {tag:7s} idx={idx:4d} cbADE={ade:5.1f}px {chase}")

    print(f"\nAll saved to {out_dir}/")


if __name__ == "__main__":
    main()
