# -*- coding: utf-8 -*-
'''
Attention visualization for mouse trajectory prediction (15 nodes = 3 mice × 5 keypoints).
- Legacy style: circle radius = weight, partial trajectory, few points per mouse when occlusion.
- 权重和不为1说明: 每个节点对14个目标的attention经softmax后和为1; 3×3热力图是5个节点的平均,
  且只展示跨鼠部分, 同鼠内4个关键点占大部分attention, 故跨鼠行和约0.1~0.2.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import argparse
import seaborn as sns
import os

# 5 keypoints per mouse: Nose(0), L-Ear(1), R-Ear(2), CenterBack(3), TailBase(4)
MOUSE_5PT_CONNECTIONS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
MICE_COLORS = ('#2ecc71', '#3498db', '#e74c3c')  # green, blue, red
KPS_PER_MOUSE = 5

# 遮挡时每鼠只取的代表点 (Nose, CenterBack) 用于画权重圆
REP_NODES_PER_MOUSE = [0, 3]  # 每鼠2个点
OCCLUSION_DIST = 0.15  # 两鼠中心距离小于此视为遮挡, 只画代表点

# 15节点名称: Nose, L-Ear, R-Ear, CenterBack, TailBase
KP_NAMES = ['Nose', 'L-Ear', 'R-Ear', 'CenterBack', 'TailBase']
NODE_NAMES = ['M{}-{}'.format(m, KP_NAMES[k]) for m in range(3) for k in range(5)]


def get_mouse_attn_matrix(attn_t):
    """Aggregate 15-node attention to 3×3 mouse-level matrix. attn_t: dict node->(weights, others)."""
    M = np.zeros((3, 3))
    for src_m in range(3):
        for dst_m in range(3):
            if src_m == dst_m:
                continue
            vals = []
            for src_n in range(src_m * KPS_PER_MOUSE, (src_m + 1) * KPS_PER_MOUSE):
                if src_n not in attn_t:
                    continue
                w, others = attn_t[src_n]
                for i, o in enumerate(others):
                    if dst_m * KPS_PER_MOUSE <= o < (dst_m + 1) * KPS_PER_MOUSE:
                        vals.append(w[i])
            M[src_m, dst_m] = np.mean(vals) if vals else 0
    return M


def get_mouse_pose(nodes, mouse_id):
    """Extract 5 keypoints for one mouse from 15-node array."""
    start = mouse_id * KPS_PER_MOUSE
    return nodes[start:start + KPS_PER_MOUSE, :]


def get_mouse_center(pose):
    """Center of 5 keypoints (e.g. CenterBack or mean)."""
    return np.mean(pose, axis=0) / 10.0


def print_full_attn(attn_t, seq_id=0, pred_step=0):
    """Print full attention distribution for each source node (14 targets, sum=1)."""
    print('\n========== Seq {} PredStep {} Full Attention =========='.format(seq_id, pred_step))
    for src in range(15):
        if src not in attn_t:
            continue
        w, others = attn_t[src]
        assert len(w) == 14 and len(others) == 14
        total = np.sum(w)
        print('\n--- {} (node {}) -> 14 targets, sum={:.6f} ---'.format(NODE_NAMES[src], src, total))
        for i, tgt in enumerate(others):
            tgt_m = tgt // KPS_PER_MOUSE
            print('  {} -> {} (node {}): {:.6f}'.format(
                NODE_NAMES[src], NODE_NAMES[tgt], tgt, w[i]))
        # Summary by target mouse
        for dst_m in range(3):
            s = sum(w[i] for i, o in enumerate(others) if o // KPS_PER_MOUSE == dst_m)
            if dst_m != src // KPS_PER_MOUSE or s > 0:
                print('  -> M{} total: {:.6f}'.format(dst_m, s))
    print('=' * 55)


def plot_mouse_skeleton(ax, pose, color, linestyle='solid', linewidth=1.5, alpha=1.0):
    """Draw 5-point mouse skeleton. pose: (5, 2) in [0,10] scale."""
    x = pose[:, 0] / 10.0
    y = pose[:, 1] / 10.0
    for (i, j) in MOUSE_5PT_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha)
    ax.plot(x, y, 'o', color=color, markersize=3, alpha=alpha)


def plot_attention_circle(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                         name, plot_directory, seq_id=0, pred_step=0, coord_scale='mouse',
                         partial_frames=5):
    """
    按权重大小绘制圆半径: 半径=权重. 轨迹只画部分; 遮挡时每鼠只取代表点.
    改进: 画骨架、图例、箭头、权重标注，便于理解.
    """
    traj_length, numNodes, _ = true_pos.shape
    assert numNodes == 15

    def to_plot(p):
        return p / 10.0 if coord_scale == 'mouse' else (p + 1) / 2

    if pred_step >= len(attn_weights):
        pred_step = 0
    attn_t = attn_weights[pred_step]
    t_obs = observed_length - 1
    pos = true_pos[t_obs]
    obs_start = max(0, t_obs - partial_frames + 1)

    # 检查是否遮挡
    centers = [to_plot(np.mean(get_mouse_pose(pos, m), axis=0)) for m in range(3)]
    def dist(i, j):
        return np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
    occluded = any(dist(i, j) < OCCLUSION_DIST for i in range(3) for j in range(i + 1, 3))

    src_nodes = [0, 5, 10]
    if occluded:
        tgt_nodes = [m * KPS_PER_MOUSE + r for m in range(3) for r in REP_NODES_PER_MOUSE]
    else:
        tgt_nodes = list(range(15))

    for src in src_nodes:
        src_m = src // KPS_PER_MOUSE
        if src not in attn_t:
            continue
        w, others = attn_t[src]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')

        # 1. 三只鼠的5点骨架（当前帧）
        for m in range(3):
            pose = get_mouse_pose(pos, m)
            plot_mouse_skeleton(ax, pose, MICE_COLORS[m], linewidth=2)

        # 2. 部分轨迹（最近几帧，用中心点连线）
        for m in range(3):
            poses_m = [get_mouse_pose(true_pos[t], m) for t in range(obs_start, t_obs + 1)]
            xs = [to_plot(np.mean(p, axis=0))[0] for p in poses_m]
            ys = [to_plot(np.mean(p, axis=0))[1] for p in poses_m]
            ax.plot(xs, ys, color=MICE_COLORS[m], linestyle='-', linewidth=2, alpha=0.5)

        # 3. 源鼠当前位置用菱形突出
        c_src = to_plot(np.mean(get_mouse_pose(pos, src_m), axis=0))
        ax.scatter(c_src[0], c_src[1], color=MICE_COLORS[src_m], marker='D', s=120, zorder=10,
                  edgecolors='black', linewidths=2, label='Source (M{})'.format(src_m))

        # 4. 目标点画圆 + 箭头 + 权重标注
        radius_scale = 0.12
        for tgt in tgt_nodes:
            if tgt == src or tgt not in others:
                continue
            idx = others.index(tgt)
            weight = w[idx]
            pt = to_plot(pos[tgt])
            # 圆: 半径=权重
            circle = plt.Circle((pt[0], pt[1]), weight * radius_scale, fill=False,
                               color=MICE_COLORS[src_m], linewidth=2, alpha=0.9)
            ax.add_artist(circle)
            # 箭头: 从源到目标，线宽=权重
            lw = 1 + 4 * weight
            ax.annotate('', xy=(pt[0], pt[1]), xytext=(c_src[0], c_src[1]),
                       arrowprops=dict(arrowstyle='->', color=MICE_COLORS[src_m],
                                     lw=lw, alpha=0.6))
            # 权重数值
            ax.text(pt[0] + 0.02, pt[1], '{:.2f}'.format(weight), fontsize=8,
                   color=MICE_COLORS[src_m], fontweight='bold')

        # 5. 图例
        legend_elements = [
            Line2D([0], [0], color=MICE_COLORS[0], lw=2, label='M0'),
            Line2D([0], [0], color=MICE_COLORS[1], lw=2, label='M1'),
            Line2D([0], [0], color=MICE_COLORS[2], lw=2, label='M2'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Seq {} | M{} attends to others | circle radius = weight (sum=1)'.format(
            seq_id, src_m), fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(plot_directory, '{}_node{}_pred{}.png'.format(name, src, pred_step))
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_attention_full(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                       name, plot_directory, seq_id=0, pred_step=0, coord_scale='mouse',
                       partial_frames=5):
    """完整可视化: 始终显示全部14个目标, 权重保留4位小数."""
    traj_length, numNodes, _ = true_pos.shape
    assert numNodes == 15

    def to_plot(p):
        return p / 10.0 if coord_scale == 'mouse' else (p + 1) / 2

    if pred_step >= len(attn_weights):
        pred_step = 0
    attn_t = attn_weights[pred_step]
    t_obs = observed_length - 1
    pos = true_pos[t_obs]
    obs_start = max(0, t_obs - partial_frames + 1)

    # 始终使用全部14个目标(排除自身)
    src_nodes = [0, 5, 10]
    for src in src_nodes:
        src_m = src // KPS_PER_MOUSE
        if src not in attn_t:
            continue
        w, others = attn_t[src]
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')

        for m in range(3):
            pose = get_mouse_pose(pos, m)
            plot_mouse_skeleton(ax, pose, MICE_COLORS[m], linewidth=2)
        for m in range(3):
            poses_m = [get_mouse_pose(true_pos[t], m) for t in range(obs_start, t_obs + 1)]
            xs = [to_plot(np.mean(p, axis=0))[0] for p in poses_m]
            ys = [to_plot(np.mean(p, axis=0))[1] for p in poses_m]
            ax.plot(xs, ys, color=MICE_COLORS[m], linestyle='-', linewidth=2, alpha=0.5)

        c_src = to_plot(np.mean(get_mouse_pose(pos, src_m), axis=0))
        ax.scatter(c_src[0], c_src[1], color=MICE_COLORS[src_m], marker='D', s=120, zorder=10,
                  edgecolors='black', linewidths=2)

        radius_scale = 0.12
        for i, tgt in enumerate(others):
            weight = w[i]
            pt = to_plot(pos[tgt])
            circle = plt.Circle((pt[0], pt[1]), weight * radius_scale, fill=False,
                               color=MICE_COLORS[src_m], linewidth=2, alpha=0.9)
            ax.add_artist(circle)
            lw = 1 + 4 * weight
            ax.annotate('', xy=(pt[0], pt[1]), xytext=(c_src[0], c_src[1]),
                       arrowprops=dict(arrowstyle='->', color=MICE_COLORS[src_m], lw=lw, alpha=0.6))
            ax.text(pt[0] + 0.02, pt[1], '{:.4f}'.format(weight), fontsize=7,
                   color=MICE_COLORS[src_m], fontweight='bold')

        legend_elements = [Line2D([0], [0], color=MICE_COLORS[0], lw=2, label='M0'),
                          Line2D([0], [0], color=MICE_COLORS[1], lw=2, label='M1'),
                          Line2D([0], [0], color=MICE_COLORS[2], lw=2, label='M2')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Seq {} | M{} full attn (14 targets, sum={:.4f})'.format(
            seq_id, src_m, float(np.sum(w))), fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(plot_directory, '{}_node{}_pred{}_full.png'.format(name, src, pred_step))
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_attention_per_point(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                             name, plot_directory, seq_id=0, pred_step=0, coord_scale='mouse',
                             partial_frames=5):
    """每个点单独绘制: 每个源节点生成14个子图, 每个子图只显示一个(src->tgt)连接."""
    traj_length, numNodes, _ = true_pos.shape
    assert numNodes == 15

    def to_plot(p):
        return p / 10.0 if coord_scale == 'mouse' else (p + 1) / 2

    if pred_step >= len(attn_weights):
        pred_step = 0
    attn_t = attn_weights[pred_step]
    t_obs = observed_length - 1
    pos = true_pos[t_obs]
    obs_start = max(0, t_obs - partial_frames + 1)

    src_nodes = [0, 5, 10]
    for src in src_nodes:
        src_m = src // KPS_PER_MOUSE
        if src not in attn_t:
            continue
        w, others = attn_t[src]
        fig, axes = plt.subplots(2, 7, figsize=(21, 6))
        axes = axes.flatten()

        for idx, (tgt, weight) in enumerate(zip(others, w)):
            ax = axes[idx]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect('equal')

            for m in range(3):
                pose = get_mouse_pose(pos, m)
                plot_mouse_skeleton(ax, pose, MICE_COLORS[m], linewidth=1, alpha=0.5)
            for m in range(3):
                poses_m = [get_mouse_pose(true_pos[t], m) for t in range(obs_start, t_obs + 1)]
                xs = [to_plot(np.mean(p, axis=0))[0] for p in poses_m]
                ys = [to_plot(np.mean(p, axis=0))[1] for p in poses_m]
                ax.plot(xs, ys, color=MICE_COLORS[m], linestyle='-', linewidth=1, alpha=0.4)

            c_src = to_plot(np.mean(get_mouse_pose(pos, src_m), axis=0))
            pt = to_plot(pos[tgt])
            ax.scatter(c_src[0], c_src[1], color=MICE_COLORS[src_m], marker='D', s=60, zorder=10,
                      edgecolors='black', linewidths=1)
            ax.plot(pt[0], pt[1], 'o', color=MICE_COLORS[tgt // KPS_PER_MOUSE], markersize=8, zorder=9)
            circle = plt.Circle((pt[0], pt[1]), weight * 0.12, fill=False,
                               color=MICE_COLORS[src_m], linewidth=2, alpha=0.9)
            ax.add_artist(circle)
            ax.annotate('', xy=(pt[0], pt[1]), xytext=(c_src[0], c_src[1]),
                       arrowprops=dict(arrowstyle='->', color=MICE_COLORS[src_m], lw=2, alpha=0.8))
            ax.set_title('{}->{}: {:.4f}'.format(NODE_NAMES[src], NODE_NAMES[tgt], weight), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle('Seq {} | M{} per-point attention (14 targets)'.format(seq_id, src_m), fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(plot_directory, '{}_node{}_pred{}_perpoint.png'.format(name, src, pred_step))
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()


def plot_attention_mouse(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                        name, plot_directory, seq_id=0, pred_step=0, coord_scale='mouse'):
    """
    Plot: (1) Trajectory + skeleton + attention arrows  (2) 3×3 mouse attention heatmap.
    true_pos, pred_pos: (seq_len, 15, 2). Mouse coords in [0,10].
    """
    traj_length, numNodes, _ = true_pos.shape
    assert numNodes == 15

    # Normalize coords for plotting
    def to_plot(p):
        if coord_scale == 'mouse':
            return p / 10.0
        return (p + 1) / 2

    # Get attention for selected pred step
    if pred_step >= len(attn_weights):
        pred_step = 0
    attn_t = attn_weights[pred_step]
    M = get_mouse_attn_matrix(attn_t)

    # Frame to visualize: last observed (for skeleton) + first pred (for attention context)
    t_obs = observed_length - 1
    t_pred = observed_length + pred_step
    pos_obs = true_pos[t_obs]
    pos_pred_true = true_pos[t_pred] if t_pred < traj_length else pos_obs
    pos_pred_pred = pred_pos[t_pred] if t_pred < traj_length else pos_obs

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.2, 0.8]})

    # ---- Left: Trajectory + skeleton + attention arrows ----
    ax_traj = axes[0]
    ax_traj.set_xlim(0, 1)
    ax_traj.set_ylim(1, 0)
    ax_traj.set_aspect('equal')
    ax_traj.axis('off')

    # Observed trajectory: draw path for each mouse (last few frames for clarity)
    obs_start = max(0, t_obs - 4)
    for m in range(3):
        color = MICE_COLORS[m]
        poses = [get_mouse_pose(true_pos[t], m) for t in range(obs_start, t_obs + 1)]
        xs = [to_plot(get_mouse_center(p))[0] for p in poses]
        ys = [to_plot(get_mouse_center(p))[1] for p in poses]
        ax_traj.plot(xs, ys, color=color, linestyle='-', linewidth=2, alpha=0.6)

    # Current frame: 3 mice as skeletons (observed positions)
    for m in range(3):
        pose = get_mouse_pose(pos_obs, m)
        plot_mouse_skeleton(ax_traj, pose, MICE_COLORS[m], linewidth=2)

    # Attention arrows: from source mouse center to target mouse center
    # Linewidth 2–12 by attention weight (normalize M to [0,1] for display)
    M_max = M.max() if M.max() > 0 else 1
    for src in range(3):
        for dst in range(3):
            if src == dst or M[src, dst] <= 0:
                continue
            c_src = get_mouse_center(get_mouse_pose(pos_obs, src))
            c_dst = get_mouse_center(get_mouse_pose(pos_obs, dst))
            c_src = to_plot(c_src)
            c_dst = to_plot(c_dst)
            w = M[src, dst] / M_max
            lw = 2 + 8 * w
            ax_traj.annotate('', xy=(c_dst[0], c_dst[1]), xytext=(c_src[0], c_src[1]),
                            arrowprops=dict(arrowstyle='->', color=MICE_COLORS[src],
                                          lw=lw, alpha=0.7 + 0.2 * w))
    ax_traj.set_title('Seq {} | Frame {} | Attention (arrows: source->target)'.format(seq_id, t_pred), fontsize=11)

    # ---- Right: 3×3 heatmap ----
    ax_hm = axes[1]
    labels = ['M0', 'M1', 'M2']
    sns.heatmap(M, ax=ax_hm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=M_max if M_max > 0 else 0.2,
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Attention'})
    ax_hm.set_xlabel('Target mouse')
    ax_hm.set_ylabel('Source mouse')
    ax_hm.set_title('Mouse-level attention (row sum!=1: cross-mouse mean)', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(plot_directory, '{}_pred{}.png'.format(name, pred_step))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_multi_step(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                              name, plot_directory, seq_id=0, pred_steps=(0, 5, 9), coord_scale='mouse'):
    """Plot trajectory + multiple heatmaps for different pred steps (attention evolution)."""
    traj_length, numNodes, _ = true_pos.shape
    assert numNodes == 15

    def to_plot(p):
        return p / 10.0 if coord_scale == 'mouse' else (p + 1) / 2

    n_steps = len(pred_steps)
    fig, axes = plt.subplots(2, n_steps + 1, figsize=(4 * (n_steps + 1), 8),
                            gridspec_kw={'height_ratios': [1, 1]})

    # Row 0: trajectory + skeleton for first step; then heatmaps
    ax_traj = axes[0, 0]
    ax_traj.set_xlim(0, 1)
    ax_traj.set_ylim(1, 0)
    ax_traj.set_aspect('equal')
    ax_traj.axis('off')
    t_obs = observed_length - 1
    pos_obs = true_pos[t_obs]
    obs_start = max(0, t_obs - 4)
    for m in range(3):
        color = MICE_COLORS[m]
        poses = [get_mouse_pose(true_pos[t], m) for t in range(obs_start, t_obs + 1)]
        xs = [to_plot(get_mouse_center(p))[0] for p in poses]
        ys = [to_plot(get_mouse_center(p))[1] for p in poses]
        ax_traj.plot(xs, ys, color=color, linestyle='-', linewidth=2, alpha=0.6)
    for m in range(3):
        plot_mouse_skeleton(ax_traj, get_mouse_pose(pos_obs, m), MICE_COLORS[m], linewidth=2)
    ax_traj.set_title('Seq {} | Trajectory'.format(seq_id), fontsize=10)

    vmax_global = 0.1
    for pred_step in pred_steps:
        if pred_step < len(attn_weights):
            M = get_mouse_attn_matrix(attn_weights[pred_step])
            if M.max() > 0:
                vmax_global = max(vmax_global, M.max())
    for col, pred_step in enumerate(pred_steps):
        if pred_step >= len(attn_weights):
            continue
        M = get_mouse_attn_matrix(attn_weights[pred_step])
        ax_hm = axes[0, col + 1]
        sns.heatmap(M, ax=ax_hm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=vmax_global,
                    xticklabels=['M0','M1','M2'], yticklabels=['M0','M1','M2'])
        ax_hm.set_title('Pred step {}'.format(pred_step), fontsize=10)

    # Row 1: attention arrows for first step
    ax_arrows = axes[1, 0]
    ax_arrows.set_xlim(0, 1)
    ax_arrows.set_ylim(1, 0)
    ax_arrows.set_aspect('equal')
    ax_arrows.axis('off')
    if pred_steps[0] < len(attn_weights):
        M = get_mouse_attn_matrix(attn_weights[pred_steps[0]])
        M_max = M.max() if M.max() > 0 else 1
        for src in range(3):
            for dst in range(3):
                if src == dst or M[src, dst] <= 0:
                    continue
                c_src = to_plot(get_mouse_center(get_mouse_pose(pos_obs, src)))
                c_dst = to_plot(get_mouse_center(get_mouse_pose(pos_obs, dst)))
                w = M[src, dst] / M_max
                lw = 2 + 8 * w
                ax_arrows.annotate('', xy=(c_dst[0], c_dst[1]), xytext=(c_src[0], c_src[1]),
                                  arrowprops=dict(arrowstyle='->', color=MICE_COLORS[src], lw=lw, alpha=0.7 + 0.2 * w))
        for m in range(3):
            plot_mouse_skeleton(ax_arrows, get_mouse_pose(pos_obs, m), MICE_COLORS[m], linewidth=2)
    ax_arrows.set_title('Attention arrows (step 0)', fontsize=10)

    for col in range(1, n_steps + 1):
        axes[1, col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(plot_directory, '{}_multi.png'.format(name))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=int, default=0)
    parser.add_argument('--use_mouse_data', action='store_true', help='Use mouse dataset')
    parser.add_argument('--exp_tag', type=str, default='obs15_pred10_fs2')
    parser.add_argument('--from_distributed', action='store_true')
    parser.add_argument('--sequence', type=int, default=None, help='Only plot this sequence (default: all)')
    parser.add_argument('--pred_step', type=int, default=0, help='Which prediction step to visualize')
    parser.add_argument('--pred_steps', type=str, default=None,
                        help='Comma-separated pred steps for multi-step figure (e.g. 0,5,9)')
    parser.add_argument('--viz_style', type=str, default='circle',
                        choices=['circle', 'heatmap', 'full', 'per_point'],
                        help='circle/heatmap/full(14 targets, 4 decimals)/per_point(14 subplots per src)')
    parser.add_argument('--partial_frames', type=int, default=5,
                        help='轨迹只画最近几帧 (circle/full/per_point)')
    parser.add_argument('--print_attn', action='store_true',
                        help='Print full attention distribution to console')
    args = parser.parse_args()

    if args.use_mouse_data:
        subdir = 'save_attention' if args.from_distributed else 'save_attention_single'
        save_directory = os.path.join('save', 'mice', args.exp_tag, subdir)
        plot_directory = os.path.join('plot', 'mice', args.exp_tag, 'plot_attention_viz')
    else:
        save_directory = os.path.join('save', str(args.test_dataset), 'save_attention')
        plot_directory = os.path.join('plot', 'plot_attention_viz', str(args.test_dataset))

    os.makedirs(plot_directory, exist_ok=True)

    with open(os.path.join(save_directory, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    pred_steps = None
    if args.pred_steps:
        pred_steps = [int(x.strip()) for x in args.pred_steps.split(',')]

    indices = [args.sequence] if args.sequence is not None else range(len(results))
    for i in indices:
        if i >= len(results):
            print('Skipping sequence {} (max {})'.format(i, len(results)-1))
            continue
        print('Plotting sequence {}'.format(i))
        true_pos = np.array(results[i][0])
        pred_pos = np.array(results[i][1])
        nodes_present = results[i][2]
        observed_length = int(results[i][3])
        attn_weights = results[i][4]
        coord_scale = 'mouse' if args.use_mouse_data else 'human'

        if args.print_attn:
            pred_step = min(args.pred_step, len(attn_weights) - 1) if attn_weights else 0
            if attn_weights:
                print_full_attn(attn_weights[pred_step], seq_id=i, pred_step=pred_step)

        if pred_steps:
            plot_attention_multi_step(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                                    'sequence{}'.format(i), plot_directory, seq_id=i, pred_steps=pred_steps,
                                    coord_scale=coord_scale)
        elif args.viz_style == 'circle':
            plot_attention_circle(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                                 'sequence{}'.format(i), plot_directory, seq_id=i, pred_step=args.pred_step,
                                 coord_scale=coord_scale, partial_frames=args.partial_frames)
        elif args.viz_style == 'full':
            plot_attention_full(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                               'sequence{}'.format(i), plot_directory, seq_id=i, pred_step=args.pred_step,
                               coord_scale=coord_scale, partial_frames=args.partial_frames)
        elif args.viz_style == 'per_point':
            plot_attention_per_point(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                                    'sequence{}'.format(i), plot_directory, seq_id=i, pred_step=args.pred_step,
                                    coord_scale=coord_scale, partial_frames=args.partial_frames)
        else:
            plot_attention_mouse(true_pos, pred_pos, nodes_present, observed_length, attn_weights,
                                'sequence{}'.format(i), plot_directory, seq_id=i, pred_step=args.pred_step,
                                coord_scale=coord_scale)
    print('Saved to {}'.format(plot_directory))


if __name__ == '__main__':
    main()
