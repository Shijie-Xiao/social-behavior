'''
Video visualization for mouse trajectory prediction.
Draws one sequence with 5 keypoints per mouse connected as skeleton.
Observation: solid lines. Prediction: true=solid, pred=dashed.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import argparse
import os

# 5 keypoints per mouse: Nose(0), L-Ear(1), R-Ear(2), CenterBack(3), TailBase(4)
# Skeleton connections: head triangle + ears to spine + spine to tail
MOUSE_5PT_CONNECTIONS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]

MICE_COLORS = ('#2ecc71', '#3498db', '#e74c3c')  # green, blue, red
KPS_PER_MOUSE = 5


def get_mouse_pose(nodes, mouse_id):
    """Extract 5 keypoints for one mouse from 15-node array."""
    start = mouse_id * KPS_PER_MOUSE
    return nodes[start:start + KPS_PER_MOUSE, :]


def plot_mouse_skeleton(ax, pose, color, linestyle='solid', linewidth=2, alpha=1.0):
    """Draw 5-point mouse skeleton. pose: (5, 2) in [0,10] scale."""
    x = pose[:, 0] / 10.0
    y = pose[:, 1] / 10.0
    for (i, j) in MOUSE_5PT_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha)
    ax.plot(x, y, 'o', color=color, markersize=4, alpha=alpha)


def create_video(true_trajs, pred_trajs, nodesPresent, obs_length, out_path,
                 seq_id=0, fps=5):
    """
    Create video: obs period = true only (solid); pred period = true (solid) + pred (dashed).
    true_trajs, pred_trajs: (seq_length, 15, 2) in [0,10] scale
    """
    seq_length, num_nodes, _ = true_trajs.shape
    assert num_nodes == 15

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # image coords
    ax.set_aspect('equal')
    ax.axis('off')

    def init():
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        return []

    def animate_frame(t):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect('equal')
        ax.axis('off')

        present = nodesPresent[t] if t < len(nodesPresent) else nodesPresent[-1]
        true_pos = true_trajs[t]
        pred_pos = pred_trajs[t]

        for mouse_id in range(3):
            color = MICE_COLORS[mouse_id]
            base = mouse_id * KPS_PER_MOUSE
            if base not in present and base + 1 not in present:
                continue

            pose_true = get_mouse_pose(true_pos, mouse_id)
            pose_pred = get_mouse_pose(pred_pos, mouse_id)

            if t < obs_length:
                plot_mouse_skeleton(ax, pose_true, color, linestyle='solid', linewidth=2)
            else:
                plot_mouse_skeleton(ax, pose_true, color, linestyle='solid', linewidth=2, alpha=0.9)
                plot_mouse_skeleton(ax, pose_pred, color, linestyle='--', linewidth=1.5, alpha=0.7)

        phase = 'Observed' if t < obs_length else 'Predicted'
        ax.set_title(f'Sequence {seq_id} | Frame {t} | {phase}', fontsize=12)
        return []

    ani = animation.FuncAnimation(fig, animate_frame, frames=seq_length,
                                init_func=init, blit=False, interval=1000//fps)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    try:
        ani.save(out_path, writer='ffmpeg', fps=fps)
        print('Saved:', out_path)
    except Exception as e:
        print('ffmpeg failed:', e)
        out_path_gif = out_path.replace('.mp4', '.gif')
        ani.save(out_path_gif, writer='pillow', fps=fps)
        print('Saved:', out_path_gif)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=int, default=0, help='Sequence index to visualize')
    parser.add_argument('--exp_tag', type=str, default='obs15_pred10_fs2')
    parser.add_argument('--from_distributed', action='store_true')
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()

    subdir = 'save_attention' if args.from_distributed else 'save_attention_single'
    save_dir = os.path.join('save', 'mice', args.exp_tag, subdir)
    plot_dir = os.path.join('plot', 'mice', args.exp_tag, 'plot_video')
    os.makedirs(plot_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)

    if args.sequence >= len(results):
        print(f'Sequence {args.sequence} not found. Max index: {len(results)-1}')
        return

    true_trajs, pred_trajs, nodesPresent, obs_length = results[args.sequence][:4]
    true_trajs = np.array(true_trajs)
    pred_trajs = np.array(pred_trajs)
    # nodesPresent: list of lists (seq_length), each list = node IDs present
    if hasattr(nodesPresent[0], 'tolist'):
        nodesPresent = [list(p.tolist()) if hasattr(p, 'tolist') else list(p) for p in nodesPresent]
    obs_length = int(obs_length)

    out_path = os.path.join(plot_dir, f'sequence{args.sequence}_video.mp4')
    create_video(true_trajs, pred_trajs, nodesPresent, obs_length, out_path,
                 seq_id=args.sequence, fps=args.fps)


if __name__ == '__main__':
    main()
