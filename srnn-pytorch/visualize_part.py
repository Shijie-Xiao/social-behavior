'''
Visualization script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Modified by: ChatGPT
Date: 2025
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os


def plot_trajectories(true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory, selected_nodes=None):
    '''
    Parameters
    ----------
    true_trajs : np.array of shape (seq_length, numNodes, 2)
        True trajectories of nodes.
    pred_trajs : np.array of shape (seq_length, numNodes, 2)
        Predicted trajectories of nodes.
    nodesPresent : list of lists
        Nodes present at each timestep.
    obs_length : int
        Length of observed trajectory.
    name : str
        Name of the plot.
    plot_directory : str
        Directory to save plots.
    selected_nodes : list, optional
        List of specific nodes to visualize.
    '''

    traj_length, numNodes, _ = true_trajs.shape

    # Select a subset of nodes to visualize
    if selected_nodes is None:
        selected_nodes = list(range(min(5, numNodes)))  # Default: first 5 nodes

    plt.figure(figsize=(8, 6))

    for ped in selected_nodes:
        if ped >= numNodes:
            continue

        true_x = []
        true_y = []
        pred_x = []
        pred_y = []

        for t in range(traj_length):
            if ped in nodesPresent[t]:  # Only plot if node was present
                true_x.append(true_trajs[t, ped, 0])
                true_y.append(true_trajs[t, ped, 1])
                pred_x.append(pred_trajs[t, ped, 0])
                pred_y.append(pred_trajs[t, ped, 1])

        if len(true_x) > 0:
            plt.plot(true_x, true_y, linestyle='solid', marker='o', label=f'True {ped}')
            plt.plot(pred_x, pred_y, linestyle='dashed', marker='+', label=f'Pred {ped}')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.title(f'Trajectory Visualization: {name}')
    plt.xlim(-2, 2)  # Adjust as needed
    plt.ylim(-2, 2)

    # Save and show the figure
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    save_path = os.path.join(plot_directory, name + '.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=int, default=3, help='test dataset index')
    parser.add_argument('--use_mouse_data', action='store_true',
                        help='Use mouse dataset results')
    parser.add_argument('--exp_tag', type=str, default='obs15_pred5_fs4',
                        help='Experiment tag; must match train/sample')
    parser.add_argument('--from_distributed', action='store_true',
                        help='Load from train.py (save_attention) instead of train_single')

    args = parser.parse_args()

    if args.use_mouse_data:
        subdir = 'save_attention' if args.from_distributed else 'save_attention_single'
        save_directory = os.path.join('save', 'mice', args.exp_tag, subdir)
        plot_directory = os.path.join('plot', 'mice', args.exp_tag, 'plot_attention_part')
    else:
        save_directory = f'./save/{args.test_dataset}/save_attention/'
        plot_directory = f'plot/plot_attention_part/'

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Load results
    try:
        with open(os.path.join(save_directory, 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: results.pkl not found in {save_directory}. Run sample.py first.")
        return

    # Select nodes to visualize (modify as needed)
    selected_nodes = [0, 1, 2, 3, 4]

    for i, (true_trajs, pred_trajs, nodesPresent, obs_length, _, _) in enumerate(results):
        print(f"Processing sequence {i}")
        name = f'sequence_{i}'
        plot_trajectories(true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory, selected_nodes)


if __name__ == '__main__':
    main()
