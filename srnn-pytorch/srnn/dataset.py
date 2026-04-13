"""
PyTorch Dataset / DataLoader for preprocessed mouse windowed data (.npz).

Each window has shape (win, 3, 4, 2) = (time, mice, keypoints, xy).
The DataLoader reshapes this into nodes (win, N_nodes, 2) where N_nodes
depends on the selected keypoints.

Keypoint indices:
  0 = nose, 1 = neck, 2 = center_back, 3 = tail_base
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


N_MICE = 3
ALL_KPS = 4
KP_PRESETS = {
    1: [2],        # center_back only
    2: [0, 2],     # nose + center_back
    3: [0, 1, 2],  # nose + neck + center_back
    4: [0, 1, 2, 3],  # all
}
# center_back is always original kp index 2
CB_ORIGINAL_IDX = 2


def get_center_back_node_indices(n_keypoints):
    """
    Return node indices (in the flattened N_MICE*n_kps layout) that
    correspond to center_back for each of the 3 mice.

    Examples
    --------
    n_kps=1 → [0, 1, 2]         (each node IS center_back)
    n_kps=2 → [1, 3, 5]         (preset [0,2], cb at local idx 1)
    n_kps=4 → [2, 6, 10]        (preset [0,1,2,3], cb at local idx 2)
    """
    kp_list = KP_PRESETS[n_keypoints]
    local_idx = kp_list.index(CB_ORIGINAL_IDX)
    return [m * n_keypoints + local_idx for m in range(N_MICE)]


class MouseWindowDataset(Dataset):
    def __init__(self, npz_path, split="train", obs_length=10,
                 n_keypoints=4):
        data = np.load(npz_path)
        raw = data[f"{split}_data"].astype(np.float32)  # (N, W, 3, 4, 2)
        N, T = raw.shape[0], raw.shape[1]
        raw5d = raw.reshape(N, T, N_MICE, ALL_KPS, 2)

        kp_idx = KP_PRESETS[n_keypoints]
        selected = raw5d[:, :, :, kp_idx, :]  # (N, T, 3, n_kps, 2)
        n_nodes = N_MICE * n_keypoints
        self.nodes = selected.reshape(N, T, n_nodes, 2)

        self.nodes_full = raw.reshape(N, T, N_MICE * ALL_KPS, 2)

        self.lights = data[f"{split}_lights"]
        self.chase = data[f"{split}_chase"]
        self.activity = data[f"{split}_activity"].astype(np.float32)
        self.obs_length = obs_length
        self.n_keypoints = n_keypoints

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.nodes[idx]),
            torch.tensor(self.activity[idx], dtype=torch.float32),
            torch.tensor(self.lights[idx], dtype=torch.long),
            torch.tensor(self.chase[idx], dtype=torch.long),
        )


def get_mouse_dataloaders(npz_path, obs_length=10, batch_size=64,
                          num_workers=4, n_keypoints=4):
    loaders = {}
    for split in ("train", "val", "test"):
        ds = MouseWindowDataset(npz_path, split=split,
                                obs_length=obs_length,
                                n_keypoints=n_keypoints)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=shuffle,
        )
    return loaders
