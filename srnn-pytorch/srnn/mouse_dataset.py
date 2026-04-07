"""
PyTorch Dataset / DataLoader for preprocessed mouse windowed data (.npz).

Each window has shape (win, 3, 4, 2) = (time, mice, keypoints, xy).
The DataLoader reshapes this into nodes (win, 12, 2).
Edge features are computed on GPU in the model (build_edges_from_nodes).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


N_MICE = 3
N_KPS = 4
N_NODES = N_MICE * N_KPS  # 12


class MouseWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping a preprocessed .npz split.
    Returns only nodes (no edge pre-computation for speed).
    """

    def __init__(self, npz_path, split="train", obs_length=10):
        data = np.load(npz_path)
        raw = data[f"{split}_data"].astype(np.float32)   # (N, W, 3, 4, 2)
        T = raw.shape[1]
        self.nodes = raw.reshape(len(raw), T, N_NODES, 2)  # (N, W, 12, 2)
        self.lights = data[f"{split}_lights"]
        self.chase = data[f"{split}_chase"]
        self.activity = data[f"{split}_activity"].astype(np.float32)
        self.obs_length = obs_length

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.nodes[idx]),           # (W, 12, 2)
            torch.tensor(self.activity[idx], dtype=torch.float32),
            torch.tensor(self.lights[idx], dtype=torch.long),
            torch.tensor(self.chase[idx], dtype=torch.long),
        )


def get_mouse_dataloaders(npz_path, obs_length=10, batch_size=64,
                          num_workers=4):
    """
    Convenience function returning train / val / test DataLoaders.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = MouseWindowDataset(npz_path, split=split,
                                obs_length=obs_length)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=shuffle,
        )
    return loaders
