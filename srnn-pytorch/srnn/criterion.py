"""
Loss functions for mouse trajectory prediction.

L_total = w(activity) * L_trajectory  +  lambda * L_dist

- L_trajectory: bivariate Gaussian NLL (per-node, per-frame)
- L_dist:       relative pairwise distance loss — compares predicted vs GT
                intra-mouse keypoint distances.  Disabled when n_kps=1.
- w(activity):  per-window scalar that down-weights static windows
"""

import torch
import numpy as np

N_MICE = 3

_BONE_CACHE = {}


def compute_bone_stats(train_nodes, n_keypoints=4):
    """
    Compute pairwise distance statistics (for logging) and cache
    the intra-mouse pair indices used by body_distance_loss().

    For n_keypoints=1, no pairs exist → distance loss is disabled.

    Parameters
    ----------
    train_nodes : np.ndarray  (N, T, ..., 2)
    n_keypoints : int
    """
    _BONE_CACHE.clear()
    _BONE_CACHE["n_kps"] = n_keypoints

    if n_keypoints <= 1:
        _BONE_CACHE["n_pairs"] = 0
        _BONE_CACHE["_device_cache"] = {}
        print(f"  n_keypoints={n_keypoints}: no intra-mouse pairs, "
              f"distance loss disabled")
        return

    N = train_nodes.shape[0]
    T = train_nodes.shape[1]
    n_nodes = N_MICE * n_keypoints
    data = train_nodes.reshape(N, T, N_MICE, n_keypoints, 2)

    for ki in range(n_keypoints):
        for kj in range(ki + 1, n_keypoints):
            dists = []
            for m in range(N_MICE):
                d = np.linalg.norm(data[:, :, m, ki, :] - data[:, :, m, kj, :],
                                   axis=-1)
                dists.append(d.ravel())
            d_all = np.concatenate(dists)
            print(f"  pair ({ki}→{kj}): mean={d_all.mean():.5f}  "
                  f"std={d_all.std():.5f}")

    idx_i, idx_j = [], []
    for m in range(N_MICE):
        offset = m * n_keypoints
        for ki in range(n_keypoints):
            for kj in range(ki + 1, n_keypoints):
                idx_i.append(offset + ki)
                idx_j.append(offset + kj)

    _BONE_CACHE["idx_i"] = np.array(idx_i, dtype=np.int64)
    _BONE_CACHE["idx_j"] = np.array(idx_j, dtype=np.int64)
    _BONE_CACHE["n_pairs"] = len(idx_i)
    _BONE_CACHE["_device_cache"] = {}
    print(f"  Total intra-mouse pairs: {len(idx_i)}")


def _get_pair_indices(device):
    dc = _BONE_CACHE["_device_cache"]
    if device not in dc:
        dc[device] = (
            torch.tensor(_BONE_CACHE["idx_i"], dtype=torch.long, device=device),
            torch.tensor(_BONE_CACHE["idx_j"], dtype=torch.long, device=device),
        )
    return dc[device]


def gaussian_2d_nll(outputs, targets, pred_length):
    """
    Bivariate Gaussian NLL loss (batch-parallel, numerically stable).

    output[t] predicts target[t+1].
    """
    seq_len = outputs.size(1)
    obs_len = seq_len - pred_length

    out = outputs[:, obs_len - 1: -1, :, :]
    tgt = targets[:, obs_len:, :, :]

    mux = out[..., 0]
    muy = out[..., 1]
    sx = torch.exp(torch.clamp(out[..., 2], min=-6, max=6))
    sy = torch.exp(torch.clamp(out[..., 3], min=-6, max=6))
    corr = torch.tanh(out[..., 4])

    normx = tgt[..., 0] - mux
    normy = tgt[..., 1] - muy

    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * corr * normx * normy / sxsy
    z = z.clamp(min=0.0)
    neg_rho = (1.0 - corr ** 2).clamp(min=1e-4)

    log_prob = (
        -0.5 * z / neg_rho
        - torch.log(sxsy + 1e-8)
        - 0.5 * torch.log(neg_rho)
        - np.log(2 * np.pi)
    )

    nll = -log_prob.mean(dim=(1, 2))
    return nll


def body_distance_loss(pred_pos, gt_pos):
    """
    Relative pairwise distance loss for all C(n_kps,2) intra-mouse
    keypoint pairs × 3 mice.

    Returns zero tensor if n_kps <= 1 (no pairs).
    """
    if _BONE_CACHE.get("n_pairs", 0) == 0:
        return torch.zeros(pred_pos.size(0), device=pred_pos.device)

    idx_i, idx_j = _get_pair_indices(pred_pos.device)

    d_pred = torch.norm(pred_pos[:, :, idx_i, :] - pred_pos[:, :, idx_j, :],
                        dim=-1)
    d_gt = torch.norm(gt_pos[:, :, idx_i, :] - gt_pos[:, :, idx_j, :],
                      dim=-1)

    loss = ((d_pred - d_gt) / (d_gt + 0.01)) ** 2
    return loss.mean(dim=(1, 2))


def activity_weight(activity, tau=None, w_min=0.1):
    if tau is None:
        tau = activity.median().clamp(min=1e-6)
    w = (activity / tau).clamp(min=w_min, max=1.0)
    return w


def combined_loss(outputs, nodes, activity, pred_length, lambda_dist=0.1):
    """
    Full training loss.
    """
    B = outputs.size(0)
    seq_len = outputs.size(1)
    obs_len = seq_len - pred_length

    nll = gaussian_2d_nll(outputs, nodes, pred_length)

    w = activity_weight(activity)
    weighted_nll = (w * nll).mean()

    pred_params = outputs[:, obs_len - 1: -1, :, :]
    pred_pos = pred_params[..., :2]
    gt_pos = nodes[:, obs_len:, :, :]

    d_loss = body_distance_loss(pred_pos, gt_pos).mean()

    total = weighted_nll + lambda_dist * d_loss

    loss_dict = {
        "total": total.item(),
        "nll": nll.mean().item(),
        "nll_weighted": weighted_nll.item(),
        "dist": d_loss.item(),
        "w_mean": w.mean().item(),
    }
    return total, loss_dict
