"""
Loss functions for mouse trajectory prediction.

L_total = w(activity) * L_trajectory  +  lambda * L_distance

- L_trajectory: bivariate Gaussian NLL (per-node, per-frame)
- L_distance:   relative pairwise distance MSE across all 66 node pairs
- w(activity):  per-window scalar that down-weights static windows
"""

import torch
import numpy as np

N_NODES = 12
_UPPER_TRI = None  # lazily built mask


def _get_upper_tri_mask(n, device):
    """Return (n, n) bool mask for upper triangle (excluding diagonal)."""
    global _UPPER_TRI
    if _UPPER_TRI is None or _UPPER_TRI.device != device:
        _UPPER_TRI = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1
        )
    return _UPPER_TRI


def gaussian_2d_nll(outputs, targets, pred_length):
    """
    Bivariate Gaussian NLL loss (batch-parallel, numerically stable).

    Following the original socialAttention design: output at time t
    predicts the position at time t+1.  So we align:
        output[:, obs-1 : obs+pred-1]  vs  target[:, obs : obs+pred]

    Parameters
    ----------
    outputs : (B, seq_len, N, 5)
        Predicted parameters: mux, muy, log_sx, log_sy, raw_corr
    targets : (B, seq_len, N, 2)
        Ground truth (x, y)
    pred_length : int
        Number of future frames to predict.

    Returns
    -------
    nll : (B,)  per-sample NLL averaged over pred frames and nodes
    """
    seq_len = outputs.size(1)
    obs_len = seq_len - pred_length

    # output[t] predicts target[t+1]
    out = outputs[:, obs_len - 1: -1, :, :]  # (B, pred, N, 5)
    tgt = targets[:, obs_len:, :, :]          # (B, pred, N, 2)

    mux = out[..., 0]
    muy = out[..., 1]
    sx = torch.exp(torch.clamp(out[..., 2], min=-6, max=6))
    sy = torch.exp(torch.clamp(out[..., 3], min=-6, max=6))
    corr = torch.tanh(out[..., 4])

    normx = tgt[..., 0] - mux
    normy = tgt[..., 1] - muy

    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * corr * normx * normy / sxsy
    neg_rho = 1.0 - corr ** 2

    log_prob = (
        -0.5 * z / (neg_rho + 1e-6)
        - torch.log(sxsy + 1e-8)
        - 0.5 * torch.log(neg_rho + 1e-6)
        - np.log(2 * np.pi)
    )

    nll = -log_prob.mean(dim=(1, 2))  # mean over pred_frames, nodes → (B,)
    return nll


def distance_loss(pred_pos, gt_pos):
    """
    Relative pairwise distance loss across all C(12,2)=66 node pairs.

    L = mean_{pairs} [ (d_pred - d_gt)^2 / (d_gt + eps)^2 ]

    eps is set to 0.01 (≈4.5 px) — small enough to preserve gradients
    in the normalised [0,1] coordinate space, large enough to avoid
    division-by-zero for coincident nodes.

    Parameters
    ----------
    pred_pos : (B, pred_len, N, 2)
    gt_pos   : (B, pred_len, N, 2)

    Returns
    -------
    loss : (B,)
    """
    N = pred_pos.size(2)
    mask = _get_upper_tri_mask(N, pred_pos.device)

    diff_pred = pred_pos.unsqueeze(3) - pred_pos.unsqueeze(2)  # (B, T, N, N, 2)
    diff_gt = gt_pos.unsqueeze(3) - gt_pos.unsqueeze(2)

    d_pred = torch.norm(diff_pred, dim=-1)  # (B, T, N, N)
    d_gt = torch.norm(diff_gt, dim=-1)

    d_pred_pairs = d_pred[:, :, mask]  # (B, T, 66)
    d_gt_pairs = d_gt[:, :, mask]

    rel_err = ((d_pred_pairs - d_gt_pairs) / (d_gt_pairs + 0.01)) ** 2
    loss = rel_err.mean(dim=(1, 2))  # (B,)
    return loss


def activity_weight(activity, tau=None, w_min=0.1):
    """
    Compute per-sample trajectory loss weight based on activity level.

    Parameters
    ----------
    activity : (B,) normalised mean displacement
    tau : float or None.  If None, use batch median as adaptive threshold.
    w_min : float, minimum weight

    Returns
    -------
    w : (B,)  weights in [w_min, 1.0]
    """
    if tau is None:
        tau = activity.median().clamp(min=1e-6)
    w = (activity / tau).clamp(min=w_min, max=1.0)
    return w


def combined_loss(outputs, nodes, activity, pred_length, lambda_dist=0.1):
    """
    Full training loss.

    Parameters
    ----------
    outputs  : (B, seq_len, N, 5) — model output (Gaussian params)
    nodes    : (B, seq_len, N, 2) — ground truth positions
    activity : (B,) — per-window activity level
    pred_length : int
    lambda_dist : float — distance loss weight

    Returns
    -------
    total_loss : scalar
    loss_dict  : dict with individual loss components for logging
    """
    B = outputs.size(0)
    seq_len = outputs.size(1)
    obs_len = seq_len - pred_length

    # 1. Trajectory NLL (per sample) — output[t] predicts target[t+1]
    nll = gaussian_2d_nll(outputs, nodes, pred_length)  # (B,)

    # 2. Activity weighting for trajectory loss
    w = activity_weight(activity)  # (B,)
    weighted_nll = (w * nll).mean()

    # 3. Distance loss on predicted positions (same time shift)
    pred_params = outputs[:, obs_len - 1: -1, :, :]
    pred_pos = pred_params[..., :2]  # (B, pred, N, 2) — use mux, muy
    gt_pos = nodes[:, obs_len:, :, :]

    d_loss = distance_loss(pred_pos, gt_pos).mean()

    total = weighted_nll + lambda_dist * d_loss

    loss_dict = {
        "total": total.item(),
        "nll": nll.mean().item(),
        "nll_weighted": weighted_nll.item(),
        "dist": d_loss.item(),
        "w_mean": w.mean().item(),
    }
    return total, loss_dict
