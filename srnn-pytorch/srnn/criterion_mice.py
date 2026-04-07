"""
Loss functions for mouse trajectory prediction.

L_total = w(activity) * L_trajectory  +  lambda * L_body

- L_trajectory: bivariate Gaussian NLL (per-node, per-frame)
- L_body:       intra-mouse pairwise distance MSE (body rigidity constraint)
                Only enforces distances within each mouse (C(4,2)=6 × 3 = 18 pairs).
                Inter-mouse distances are NOT constrained here — the attention
                mechanism should learn social interactions from the NLL gradient.
- w(activity):  per-window scalar that down-weights static windows
"""

import torch
import numpy as np

N_NODES = 12
N_MICE = 3
N_KPS = 4

_INTRA_MOUSE_MASK = None


def _get_intra_mouse_mask(device):
    """
    Build (12, 12) bool mask selecting only intra-mouse upper-triangle pairs.

    For 3 mice × 4 keypoints:
      Mouse 0: nodes 0-3  → C(4,2)=6 pairs
      Mouse 1: nodes 4-7  → 6 pairs
      Mouse 2: nodes 8-11 → 6 pairs
      Total: 18 pairs  (NOT the 48 inter-mouse pairs)
    """
    global _INTRA_MOUSE_MASK
    if _INTRA_MOUSE_MASK is not None and _INTRA_MOUSE_MASK.device == device:
        return _INTRA_MOUSE_MASK
    mask = torch.zeros(N_NODES, N_NODES, dtype=torch.bool, device=device)
    for m in range(N_MICE):
        start = m * N_KPS
        for i in range(N_KPS):
            for j in range(i + 1, N_KPS):
                mask[start + i, start + j] = True
    _INTRA_MOUSE_MASK = mask
    return mask


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
    z = z.clamp(min=0.0)
    neg_rho = (1.0 - corr ** 2).clamp(min=1e-4)

    log_prob = (
        -0.5 * z / neg_rho
        - torch.log(sxsy + 1e-8)
        - 0.5 * torch.log(neg_rho)
        - np.log(2 * np.pi)
    )

    nll = -log_prob.mean(dim=(1, 2))  # mean over pred_frames, nodes → (B,)
    return nll


def body_distance_loss(pred_pos, gt_pos):
    """
    Intra-mouse pairwise distance loss: C(4,2)=6 pairs × 3 mice = 18 pairs.

    Enforces body rigidity — the 4 keypoints of each mouse should maintain
    their relative distances.  Inter-mouse distances are deliberately excluded
    so the attention mechanism must learn social interactions from the NLL.

    L = mean_{intra_pairs} [ (d_pred - d_gt)^2 / (d_gt + eps)^2 ]

    eps = 0.01 (≈4.5 px in normalised coords).

    Parameters
    ----------
    pred_pos : (B, pred_len, N, 2)
    gt_pos   : (B, pred_len, N, 2)

    Returns
    -------
    loss : (B,)
    """
    mask = _get_intra_mouse_mask(pred_pos.device)  # (12, 12) with 18 True entries

    diff_pred = pred_pos.unsqueeze(3) - pred_pos.unsqueeze(2)  # (B, T, 12, 12, 2)
    diff_gt = gt_pos.unsqueeze(3) - gt_pos.unsqueeze(2)

    d_pred = torch.norm(diff_pred, dim=-1)  # (B, T, 12, 12)
    d_gt = torch.norm(diff_gt, dim=-1)

    d_pred_pairs = d_pred[:, :, mask]  # (B, T, 18)
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

    # 3. Body distance loss on predicted positions (intra-mouse only)
    pred_params = outputs[:, obs_len - 1: -1, :, :]
    pred_pos = pred_params[..., :2]  # (B, pred, N, 2) — use mux, muy
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
