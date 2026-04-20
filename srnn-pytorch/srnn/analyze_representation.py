"""Representation analysis for MouseSRNN hidden states + inter-attention.

Extracts node-RNN hidden states and inter-mouse attention weights for each
timestep of each validation / test window, then runs:

  1. PCA visualisations (colored by light / activity / chase)
  2. Temporal dynamics (inter-condition distance, hidden-state velocity,
     PCA trajectory, attention entropy over time)
  3. Probing classifiers (LogReg / MLP) that predict the ``light`` label
     from various embedding views:
        - last observed frame node state (128d)
        - global mean node state       (128d)
        - flattened inter-attention     (96d)
        - last-obs state + attention   (224d)
        - temporal snapshots @t=0,4,9,14,19 (640d)
     Plus a raw-feature baseline using activity / speed / mean inter-CB
     distance.
  4. Chase classification (only if enough positives).
  5. Attention pattern differences between light=0 / light=1 conditions.
  6. Final bar-chart summary figure.

All figures are saved under ``<save_root>/plots/representation``.  Numerical
results are printed to stdout and also written to
``<save_root>/plots/representation/metrics.txt``.

Usage (from ``srnn-pytorch/srnn``):

    python analyze_representation.py \
        --ckpt ../save/mice/v3_4kp_full/best_model.tar \
        --data ../data/mice/dataset_r1_w20_s10_fs4.npz

Add ``--splits val`` to skip the (slow) train-set embedding extraction when
you only need the visualisations.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

from model import MouseSRNN, build_edges_from_nodes

warnings.filterwarnings("ignore")


N_MICE = 3
N_KPS = 4
N_NODES = N_MICE * N_KPS  # 12
OBS_DEFAULT = 10
KP_NAMES = ["nose", "left_ear", "center_back", "tail_base"]
MOUSE_NAMES = [f"Mouse {i}" for i in range(N_MICE)]


# ═══════════════════════════════════════════════════════════════════════════
# Forward pass with state collection
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_representations(net, nodes_tensor):
    """Run a teacher-forced forward pass and return per-timestep hidden states.

    Returns
    -------
    node_states : ndarray  (B, T, N, nr)
        Node-RNN hidden state after each timestep.
    attn_inter  : ndarray  (B, T, N, K_inter)
        Inter-mouse attention weights for each node at each timestep.
    """
    B, T, N, _ = nodes_tensor.shape
    device = nodes_tensor.device

    h_temp = torch.zeros(B, N, net.er, device=device)
    c_temp = torch.zeros(B, N, net.er, device=device)
    h_spat = torch.zeros(B, net.n_spatial, net.er, device=device)
    c_spat = torch.zeros(B, net.n_spatial, net.er, device=device)
    h_node = torch.zeros(B, N, net.nr, device=device)
    c_node = torch.zeros(B, N, net.nr, device=device)

    edges_t, edges_s = build_edges_from_nodes(
        nodes_tensor, net._spatial_src, net._spatial_dst)

    node_states, attn_inter_all = [], []

    for t in range(T):
        te_in = net.temporal_edge_enc(edges_t[:, t])
        ht_f, ct_f = net.temporal_edge_rnn(
            te_in.reshape(B * N, -1),
            (h_temp.reshape(B * N, -1), c_temp.reshape(B * N, -1)))
        h_temp = ht_f.reshape(B, N, -1)
        c_temp = ct_f.reshape(B, N, -1)

        se_in = net.spatial_edge_enc(net._encode_spatial(edges_s[:, t]))
        hs_f, cs_f = net.spatial_edge_rnn(
            se_in.reshape(B * net.n_spatial, -1),
            (h_spat.reshape(B * net.n_spatial, -1),
             c_spat.reshape(B * net.n_spatial, -1)))
        h_spat = hs_f.reshape(B, net.n_spatial, -1)
        c_spat = cs_f.reshape(B, net.n_spatial, -1)

        h_intra, h_inter, _w_intra, w_inter, _ = net._attend(
            h_temp, h_spat, B, N, device)

        node_in = net.node_enc(nodes_tensor[:, t])
        edge_in = net.edge_attn_enc(
            torch.cat([h_temp, h_intra, h_inter], dim=-1))
        rnn_in = torch.cat([node_in, edge_in], dim=-1)
        hn_f, cn_f = net.node_rnn(
            rnn_in.reshape(B * N, -1),
            (h_node.reshape(B * N, -1), c_node.reshape(B * N, -1)))
        h_node = hn_f.reshape(B, N, -1)
        c_node = cn_f.reshape(B, N, -1)

        node_states.append(h_node.cpu().numpy())
        attn_inter_all.append(w_inter.cpu().numpy())

    return (np.stack(node_states, axis=1),
            np.stack(attn_inter_all, axis=1))


def compute_embeddings(net, data, batch_size, device, obs_length):
    """Run extract_representations batched and build summary embeddings."""
    n = len(data)
    ns_list, attn_list = [], []
    for i in range(0, n, batch_size):
        batch = torch.tensor(data[i:i + batch_size],
                             dtype=torch.float32).to(device)
        ns, attn = extract_representations(net, batch)
        ns_list.append(ns)
        attn_list.append(attn)
    node_states = np.concatenate(ns_list, axis=0)    # (N, T, 12, nr)
    attn_weights = np.concatenate(attn_list, axis=0)  # (N, T, 12, K_inter)

    last_obs = obs_length - 1
    out = {
        "node_states": node_states,
        "attn_weights": attn_weights,
        # mean over all keypoints at last observed frame
        "emb_last_obs": node_states[:, last_obs].mean(axis=1),
        # mean over all time steps and keypoints
        "emb_global": node_states.mean(axis=(1, 2)),
        # 5 temporal snapshots (t=0, mid-obs, last-obs, mid-pred, last-pred)
        "emb_temporal": np.concatenate(
            [node_states[:, t].mean(axis=1)
             for t in [0, last_obs // 2, last_obs,
                       last_obs + (node_states.shape[1] - obs_length) // 2,
                       node_states.shape[1] - 1]],
            axis=1),
        # flattened inter-attention pattern at last obs step
        "emb_attn": attn_weights[:, last_obs].reshape(n, -1),
    }
    out["emb_combined"] = np.concatenate(
        [out["emb_last_obs"], out["emb_attn"]], axis=1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════


def plot_pca_scatter(val, save_path):
    emb_global = val["emb_global"]
    emb_attn = val["emb_attn"]
    lights = val["lights"]
    activity = val["activity"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("PCA of Hidden Representations",
                 fontsize=16, fontweight="bold")

    configs = [
        ("Global Node State", emb_global, "Light Condition", lights, "coolwarm"),
        ("Global Node State", emb_global, "Activity Level", activity, "viridis"),
        ("Attention Pattern (t=obs-1)", emb_attn, "Light Condition", lights,
         "coolwarm"),
    ]
    for col, (emb_name, emb, label_name, labels, cmap) in enumerate(configs):
        X_s = StandardScaler().fit_transform(emb)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_s)
        for row, (px, py) in enumerate([(0, 1), (1, 2)]):
            ax = axes[row, col]
            sc = ax.scatter(X_pca[:, px], X_pca[:, py], c=labels, cmap=cmap,
                            alpha=0.3, s=5, rasterized=True)
            ax.set_xlabel(f"PC{px + 1} "
                          f"({pca.explained_variance_ratio_[px]:.1%})")
            ax.set_ylabel(f"PC{py + 1} "
                          f"({pca.explained_variance_ratio_[py]:.1%})")
            if row == 0:
                ax.set_title(f"{emb_name}\nColored by {label_name}", fontsize=11)
            plt.colorbar(sc, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pca_chase(val, save_path):
    emb_global = val["emb_global"]
    chase = val["chase"]
    X_s = StandardScaler().fit_transform(emb_global)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_s)

    chase_mask = chase == 1
    no_chase_mask = chase == 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PCA: Chase Events Highlighted",
                 fontsize=14, fontweight="bold")
    for i, (px, py) in enumerate([(0, 1), (1, 2), (0, 2)]):
        ax = axes[i]
        ax.scatter(X_pca[no_chase_mask, px], X_pca[no_chase_mask, py],
                   c="lightgray", alpha=0.2, s=3,
                   label=f"No chase (n={no_chase_mask.sum()})")
        ax.scatter(X_pca[chase_mask, px], X_pca[chase_mask, py],
                   c="red", alpha=0.9, s=30, marker="*",
                   edgecolors="black", linewidths=0.5,
                   label=f"Chase (n={chase_mask.sum()})", zorder=5)
        ax.set_xlabel(f"PC{px + 1} "
                      f"({pca.explained_variance_ratio_[px]:.1%})")
        ax.set_ylabel(f"PC{py + 1} "
                      f"({pca.explained_variance_ratio_[py]:.1%})")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_dynamics(val, save_path, obs_length):
    node_states = val["node_states"]         # (N, T, 12, nr)
    attn_weights = val["attn_weights"]       # (N, T, 12, K_inter)
    lights = val["lights"]
    T = node_states.shape[1]
    all_temporal = node_states.mean(axis=2)  # (N, T, nr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Temporal Dynamics of Hidden Representations",
                 fontsize=14, fontweight="bold")

    # (a) inter-condition distance over time
    mean_repr = {
        k: all_temporal[m].mean(axis=0)
        for k, m in [("light=0", lights == 0), ("light=1", lights == 1)]
    }
    dists = [np.linalg.norm(mean_repr["light=0"][t] - mean_repr["light=1"][t])
             for t in range(T)]
    ax = axes[0, 0]
    ax.plot(range(T), dists, "b-o", markersize=4)
    ax.axvline(x=obs_length - 0.5, color="red", linestyle="--", alpha=0.7,
               label="obs→pred boundary")
    ax.set(xlabel="Timestep", ylabel="L2 Distance",
           title="Light=0 vs Light=1 Representation Distance")
    ax.legend(); ax.grid(alpha=0.3)

    # (b) hidden state velocity
    repr_vel = np.linalg.norm(np.diff(all_temporal, axis=1), axis=-1)
    ax = axes[0, 1]
    for lbl, mask, color in [("Light=0", lights == 0, "blue"),
                             ("Light=1", lights == 1, "orange")]:
        vm = repr_vel[mask].mean(axis=0)
        vs = repr_vel[mask].std(axis=0) / np.sqrt(max(mask.sum(), 1))
        ax.plot(range(T - 1), vm, color=color, label=lbl, linewidth=2)
        ax.fill_between(range(T - 1), vm - vs, vm + vs, color=color, alpha=0.2)
    ax.axvline(x=obs_length - 1, color="red", linestyle="--", alpha=0.7)
    ax.set(xlabel="Timestep transition", ylabel="Hidden state velocity (L2)",
           title="Rate of Hidden State Change")
    ax.legend(); ax.grid(alpha=0.3)

    # (c) mean trajectory in PCA space
    pca_traj = PCA(n_components=2)
    all_tp = np.vstack([mean_repr["light=0"], mean_repr["light=1"]])
    all_pca = pca_traj.fit_transform(all_tp)
    traj_l0, traj_l1 = all_pca[:T], all_pca[T:]
    ax = axes[1, 0]
    for traj, lbl, color in [(traj_l0, "Light=0", "blue"),
                             (traj_l1, "Light=1", "orange")]:
        ax.plot(traj[:obs_length, 0], traj[:obs_length, 1],
                color=color, linewidth=2, label=f"{lbl} (obs)")
        ax.plot(traj[obs_length - 1:, 0], traj[obs_length - 1:, 1],
                color=color, linewidth=2, linestyle="--", alpha=0.6)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o",
                   s=60, zorder=5)
        ax.scatter(traj[obs_length - 1, 0], traj[obs_length - 1, 1],
                   color=color, marker="s", s=60, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="^",
                   s=60, zorder=5)
    ax.set(xlabel=f"PC1 ({pca_traj.explained_variance_ratio_[0]:.1%})",
           ylabel=f"PC2 ({pca_traj.explained_variance_ratio_[1]:.1%})",
           title="Mean Representation Trajectory in PCA Space")
    ax.legend(); ax.grid(alpha=0.3)

    # (d) attention entropy over time
    ent = -(attn_weights * np.log(attn_weights + 1e-10)).sum(axis=-1)
    ent_t = ent.mean(axis=2)  # (N, T)
    ax = axes[1, 1]
    for lbl, mask, color in [("Light=0", lights == 0, "blue"),
                             ("Light=1", lights == 1, "orange")]:
        em = ent_t[mask].mean(axis=0)
        es = ent_t[mask].std(axis=0) / np.sqrt(max(mask.sum(), 1))
        ax.plot(range(T), em, color=color, label=lbl, linewidth=2)
        ax.fill_between(range(T), em - es, em + es, color=color, alpha=0.2)
    ax.axvline(x=obs_length - 0.5, color="red", linestyle="--", alpha=0.7)
    ax.set(xlabel="Timestep", ylabel="Attention Entropy",
           title="Inter-Mouse Attention Entropy Over Time")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_classification_summary(results, save_path):
    """Bar chart of light-condition classification AUC / balanced accuracy."""
    methods = [r["name"] for r in results]
    lr_auc = [r["lr_auc"] for r in results]
    mlp_auc = [r["mlp_auc"] if r["mlp_auc"] is not None else 0 for r in results]
    lr_bacc = [r["lr_bacc"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Light Condition Classification Performance",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(methods))
    w = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - w / 2, lr_auc, w, label="LogReg AUC", color="steelblue")
    ax.bar(x + w / 2, mlp_auc, w, label="MLP AUC", color="coral")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9, rotation=15)
    ax.set_ylabel("AUC-ROC"); ax.set_title("AUC-ROC by Embedding Type")
    ax.legend(); ax.set_ylim(0.4, max(0.9, max(lr_auc + mlp_auc) + 0.05))
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, lr_auc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.bar(x, lr_bacc, 0.5, color="steelblue")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9, rotation=15)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy (LogReg)")
    ax.legend(); ax.set_ylim(0.4, max(0.8, max(lr_bacc) + 0.05))
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(lr_bacc):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_by_condition(val, save_path, obs_length):
    """Heatmap of inter attention weights averaged over obs frames per light."""
    attn = val["attn_weights"]        # (N, T, 12, K_inter)
    lights = val["lights"]
    attn_obs = attn[:, :obs_length].mean(axis=1)  # (N, 12, K_inter)
    K_inter = attn_obs.shape[-1]

    fig, axes = plt.subplots(2, N_MICE, figsize=(6 * N_MICE, 10))
    fig.suptitle("Inter-Mouse Attention Patterns by Light Condition "
                 "(observation mean)", fontsize=14, fontweight="bold")

    for m_idx in range(N_MICE):
        other_mice = [i for i in range(N_MICE) if i != m_idx]
        neighbor_labels = [
            f"M{om}_{KP_NAMES[kp][:3]}"
            for om in other_mice for kp in range(N_KPS)
        ][:K_inter]
        for row, (label, mask) in enumerate(
                [("Light=0", lights == 0), ("Light=1", lights == 1)]):
            ax = axes[row, m_idx]
            s = m_idx * N_KPS
            am = attn_obs[mask, s:s + N_KPS, :].mean(axis=0)  # (4, K_inter)
            im = ax.imshow(am, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)
            ax.set_xticks(range(K_inter))
            ax.set_xticklabels(neighbor_labels, rotation=45, ha="right",
                               fontsize=8)
            ax.set_yticks(range(N_KPS))
            ax.set_yticklabels(KP_NAMES, fontsize=9)
            ax.set_title(f"{MOUSE_NAMES[m_idx]} — {label}", fontsize=11)
            for i in range(N_KPS):
                for j in range(K_inter):
                    ax.text(j, i, f"{am[i, j]:.2f}", ha="center",
                            va="center", fontsize=7,
                            color="white" if am[i, j] > 0.3 else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Numeric analyses
# ═══════════════════════════════════════════════════════════════════════════


def pca_report(val):
    print("\n" + "=" * 60)
    print("  PCA Analysis (val split)")
    print("=" * 60)
    for emb_name, key in [("Last-obs node state", "emb_last_obs"),
                          ("Global mean", "emb_global"),
                          ("Attention pattern", "emb_attn"),
                          ("Combined", "emb_combined")]:
        X = val[key]
        X_s = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X_s)
        print(f"\n  --- {emb_name} (dim={X.shape[1]}) ---")
        print(f"  Top-5 EV: {pca.explained_variance_ratio_[:5].round(4)}  "
              f"(cum {pca.explained_variance_ratio_[:5].sum():.1%})")
        for pc_i in range(3):
            g0 = X_pca[val["lights"] == 0, pc_i]
            g1 = X_pca[val["lights"] == 1, pc_i]
            d = (g1.mean() - g0.mean()) / np.sqrt(
                (g0.std() ** 2 + g1.std() ** 2) / 2 + 1e-12)
            strength = ("strong" if abs(d) > 0.8
                        else "medium" if abs(d) > 0.5 else "weak")
            c = np.corrcoef(X_pca[:, pc_i], val["activity"])[0, 1]
            print(f"  PC{pc_i + 1}: light Cohen-d={d:+.3f} ({strength})   "
                  f"activity r={c:+.3f}")


def dynamics_report(val, obs_length):
    print("\n" + "=" * 60)
    print("  Temporal Dynamics")
    print("=" * 60)
    node_states = val["node_states"]
    lights = val["lights"]
    T = node_states.shape[1]

    mean_t = {k: node_states[m].mean(axis=(0, 2))
              for k, m in [("l0", lights == 0), ("l1", lights == 1)]}
    dists = [np.linalg.norm(mean_t["l0"][t] - mean_t["l1"][t]) for t in range(T)]
    print(f"\n  Obs mean dist (t<{obs_length}):  "
          f"{np.mean(dists[:obs_length]):.4f}")
    print(f"  Pred mean dist (t>={obs_length}): "
          f"{np.mean(dists[obs_length:]):.4f}")

    all_temporal = node_states.mean(axis=2)
    repr_vel = np.linalg.norm(np.diff(all_temporal, axis=1), axis=-1).mean(axis=1)
    t_s, p_v = ttest_ind(repr_vel[lights == 0], repr_vel[lights == 1])
    print(f"\n  Hidden state velocity per sample:")
    print(f"    Light=0: {repr_vel[lights == 0].mean():.4f} "
          f"± {repr_vel[lights == 0].std():.4f}")
    print(f"    Light=1: {repr_vel[lights == 1].mean():.4f} "
          f"± {repr_vel[lights == 1].std():.4f}")
    print(f"    t={t_s:+.2f}, p={p_v:.2e}")


def _compute_raw_features(data, activity, arena_px):
    speed = np.linalg.norm(np.diff(data, axis=1), axis=-1).mean(axis=(1, 2))
    speed = speed * arena_px
    inter_dist = []
    for m1 in range(N_MICE):
        for m2 in range(m1 + 1, N_MICE):
            cb1 = data[:, :, m1 * N_KPS + 2]
            cb2 = data[:, :, m2 * N_KPS + 2]
            inter_dist.append(np.linalg.norm(cb1 - cb2, axis=-1).mean(axis=1)
                              * arena_px)
    inter_dist = np.stack(inter_dist, axis=1).mean(axis=1)
    return np.column_stack([activity, speed, inter_dist])


def classify_light(all_embeddings, raw_data_by_split, arena_px, seed=42):
    """Return a list of dicts suitable for the summary bar chart."""
    print("\n" + "=" * 60)
    print("  Light Classification (probing classifiers)")
    print("=" * 60)

    train, test = all_embeddings["train"], all_embeddings["test"]

    configs = [
        ("Raw\n(3d)",       None),
        ("Attn\n(96d)",     "emb_attn"),
        ("Last-obs\n(128d)", "emb_last_obs"),
        ("Global\n(128d)",  "emb_global"),
        ("Combined\n(224d)", "emb_combined"),
        ("Temporal\n(640d)", "emb_temporal"),
    ]

    results = []
    for name, key in configs:
        if key is None:
            X_tr = _compute_raw_features(raw_data_by_split["train"],
                                         train["activity"], arena_px)
            X_te = _compute_raw_features(raw_data_by_split["test"],
                                         test["activity"], arena_px)
        else:
            X_tr, X_te = train[key], test[key]

        y_tr, y_te = train["lights"], test["lights"]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        lr.fit(X_tr_s, y_tr)
        y_pred = lr.predict(X_te_s)
        y_prob = lr.predict_proba(X_te_s)[:, 1]
        lr_acc = accuracy_score(y_te, y_pred)
        lr_bacc = balanced_accuracy_score(y_te, y_pred)
        lr_auc = roc_auc_score(y_te, y_prob)
        lr_f1 = f1_score(y_te, y_pred)

        mlp_auc = None
        if key is not None:
            mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                                early_stopping=True, random_state=seed)
            mlp.fit(X_tr_s, y_tr)
            mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te_s)[:, 1])

        print(f"\n  {name.replace(chr(10), ' ')}:")
        print(f"    LogReg: acc={lr_acc:.3f}  bacc={lr_bacc:.3f}  "
              f"AUC={lr_auc:.3f}  F1={lr_f1:.3f}")
        if mlp_auc is not None:
            print(f"    MLP64:  AUC={mlp_auc:.3f}")

        results.append({"name": name, "lr_acc": lr_acc, "lr_bacc": lr_bacc,
                        "lr_auc": lr_auc, "lr_f1": lr_f1, "mlp_auc": mlp_auc})
    return results


def classify_chase(all_embeddings):
    print("\n" + "=" * 60)
    print("  Chase Classification (highly imbalanced)")
    print("=" * 60)
    train, test = all_embeddings["train"], all_embeddings["test"]
    n_tr = int((train["chase"] == 1).sum())
    n_te = int((test["chase"] == 1).sum())
    print(f"  Train: {n_tr} chase / {len(train['chase'])} total")
    print(f"  Test:  {n_te} chase / {len(test['chase'])} total")
    if n_tr < 10 or n_te < 3:
        print("  Too few chase positives for reliable classification.")
        return
    for name, key in [("Combined (224d)", "emb_combined"),
                      ("Temporal (640d)", "emb_temporal")]:
        X_tr, X_te = train[key], test[key]
        y_tr, y_te = train["chase"], test["chase"]
        X_tr_s = StandardScaler().fit_transform(X_tr)
        X_te_s = StandardScaler().fit(X_tr).transform(X_te)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        lr.fit(X_tr_s, y_tr)
        y_pred = lr.predict(X_te_s)
        y_prob = lr.predict_proba(X_te_s)[:, 1]
        try:
            auc = roc_auc_score(y_te, y_prob)
        except ValueError:
            auc = float("nan")
        tp = int(((y_pred == 1) & (y_te == 1)).sum())
        fp = int(((y_pred == 1) & (y_te == 0)).sum())
        fn = int(((y_pred == 0) & (y_te == 1)).sum())
        print(f"\n  {name}:")
        print(f"    acc={accuracy_score(y_te, y_pred):.3f}  "
              f"bacc={balanced_accuracy_score(y_te, y_pred):.3f}  "
              f"AUC={auc:.3f}")
        print(f"    Chase recall {tp}/{tp + fn}, false positives {fp}")


def attention_report(val, obs_length):
    print("\n" + "=" * 60)
    print("  Attention Patterns by Light Condition")
    print("=" * 60)
    attn = val["attn_weights"]
    lights = val["lights"]
    attn_obs_mean = attn[:, :obs_length].mean(axis=(1, 2))  # (N, K_inter)
    for k in range(attn_obs_mean.shape[1]):
        a0 = attn_obs_mean[lights == 0, k].mean()
        a1 = attn_obs_mean[lights == 1, k].mean()
        print(f"    neighbor {k}: light0={a0:.4f}  "
              f"light1={a1:.4f}  Δ={a1 - a0:+.4f}")
    ent = -(attn * np.log(attn + 1e-10)).sum(axis=-1).mean(axis=(1, 2))
    t_s, p_v = ttest_ind(ent[lights == 0], ent[lights == 1])
    print(f"\n  Attention entropy (mean over time & nodes):")
    print(f"    Light=0: {ent[lights == 0].mean():.4f}")
    print(f"    Light=1: {ent[lights == 1].mean():.4f}")
    print(f"    t={t_s:+.2f}, p={p_v:.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="../save/mice/v3_4kp_full/best_model.tar",
                        help="Path to a MouseSRNN checkpoint (.tar)")
    parser.add_argument("--data", type=str,
                        default="../data/mice/dataset_r1_w20_s10_fs4.npz")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory. Defaults to "
                             "<ckpt_dir>/plots/representation")
    parser.add_argument("--obs_length", type=int, default=OBS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--arena_px", type=float, default=450.0)
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to extract embeddings for. "
                             "Classification requires train+test.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    net = MouseSRNN(saved_args).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"  Model loaded (epoch={ckpt.get('epoch', '?')})")

    save_dir = args.save_dir
    if save_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        save_dir = os.path.join(ckpt_dir, "plots", "representation")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    raw = np.load(args.data)
    raw_data_by_split, all_embeddings = {}, {}
    for split in args.splits:
        split_data = raw[f"{split}_data"].reshape(-1, 20, N_NODES, 2)
        lights = raw[f"{split}_lights"]
        chase = raw[f"{split}_chase"]
        activity = raw[f"{split}_activity"]
        print(f"\n[{split}] extracting embeddings for {len(split_data)} "
              f"windows ...")
        emb = compute_embeddings(net, split_data, args.batch_size, device,
                                 args.obs_length)
        emb["lights"] = lights
        emb["chase"] = chase
        emb["activity"] = activity
        all_embeddings[split] = emb
        raw_data_by_split[split] = split_data
        print(f"  node_states shape = {emb['node_states'].shape}")
        print(f"  lights counts = "
              f"{dict(zip(*np.unique(lights, return_counts=True)))}")

    metrics_path = os.path.join(save_dir, "metrics.txt")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
        def flush(self):
            for st in self.streams:
                st.flush()

    with open(metrics_path, "w") as fh, redirect_stdout(_Tee(sys.stdout, fh)):
        if "val" in all_embeddings:
            pca_report(all_embeddings["val"])
            dynamics_report(all_embeddings["val"], args.obs_length)

        results = None
        if "train" in all_embeddings and "test" in all_embeddings:
            results = classify_light(all_embeddings, raw_data_by_split,
                                     args.arena_px, seed=args.seed)
            classify_chase(all_embeddings)

        if "val" in all_embeddings:
            attention_report(all_embeddings["val"], args.obs_length)

    # ── Figures (only require val) ──────────────────────────────────────
    if "val" in all_embeddings:
        val = all_embeddings["val"]
        print("\nSaving figures ...")
        plot_pca_scatter(val, os.path.join(save_dir, "pca_scatter.png"))
        plot_pca_chase(val, os.path.join(save_dir, "pca_chase.png"))
        plot_temporal_dynamics(val,
                               os.path.join(save_dir, "temporal_dynamics.png"),
                               args.obs_length)
        plot_attention_by_condition(
            val, os.path.join(save_dir, "attention_by_condition.png"),
            args.obs_length)
        print(f"  Saved: {save_dir}/pca_scatter.png")
        print(f"  Saved: {save_dir}/pca_chase.png")
        print(f"  Saved: {save_dir}/temporal_dynamics.png")
        print(f"  Saved: {save_dir}/attention_by_condition.png")

    if results is not None:
        plot_classification_summary(
            results, os.path.join(save_dir, "classification_summary.png"))
        print(f"  Saved: {save_dir}/classification_summary.png")

    print(f"\nDone. Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
