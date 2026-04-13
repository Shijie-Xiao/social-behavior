"""
Batched SRNN for configurable mouse graph.

Supports different numbers of keypoints per mouse and two graph types:
  - "full":  all nodes connected (including intra-mouse)
  - "inter": only inter-mouse edges (no intra-mouse connections)

Architecture (per time-step):
  1. Temporal edge RNN:  edge(i,i) encodes displacement from t-1 to t
  2. Spatial edge RNN:   edge(i,j) encodes relative position at t
     - Input: [direction, log(dist), kp_type_emb_src, kp_type_emb_dst]
  3. Two-stream attention: for each node i,
     - Intra-mouse attention: softmax over same-mouse neighbors (body structure)
     - Inter-mouse attention: softmax over other-mouse neighbors (social interaction)
  4. Node RNN: combine (position, temporal_h, intra_attn, inter_attn) → LSTM
  5. Output:   linear → 5-d Gaussian parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MICE = 3
KP_EMB_DIM = 8


def build_spatial_index(n_nodes, n_kps, graph_type="full"):
    """
    Build index tensors for the spatial edge graph, split into
    intra-mouse and inter-mouse groups.

    Returns
    -------
    src, dst          : (E,) source/destination node indices
    node_to_edges     : (n_nodes, K) all edge indices per node
    node_to_intra     : (n_nodes, K_intra) or None if no intra-mouse edges
    node_to_inter     : (n_nodes, K_inter) inter-mouse edge indices per node
    """
    src, dst = [], []
    node_to_edges = [[] for _ in range(n_nodes)]
    node_to_intra = [[] for _ in range(n_nodes)]
    node_to_inter = [[] for _ in range(n_nodes)]
    idx = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if graph_type == "inter":
                if i // n_kps == j // n_kps:
                    continue
            src.append(i)
            dst.append(j)
            node_to_edges[i].append(idx)
            if i // n_kps == j // n_kps:
                node_to_intra[i].append(idx)
            else:
                node_to_inter[i].append(idx)
            idx += 1

    k_per_node = len(node_to_edges[0])
    k_intra = len(node_to_intra[0])
    k_inter = len(node_to_inter[0])
    for i in range(n_nodes):
        assert len(node_to_edges[i]) == k_per_node
        assert len(node_to_intra[i]) == k_intra
        assert len(node_to_inter[i]) == k_inter

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    edges_t = torch.tensor(node_to_edges, dtype=torch.long)
    intra_t = torch.tensor(node_to_intra, dtype=torch.long) if k_intra > 0 else None
    inter_t = torch.tensor(node_to_inter, dtype=torch.long)

    return src_t, dst_t, edges_t, intra_t, inter_t


class MouseSRNN(nn.Module):
    def __init__(self, args, infer=False):
        super().__init__()
        self.infer = infer
        self.residual = getattr(args, "residual", False)

        n_kps = getattr(args, "n_keypoints", 4)
        graph_type = getattr(args, "graph_type", "full")
        self.n_nodes = N_MICE * n_kps
        self.n_kps = n_kps
        self.graph_type = graph_type

        spatial_src, spatial_dst, node_to_edges, node_to_intra, node_to_inter = \
            build_spatial_index(self.n_nodes, n_kps, graph_type)

        self.register_buffer("_spatial_src", spatial_src)
        self.register_buffer("_spatial_dst", spatial_dst)
        self.register_buffer("_node_to_edges", node_to_edges)
        self.n_spatial = len(spatial_src)
        self.n_neighbors = node_to_edges.shape[1]

        self.has_intra = node_to_intra is not None
        if self.has_intra:
            self.register_buffer("_node_to_intra", node_to_intra)
            self.n_intra_neighbors = node_to_intra.shape[1]
        else:
            self.register_buffer(
                "_node_to_intra",
                torch.zeros(self.n_nodes, 0, dtype=torch.long))
            self.n_intra_neighbors = 0
        self.register_buffer("_node_to_inter", node_to_inter)
        self.n_inter_neighbors = node_to_inter.shape[1]

        self.register_buffer("_spatial_src_kp", spatial_src % n_kps)
        self.register_buffer("_spatial_dst_kp", spatial_dst % n_kps)

        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length

        nr = args.human_node_rnn_size
        er = args.human_human_edge_rnn_size
        ne = args.human_node_embedding_size
        ee = args.human_human_edge_embedding_size
        attn_size = args.attention_size
        drop = args.dropout

        self.nr = nr
        self.er = er

        self.kp_type_emb = nn.Embedding(n_kps, KP_EMB_DIM)

        self.temporal_edge_enc = nn.Sequential(
            nn.Linear(2, ee), nn.ReLU(), nn.Dropout(drop))
        self.temporal_edge_rnn = nn.LSTMCell(ee, er)

        spatial_in_dim = 3 + 2 * KP_EMB_DIM
        self.spatial_edge_enc = nn.Sequential(
            nn.Linear(spatial_in_dim, ee), nn.ReLU(), nn.Dropout(drop))
        self.spatial_edge_rnn = nn.LSTMCell(ee, er)

        self.attn_q = nn.Linear(er, attn_size)
        self.attn_k_intra = nn.Linear(er, attn_size)
        self.attn_k_inter = nn.Linear(er, attn_size)
        self.attn_score_intra = nn.Linear(attn_size, 1)
        self.attn_score_inter = nn.Linear(attn_size, 1)

        self.node_enc = nn.Sequential(
            nn.Linear(2, ne), nn.ReLU(), nn.Dropout(drop))
        self.edge_attn_enc = nn.Sequential(
            nn.Linear(er * 3, ne), nn.ReLU(), nn.Dropout(drop))
        self.node_rnn = nn.LSTMCell(ne * 2, nr)

        self.output_linear = nn.Linear(nr, 5)

    # ── helpers ──────────────────────────────────────────────────────

    def _encode_spatial(self, displacement):
        """Convert raw displacement (B, E, 2) → enriched (B, E, 3+2*KP_EMB_DIM)."""
        dist = displacement.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = displacement / dist
        log_dist = torch.log(dist)
        B = displacement.size(0)
        src_emb = self.kp_type_emb(self._spatial_src_kp).unsqueeze(0).expand(B, -1, -1)
        dst_emb = self.kp_type_emb(self._spatial_dst_kp).unsqueeze(0).expand(B, -1, -1)
        return torch.cat([direction, log_dist, src_emb, dst_emb], dim=-1)

    def _attend(self, h_temp, h_spat, B, N, device):
        """
        Two-stream additive (Bahdanau) attention with separate softmax
        for intra-mouse and inter-mouse edges.

        Uses score = v^T tanh(Q(h_temp) + K(h_spat)) instead of dot-product
        to avoid gradient vanishing when h_temp ≈ 0 (LSTM zero-init).

        Returns
        -------
        h_intra_attn : (B, N, er)
        h_inter_attn : (B, N, er)
        w_intra      : (B, N, K_intra) or None
        w_inter      : (B, N, K_inter)
        entropy      : scalar tensor — mean attention entropy for regularization
        """
        q = self.attn_q(h_temp).unsqueeze(-2)          # (B, N, 1, A)

        entropy = torch.tensor(0.0, device=device)

        if self.has_intra:
            h_sp_intra = h_spat[:, self._node_to_intra, :]
            k_intra = self.attn_k_intra(h_sp_intra)    # (B, N, K_intra, A)
            s_intra = self.attn_score_intra(
                torch.tanh(q + k_intra)).squeeze(-1)    # (B, N, K_intra)
            w_intra = F.softmax(s_intra, dim=-1)
            h_intra_attn = torch.einsum("bnk,bnke->bne", w_intra, h_sp_intra)
            entropy = entropy - (w_intra * torch.log(w_intra + 1e-10)).sum(-1).mean()
        else:
            h_intra_attn = torch.zeros(B, N, self.er, device=device)
            w_intra = None

        h_sp_inter = h_spat[:, self._node_to_inter, :]
        k_inter = self.attn_k_inter(h_sp_inter)        # (B, N, K_inter, A)
        s_inter = self.attn_score_inter(
            torch.tanh(q + k_inter)).squeeze(-1)        # (B, N, K_inter)
        w_inter = F.softmax(s_inter, dim=-1)
        h_inter_attn = torch.einsum("bnk,bnke->bne", w_inter, h_sp_inter)
        entropy = entropy - (w_inter * torch.log(w_inter + 1e-10)).sum(-1).mean()

        return h_intra_attn, h_inter_attn, w_intra, w_inter, entropy

    # ── forward paths ────────────────────────────────────────────────

    def forward(self, nodes, edges_temporal=None, edges_spatial=None,
                obs_length=10, ss_prob=0.0):
        if edges_temporal is not None and ss_prob <= 0.0:
            return self._forward_tf(nodes, edges_temporal, edges_spatial)
        return self._forward_ss(nodes, obs_length, ss_prob)

    def _forward_tf(self, nodes, edges_temporal, edges_spatial):
        B, T, N, _ = nodes.shape
        device = nodes.device

        h_temp = torch.zeros(B, N, self.er, device=device)
        c_temp = torch.zeros(B, N, self.er, device=device)
        h_spat = torch.zeros(B, self.n_spatial, self.er, device=device)
        c_spat = torch.zeros(B, self.n_spatial, self.er, device=device)
        h_node = torch.zeros(B, N, self.nr, device=device)
        c_node = torch.zeros(B, N, self.nr, device=device)

        outputs = []
        total_entropy = torch.tensor(0.0, device=device)
        for t in range(T):
            te_in = self.temporal_edge_enc(edges_temporal[:, t])
            ht_f, ct_f = self.temporal_edge_rnn(
                te_in.reshape(B * N, -1),
                (h_temp.reshape(B * N, -1), c_temp.reshape(B * N, -1)))
            h_temp = ht_f.reshape(B, N, -1)
            c_temp = ct_f.reshape(B, N, -1)

            se_in = self.spatial_edge_enc(
                self._encode_spatial(edges_spatial[:, t]))
            hs_f, cs_f = self.spatial_edge_rnn(
                se_in.reshape(B * self.n_spatial, -1),
                (h_spat.reshape(B * self.n_spatial, -1),
                 c_spat.reshape(B * self.n_spatial, -1)))
            h_spat = hs_f.reshape(B, self.n_spatial, -1)
            c_spat = cs_f.reshape(B, self.n_spatial, -1)

            h_intra, h_inter, _, _, ent = self._attend(
                h_temp, h_spat, B, N, device)
            total_entropy = total_entropy + ent

            node_in = self.node_enc(nodes[:, t])
            edge_in = self.edge_attn_enc(
                torch.cat([h_temp, h_intra, h_inter], dim=-1))
            rnn_in = torch.cat([node_in, edge_in], dim=-1)
            hn_f, cn_f = self.node_rnn(
                rnn_in.reshape(B * N, -1),
                (h_node.reshape(B * N, -1), c_node.reshape(B * N, -1)))
            h_node = hn_f.reshape(B, N, -1)
            c_node = cn_f.reshape(B, N, -1)

            out = self.output_linear(h_node)
            if self.residual:
                out = out.clone()
                out[..., :2] = out[..., :2] + nodes[:, t]
            outputs.append(out)

        avg_entropy = total_entropy / T
        return torch.stack(outputs, dim=1), avg_entropy

    def _forward_ss(self, nodes, obs_length, ss_prob):
        B, T, N, _ = nodes.shape
        device = nodes.device

        h_temp = torch.zeros(B, N, self.er, device=device)
        c_temp = torch.zeros(B, N, self.er, device=device)
        h_spat = torch.zeros(B, self.n_spatial, self.er, device=device)
        c_spat = torch.zeros(B, self.n_spatial, self.er, device=device)
        h_node = torch.zeros(B, N, self.nr, device=device)
        c_node = torch.zeros(B, N, self.nr, device=device)

        prev_pos = nodes[:, 0]
        prev_out = None
        outputs = []
        total_entropy = torch.tensor(0.0, device=device)

        for t in range(T):
            if t < obs_length or prev_out is None:
                current_pos = nodes[:, t]
            elif torch.rand(1).item() < ss_prob:
                current_pos = prev_out[..., :2].detach()
            else:
                current_pos = nodes[:, t]

            if t == 0:
                e_temp = torch.zeros(B, N, 2, device=device)
            else:
                e_temp = prev_pos - current_pos
            e_spat_raw = (current_pos[:, self._spatial_src, :] -
                          current_pos[:, self._spatial_dst, :])

            te_in = self.temporal_edge_enc(e_temp)
            ht_f, ct_f = self.temporal_edge_rnn(
                te_in.reshape(B * N, -1),
                (h_temp.reshape(B * N, -1), c_temp.reshape(B * N, -1)))
            h_temp = ht_f.reshape(B, N, -1)
            c_temp = ct_f.reshape(B, N, -1)

            se_in = self.spatial_edge_enc(self._encode_spatial(e_spat_raw))
            hs_f, cs_f = self.spatial_edge_rnn(
                se_in.reshape(B * self.n_spatial, -1),
                (h_spat.reshape(B * self.n_spatial, -1),
                 c_spat.reshape(B * self.n_spatial, -1)))
            h_spat = hs_f.reshape(B, self.n_spatial, -1)
            c_spat = cs_f.reshape(B, self.n_spatial, -1)

            h_intra, h_inter, _, _, ent = self._attend(
                h_temp, h_spat, B, N, device)
            total_entropy = total_entropy + ent

            node_in = self.node_enc(current_pos)
            edge_in = self.edge_attn_enc(
                torch.cat([h_temp, h_intra, h_inter], dim=-1))
            rnn_in = torch.cat([node_in, edge_in], dim=-1)
            hn_f, cn_f = self.node_rnn(
                rnn_in.reshape(B * N, -1),
                (h_node.reshape(B * N, -1), c_node.reshape(B * N, -1)))
            h_node = hn_f.reshape(B, N, -1)
            c_node = cn_f.reshape(B, N, -1)

            out = self.output_linear(h_node)
            if self.residual:
                out = out.clone()
                out[..., :2] = out[..., :2] + current_pos
            outputs.append(out)

            prev_pos = current_pos.detach()
            prev_out = out

        avg_entropy = total_entropy / T
        return torch.stack(outputs, dim=1), avg_entropy

    @torch.no_grad()
    def predict(self, nodes, obs_length, mode="mean", n_samples=1):
        B, T, N, _ = nodes.shape
        device = nodes.device
        pred_length = T - obs_length

        def _decode_output(out):
            mux = out[..., 0]
            muy = out[..., 1]
            if mode == "mean":
                return torch.stack([mux, muy], dim=-1)
            sx = torch.exp(torch.clamp(out[..., 2], -6, 6))
            sy = torch.exp(torch.clamp(out[..., 3], -6, 6))
            corr = torch.tanh(out[..., 4])
            eps = torch.randn(B, N, 2, device=device)
            dx = sx * eps[..., 0]
            dy = (corr * sy * eps[..., 0] +
                  torch.sqrt((1 - corr ** 2).clamp(min=1e-6)) * sy * eps[..., 1])
            return torch.stack([mux + dx, muy + dy], dim=-1)

        def _run_one_pass(nodes_input):
            h_temp = torch.zeros(B, N, self.er, device=device)
            c_temp = torch.zeros(B, N, self.er, device=device)
            h_spat = torch.zeros(B, self.n_spatial, self.er, device=device)
            c_spat = torch.zeros(B, self.n_spatial, self.er, device=device)
            h_node = torch.zeros(B, N, self.nr, device=device)
            c_node = torch.zeros(B, N, self.nr, device=device)

            prev_pos = nodes_input[:, 0].clone()
            pred_positions = []
            pred_params_list = []
            attn_inter_list = []
            attn_intra_list = []

            for t in range(T - 1):
                if t < obs_length:
                    current_pos = nodes_input[:, t]
                else:
                    current_pos = next_predicted

                if t == 0:
                    e_temp = torch.zeros(B, N, 2, device=device)
                else:
                    e_temp = prev_pos - current_pos
                e_spat_raw = (current_pos[:, self._spatial_src, :] -
                              current_pos[:, self._spatial_dst, :])

                te_in = self.temporal_edge_enc(e_temp)
                ht_f, ct_f = self.temporal_edge_rnn(
                    te_in.reshape(B * N, -1),
                    (h_temp.reshape(B * N, -1), c_temp.reshape(B * N, -1)))
                h_temp = ht_f.reshape(B, N, -1)
                c_temp = ct_f.reshape(B, N, -1)

                se_in = self.spatial_edge_enc(
                    self._encode_spatial(e_spat_raw))
                hs_f, cs_f = self.spatial_edge_rnn(
                    se_in.reshape(B * self.n_spatial, -1),
                    (h_spat.reshape(B * self.n_spatial, -1),
                     c_spat.reshape(B * self.n_spatial, -1)))
                h_spat = hs_f.reshape(B, self.n_spatial, -1)
                c_spat = cs_f.reshape(B, self.n_spatial, -1)

                h_intra, h_inter, w_intra, w_inter, _ = \
                    self._attend(h_temp, h_spat, B, N, device)

                node_in = self.node_enc(current_pos)
                edge_in = self.edge_attn_enc(
                    torch.cat([h_temp, h_intra, h_inter], dim=-1))
                rnn_in = torch.cat([node_in, edge_in], dim=-1)
                hn_f, cn_f = self.node_rnn(
                    rnn_in.reshape(B * N, -1),
                    (h_node.reshape(B * N, -1), c_node.reshape(B * N, -1)))
                h_node = hn_f.reshape(B, N, -1)
                c_node = cn_f.reshape(B, N, -1)

                out = self.output_linear(h_node)
                if self.residual:
                    out = out.clone()
                    out[..., :2] = out[..., :2] + current_pos
                next_predicted = _decode_output(out)

                prev_pos = current_pos.clone()
                attn_inter_list.append(w_inter)
                attn_intra_list.append(w_intra)

                if t >= obs_length - 1:
                    pred_positions.append(next_predicted)
                    pred_params_list.append(out)

            pred_pos = torch.stack(pred_positions, dim=1)
            pred_params = torch.stack(pred_params_list, dim=1)
            return pred_pos, pred_params, attn_inter_list, attn_intra_list

        if mode == "mean" or n_samples <= 1:
            return _run_one_pass(nodes)

        all_preds = []
        last_params = None
        last_attn_inter = None
        last_attn_intra = None
        for _ in range(n_samples):
            pred_pos, pred_params, a_inter, a_intra = _run_one_pass(nodes)
            all_preds.append(pred_pos)
            last_params = pred_params
            last_attn_inter = a_inter
            last_attn_intra = a_intra
        pred_pos_all = torch.stack(all_preds, dim=0)
        return pred_pos_all, last_params, last_attn_inter, last_attn_intra


def build_edges_from_nodes(nodes, spatial_src, spatial_dst):
    """
    Compute temporal and spatial edge features from node positions.

    Parameters
    ----------
    nodes : (B, T, N, 2)
    spatial_src, spatial_dst : (E,) index tensors from the model

    Returns
    -------
    edges_temporal : (B, T, N, 2)
    edges_spatial  : (B, T, E, 2)
    """
    B, T, N, _ = nodes.shape
    device = nodes.device

    temporal = torch.zeros_like(nodes)
    temporal[:, 1:] = nodes[:, :-1] - nodes[:, 1:]

    src = spatial_src.to(device)
    dst = spatial_dst.to(device)
    spatial = nodes[:, :, src, :] - nodes[:, :, dst, :]

    return temporal, spatial
