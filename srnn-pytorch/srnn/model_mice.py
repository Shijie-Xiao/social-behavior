"""
Batched SRNN for fixed-topology mouse graph (12 nodes, fully connected).

The original SRNN uses Python for-loops over nodes/edges/frames because the
graph topology varies per sample.  Our mouse graph is *fixed*: 3 mice × 4
keypoints = 12 nodes, all always present, fully connected.  This lets us
replace every loop with batch tensor ops → ~100× faster.

Architecture (per time-step):
  1. Temporal edge RNN:  edge(i,i) encodes displacement from t-1 to t
  2. Spatial edge RNN:   edge(i,j) encodes relative position at t
  3. Attention:          for each node i, attend over its 11 spatial edges
                         using its temporal edge hidden state as query
  4. Node RNN:           combine (position, temporal_h, spatial_attn) → LSTM
  5. Output:             linear → 5-d Gaussian parameters (mux, muy, sx, sy, corr)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_NODES = 12
N_SPATIAL = N_NODES * (N_NODES - 1)  # 132 directed spatial edges


def _build_spatial_index():
    """
    Pre-compute index tensors for spatial edges.
    Returns:
        src: (132,) source node of each spatial edge
        dst: (132,) destination node
        node_to_edges: (12, 11) for each node, indices into the 132 spatial edges
                       where that node is the source
    """
    src, dst = [], []
    node_to_edges = [[] for _ in range(N_NODES)]
    idx = 0
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i != j:
                src.append(i)
                dst.append(j)
                node_to_edges[i].append(idx)
                idx += 1
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(node_to_edges, dtype=torch.long),  # (12, 11)
    )


_SPATIAL_SRC, _SPATIAL_DST, _NODE_TO_EDGES = _build_spatial_index()


class MouseSRNN(nn.Module):
    def __init__(self, args, infer=False):
        super().__init__()
        self.infer = infer

        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length

        # Sizes
        nr = args.human_node_rnn_size
        er = args.human_human_edge_rnn_size
        ne = args.human_node_embedding_size
        ee = args.human_human_edge_embedding_size
        attn_size = args.attention_size
        drop = args.dropout

        self.nr = nr
        self.er = er

        # Edge RNNs
        self.temporal_edge_enc = nn.Sequential(
            nn.Linear(2, ee), nn.ReLU(), nn.Dropout(drop))
        self.temporal_edge_rnn = nn.LSTMCell(ee, er)

        self.spatial_edge_enc = nn.Sequential(
            nn.Linear(2, ee), nn.ReLU(), nn.Dropout(drop))
        self.spatial_edge_rnn = nn.LSTMCell(ee, er)

        # Attention
        self.attn_q = nn.Linear(er, attn_size)  # query from temporal edge
        self.attn_k = nn.Linear(er, attn_size)  # key from spatial edges

        # Node RNN
        self.node_enc = nn.Sequential(
            nn.Linear(2, ne), nn.ReLU(), nn.Dropout(drop))
        self.edge_attn_enc = nn.Sequential(
            nn.Linear(er * 2, ne), nn.ReLU(), nn.Dropout(drop))
        self.node_rnn = nn.LSTMCell(ne * 2, nr)

        # Output
        self.output_linear = nn.Linear(nr, 5)

    def forward(self, nodes, edges_temporal, edges_spatial):
        """
        Fully batched forward pass.

        Parameters
        ----------
        nodes : (B, T, 12, 2)
        edges_temporal : (B, T, 12, 2)   displacement from t-1 to t for each node
        edges_spatial  : (B, T, 132, 2)  relative position vectors for each spatial edge

        Returns
        -------
        outputs : (B, T, 12, 5)
        """
        B, T, N, _ = nodes.shape
        device = nodes.device

        # Move index tensors to device
        spatial_src = _SPATIAL_SRC.to(device)
        spatial_dst = _SPATIAL_DST.to(device)
        node_to_edges = _NODE_TO_EDGES.to(device)  # (12, 11)

        # Initialize hidden states
        h_temp = torch.zeros(B, N, self.er, device=device)
        c_temp = torch.zeros(B, N, self.er, device=device)
        h_spat = torch.zeros(B, N_SPATIAL, self.er, device=device)
        c_spat = torch.zeros(B, N_SPATIAL, self.er, device=device)
        h_node = torch.zeros(B, N, self.nr, device=device)
        c_node = torch.zeros(B, N, self.nr, device=device)

        outputs = []

        for t in range(T):
            # --- 1. Temporal edge RNN: (B, 12, 2) → (B, 12, er) ---
            te_in = self.temporal_edge_enc(edges_temporal[:, t])  # (B, 12, ee)
            te_flat = te_in.reshape(B * N, -1)
            ht_flat, ct_flat = self.temporal_edge_rnn(
                te_flat, (h_temp.reshape(B * N, -1), c_temp.reshape(B * N, -1))
            )
            h_temp = ht_flat.reshape(B, N, -1)
            c_temp = ct_flat.reshape(B, N, -1)

            # --- 2. Spatial edge RNN: (B, 132, 2) → (B, 132, er) ---
            se_in = self.spatial_edge_enc(edges_spatial[:, t])  # (B, 132, ee)
            se_flat = se_in.reshape(B * N_SPATIAL, -1)
            hs_flat, cs_flat = self.spatial_edge_rnn(
                se_flat, (h_spat.reshape(B * N_SPATIAL, -1),
                          c_spat.reshape(B * N_SPATIAL, -1))
            )
            h_spat = hs_flat.reshape(B, N_SPATIAL, -1)
            c_spat = cs_flat.reshape(B, N_SPATIAL, -1)

            # --- 3. Attention: for each node, attend over its 11 spatial edges ---
            # Query: temporal hidden of each node → (B, 12, attn)
            q = self.attn_q(h_temp)  # (B, 12, attn)

            # Keys: spatial hidden states gathered per node → (B, 12, 11, attn)
            h_spat_per_node = h_spat[:, node_to_edges, :]  # (B, 12, 11, er)
            k = self.attn_k(h_spat_per_node)  # (B, 12, 11, attn)

            # Scaled dot-product attention (original socialAttention scaling)
            scores = torch.einsum("bna,bnka->bnk", q, k)  # (B, 12, 11)
            num_keys = k.size(2)  # 11
            temperature = num_keys / np.sqrt(q.size(-1))
            attn_weights = F.softmax(scores * temperature, dim=-1)  # (B, 12, 11)

            # Weighted sum of spatial hidden states
            h_spatial_attn = torch.einsum(
                "bnk,bnke->bne", attn_weights, h_spat_per_node
            )  # (B, 12, er)

            # --- 4. Node RNN ---
            node_in = self.node_enc(nodes[:, t])  # (B, 12, ne)
            edge_in = self.edge_attn_enc(
                torch.cat([h_temp, h_spatial_attn], dim=-1)
            )  # (B, 12, ne)
            rnn_in = torch.cat([node_in, edge_in], dim=-1)  # (B, 12, 2*ne)

            rnn_flat = rnn_in.reshape(B * N, -1)
            hn_flat, cn_flat = self.node_rnn(
                rnn_flat, (h_node.reshape(B * N, -1), c_node.reshape(B * N, -1))
            )
            h_node = hn_flat.reshape(B, N, -1)
            c_node = cn_flat.reshape(B, N, -1)

            # --- 5. Output ---
            out = self.output_linear(h_node)  # (B, 12, 5)
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # (B, T, 12, 5)

    @torch.no_grad()
    def predict(self, nodes, obs_length, mode="mean", n_samples=1):
        """
        Autoregressive prediction following the original socialAttention
        sampling approach.

        Key design (matching original paper): output[t] predicts position[t+1].

        Steps 0..obs_length-2 : feed GT, discard output
        Step  obs_length-1     : feed GT, output → predicted pos[obs_length]
        Step  obs_length       : feed predicted, output → predicted pos[obs_length+1]
        ...
        Step  T-2              : output → predicted pos[T-1]

        Parameters
        ----------
        nodes : (B, T, 12, 2) — full window with GT positions
        obs_length : int
        mode : 'mean' (use Gaussian mean) or 'sample' (sample from Gaussian)
        n_samples : int — number of stochastic samples (only when mode='sample')

        Returns
        -------
        pred_nodes  : (B, pred_len, 12, 2) or (n_samples, B, pred_len, 12, 2)
        pred_params : (B, pred_len, 12, 5) — Gaussian params
        attn_all    : list of (B, 12, 11) attention weights per timestep
        """
        B, T, N, _ = nodes.shape
        device = nodes.device
        pred_length = T - obs_length

        def _decode_output(out):
            """Extract predicted next position from Gaussian output."""
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
            spatial_src = _SPATIAL_SRC.to(device)
            spatial_dst = _SPATIAL_DST.to(device)
            node_to_edges = _NODE_TO_EDGES.to(device)

            h_temp = torch.zeros(B, N, self.er, device=device)
            c_temp = torch.zeros(B, N, self.er, device=device)
            h_spat = torch.zeros(B, N_SPATIAL, self.er, device=device)
            c_spat = torch.zeros(B, N_SPATIAL, self.er, device=device)
            h_node = torch.zeros(B, N, self.nr, device=device)
            c_node = torch.zeros(B, N, self.nr, device=device)

            prev_pos = nodes_input[:, 0].clone()
            pred_positions = []
            pred_params_list = []
            attn_list = []

            # Process T-1 steps: output[t] predicts position[t+1]
            for t in range(T - 1):
                if t < obs_length:
                    current_pos = nodes_input[:, t]
                else:
                    current_pos = next_predicted  # from previous step

                # Edges
                if t == 0:
                    e_temp = torch.zeros(B, N, 2, device=device)
                else:
                    e_temp = prev_pos - current_pos
                e_spat = (current_pos[:, spatial_src, :] -
                          current_pos[:, spatial_dst, :])

                # 1. Temporal edge RNN
                te_in = self.temporal_edge_enc(e_temp)
                ht_flat, ct_flat = self.temporal_edge_rnn(
                    te_in.reshape(B * N, -1),
                    (h_temp.reshape(B * N, -1), c_temp.reshape(B * N, -1)))
                h_temp = ht_flat.reshape(B, N, -1)
                c_temp = ct_flat.reshape(B, N, -1)

                # 2. Spatial edge RNN
                se_in = self.spatial_edge_enc(e_spat)
                hs_flat, cs_flat = self.spatial_edge_rnn(
                    se_in.reshape(B * N_SPATIAL, -1),
                    (h_spat.reshape(B * N_SPATIAL, -1),
                     c_spat.reshape(B * N_SPATIAL, -1)))
                h_spat = hs_flat.reshape(B, N_SPATIAL, -1)
                c_spat = cs_flat.reshape(B, N_SPATIAL, -1)

                # 3. Attention (original socialAttention scaling)
                q = self.attn_q(h_temp)
                h_spat_per_node = h_spat[:, node_to_edges, :]
                k = self.attn_k(h_spat_per_node)
                scores = torch.einsum("bna,bnka->bnk", q, k)
                num_keys = k.size(2)  # 11
                temperature = num_keys / np.sqrt(q.size(-1))
                attn_weights = F.softmax(scores * temperature, dim=-1)
                h_spatial_attn = torch.einsum(
                    "bnk,bnke->bne", attn_weights, h_spat_per_node)

                # 4. Node RNN
                node_in = self.node_enc(current_pos)
                edge_in = self.edge_attn_enc(
                    torch.cat([h_temp, h_spatial_attn], dim=-1))
                rnn_in = torch.cat([node_in, edge_in], dim=-1)
                hn_flat, cn_flat = self.node_rnn(
                    rnn_in.reshape(B * N, -1),
                    (h_node.reshape(B * N, -1), c_node.reshape(B * N, -1)))
                h_node = hn_flat.reshape(B, N, -1)
                c_node = cn_flat.reshape(B, N, -1)

                # 5. Output: predicts position[t+1]
                out = self.output_linear(h_node)  # (B, 12, 5)
                next_predicted = _decode_output(out)

                prev_pos = current_pos.clone()
                attn_list.append(attn_weights)

                # Collect predictions for future frames (t+1 >= obs_length)
                if t >= obs_length - 1:
                    pred_positions.append(next_predicted)
                    pred_params_list.append(out)

            pred_pos = torch.stack(pred_positions, dim=1)
            pred_params = torch.stack(pred_params_list, dim=1)
            return pred_pos, pred_params, attn_list

        if mode == "mean" or n_samples <= 1:
            return _run_one_pass(nodes)

        all_preds = []
        last_params = None
        last_attn = None
        for _ in range(n_samples):
            pred_pos, pred_params, attn_list = _run_one_pass(nodes)
            all_preds.append(pred_pos)
            last_params = pred_params
            last_attn = attn_list
        pred_pos_all = torch.stack(all_preds, dim=0)
        return pred_pos_all, last_params, last_attn


def build_edges_from_nodes(nodes):
    """
    Compute temporal and spatial edge features from node positions.

    Parameters
    ----------
    nodes : (B, T, 12, 2)

    Returns
    -------
    edges_temporal : (B, T, 12, 2)   — displacement from previous frame
    edges_spatial  : (B, T, 132, 2)  — relative position vectors
    """
    B, T, N, _ = nodes.shape
    device = nodes.device

    # Temporal edges: nodes[t-1] - nodes[t]
    temporal = torch.zeros_like(nodes)
    temporal[:, 1:] = nodes[:, :-1] - nodes[:, 1:]

    # Spatial edges: nodes[src] - nodes[dst]
    src = _SPATIAL_SRC.to(device)
    dst = _SPATIAL_DST.to(device)
    spatial = nodes[:, :, src, :] - nodes[:, :, dst, :]  # (B, T, 132, 2)

    return temporal, spatial
