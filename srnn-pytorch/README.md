# Social Attention for Multi-Mouse Trajectory Prediction

Adapted from [Social Attention (Vemula et al., ICRA 2018)](https://arxiv.org/abs/1710.04689). Original code: [cmubig/socialAttention](https://github.com/cmubig/socialAttention/tree/master/social-attention/srnn-pytorch).

## Model Architecture

**MouseSRNN** predicts future keypoints for multiple mice given observed trajectories. The architecture has four stages per time step:

```
Observed positions (B, T, N, 2)
        │
  ┌─────┴──────┐
  │ Edge RNNs  │  Temporal edges: Δposition between consecutive frames
  │            │  Spatial edges:  relative position between connected keypoints
  └─────┬──────┘
        │ h_temp (B, N, er), h_spat (B, E, er)
  ┌─────┴──────────┐
  │ Dual-stream    │  Query: Q(h_temp)   Keys: K_intra(h_spat), K_inter(h_spat)
  │ Bahdanau Attn  │  Score = v^T tanh(Q + K)
  │ + Clamp+Temp   │  s = clamp(score, -C, C)
  │                │  w = softmax(s / τ)
  └─────┬──────────┘
        │ h_intra_attn, h_inter_attn
  ┌─────┴──────┐
  │ Node RNN   │  Input: [pos_encoding, h_temp, h_intra, h_inter]
  │ + Residual  │  Output: Δposition (residual) or bivariate Gaussian params
  └─────────────┘
```

### Key components

**Multi-keypoint graph.** Each mouse has up to 4 keypoints (nose, ear, center_back, tail_base). A `full` graph connects every keypoint pair with directed spatial edges (12 nodes, 132 edges).

**Dual-stream attention.** Intra-mouse edges (same mouse, different keypoints) and inter-mouse edges (different mice) attend separately with independent softmax, then their weighted context vectors are concatenated.

**Score clamping + temperature.** The raw Bahdanau score `v^T tanh(Q+K)` is clamped to `[-C, C]` (default C=5) and divided by temperature τ (default τ_inter=2, τ_intra=1) before softmax. This prevents the learned score vector `v` from expanding unboundedly, which would collapse attention into hard argmax and destroy interpretability.

**Keypoint-type embeddings.** A learnable encoding distinguishes edge types (e.g. nose→tail vs cb→cb), injected into the spatial edge encoder.

**Direction + log-distance encoding.** Each spatial edge encodes both the direction vector and `log(‖Δ‖ + 1)`, providing the model with explicit geometric information.

### Inference modes

| Mode | Description |
|------|-------------|
| Teacher forcing | Ground-truth positions as input at every step (training) |
| Scheduled sampling | Mix of ground-truth and model predictions (training, controlled by `ss_max`) |
| Autoregressive | Model predictions fed back as input (evaluation, via `net.predict()`) |

## Results

MABe 2022 Mouse Triplets, 10 obs → 10 pred frames (7.5 Hz). Full test set (2783 windows).

| Method | ADE (px) | cbADE (px) |
|--------|----------|------------|
| Static (repeat last) | 18.56 | 17.63 |
| Constant velocity | 24.87 | 21.77 |
| MouseSRNN 1-keypoint | 15.78 | 13.98 |
| MouseSRNN 4-keypoint | 14.42 | 12.33 |

### Attention interpretability

| Metric | Without clamp+temp | With clamp+temp |
|--------|:---:|:---:|
| Inter-attention entropy (% of uniform) | 2.9% | 40.6% |
| Hard argmax fraction (>99% weight) | 84.4% | 0% |
| Closest-keypoint match accuracy | 33.6% | 46.9% |
| \|v_inter\| norm | 16.27 | 1.41 |

### Learned attention patterns

- **Tail preference**: Inter-attention strongly favors the tail keypoint of other mice (weight ≈ 0.47), consistent with tracking and sniffing behavior.
- **Dynamic redistribution**: During sudden turns, inter-attention flattens toward uniform across targets, expressing uncertainty rather than committing to a single target.
- **Heading encoding**: Intra-attention nose-weight rises from ~0.03 to ~0.29 during sharp turns, encoding heading direction changes.
- **Distance adaptivity**: Close-range attention is distributed across multiple targets; far-range attention concentrates on a single anchor.

## Reproduction

### Environment

```bash
conda create -n sa python=3.8 && conda activate sa
pip install torch numpy matplotlib scikit-learn scipy wandb
```

### Data

This project uses **MABe 2022 Mouse Triplets** keypoint data.

- **Challenge hub:** [MABe 2022: Mouse Triplets — AIcrowd](https://www.aicrowd.com/challenges/mabe-2022-mouse-triplets)

Place the raw files under `data/mice/`:

| File | Purpose |
|------|---------|
| `user_train_r1.npy` | Round-1 source dict |
| `user_train.npy` | General user train source |

### Preprocess

```bash
cd srnn-pytorch && python preprocess.py \
  --window_size 20 --stride 10 --fs 4 \
  --out_dir data/mice
```

Writes `dataset_*_w20_s10_fs4.npz` under `data/mice/`.

### Train

```bash
cd srnn && python train.py \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --n_keypoints 4 --graph_type full \
  --num_epochs 100 --batch_size 16 \
  --learning_rate 0.001 --weight_decay 0.0 \
  --warmup_epochs 0 --ss_max 0.0 \
  --grad_clip 10.0 --lambda_dist 0.5 --lambda_attn 0.01 \
  --attn_clamp 5.0 --attn_temp_intra 1.0 --attn_temp_inter 2.0 \
  --no_wandb \
  --exp_tag experiment
```

Key attention hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--attn_clamp` | 5.0 | Clamp attention logits to [-C, C]; set ≤0 to disable |
| `--attn_temp_intra` | 1.0 | Softmax temperature for intra-mouse attention |
| `--attn_temp_inter` | 1.0 | Softmax temperature for inter-mouse attention |
| `--lambda_attn` | 0.01 | Weight for attention entropy regularization loss |

Checkpoints: `best_model.tar` (lowest val loss), `best_ade_model.tar` (lowest ADE).

### Evaluate

```bash
cd srnn && python evaluate.py \
  --checkpoint ../save/mice/<exp_tag>/best_ade_model.tar \
  --split test --mode mean
```

### Visualize attention

```bash
cd srnn && python visualize_attn.py \
  --checkpoint ../save/mice/<exp_tag>/best_ade_model.tar \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --auto_select --n_best 6 --n_active 3
```

## Citation

```bibtex
@inproceedings{VMO2018-SocialAttention,
   author = {Vemula, Anirudh and Muelling, Katharina and Oh, Jean},
   title = {Social Attention: Modeling Attention in Human Crowds},
   booktitle = {Proceedings of IEEE International Conference on Robotics and Automation (ICRA)},
   year = {2018}
}
```
