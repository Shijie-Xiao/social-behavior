# Social Attention for Multi-Mouse Trajectory Prediction

Adapted from [Social Attention: Modeling Attention in Human Crowds](https://arxiv.org/abs/1710.04689) (Vemula et al., ICRA 2018).  
Original code: [cmubig/socialAttention](https://github.com/cmubig/socialAttention/tree/master/social-attention/srnn-pytorch)

## Key contributions

- **Multi-keypoint graph**: Up to four keypoints per mouse (nose, ear, center_back, tail_base) for pose-aware prediction and attention analysis.
- **Dual-stream Bahdanau attention**: Separate softmax over intra- vs inter-mouse edges; replaces dot-product attention to avoid vanishing gradients at LSTM cold start.
- **Keypoint embeddings + direction / log-distance encoding**: Distinguishes edge semantics (e.g. nose→tail vs center_back→center_back).
- **~10% lower center_back ADE** than the single-keypoint baseline on the held-out split (see table below).
- **Interpretable attention**: center_back as a social anchor; nose as intra-mouse hub; tail weight shifts during chase-like motion.
- **Behavioral probes**: Observation-only hidden states reach about **AUC 0.77** on light condition after fixing a prior leakage bug (older ~0.83 numbers used future frames). Chase labels are extremely rare; balanced logistic regression on raw kinematics often matches or beats shallow probes on embeddings alone.

## Results

MABe 2022 triplet data, 10-frame observation → 10-frame prediction (7.5 Hz after frame skip).

| Method | CB ADE (px) | vs static baseline |
|--------|---------------|---------------------|
| Static (repeat last) | 17.63 | — |
| Constant velocity | 21.77 | — |
| MouseSRNN 1kp | 13.98 | +20.7% |
| **MouseSRNN 4kp (ours)** | **12.53** | **+29.0%** |

Use `python evaluate.py` (calls `net.predict()` with rolling edges) for all reported metrics; ad-hoc autoregressive loops must stay aligned with ground-truth time indexing.

## Reproduction

### Environment

```bash
conda create -n sa python=3.8 && conda activate sa
pip install torch numpy matplotlib scikit-learn scipy wandb
```

### Preprocess

```bash
python preprocess.py \
  --window_size 20 --stride 10 --frame_skip 4 \
  --output ../data/mice/dataset_r1_w20_s10_fs4.npz
```

### Train (v3 4kp full, aligned defaults)

```bash
cd srnn && python train.py \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --n_keypoints 4 --graph_type full \
  --num_epochs 100 --batch_size 16 \
  --learning_rate 0.001 --weight_decay 0.0 \
  --warmup_epochs 0 --ss_max 0.0 \
  --grad_clip 10.0 --lambda_dist 0.5 --lambda_attn 0.01 \
  --exp_tag v3_4kp_full
```

Checkpoints are written under `save/mice/<exp_tag>/best_model.tar`. For a stable local path, copy the best file to `checkpoints/best_model.tar` (directory tracked via `.gitkeep`; `*.tar` is gitignored).

### Evaluate

```bash
cd srnn && python evaluate.py \
  --checkpoint ../checkpoints/best_model.tar \
  --split test --mode mean
```

### Attention / trajectory figures

```bash
cd srnn && python visualize_attn.py \
  --checkpoint ../checkpoints/best_model.tar \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --auto_select --n_best 6 --n_active 3
```

## Repository layout (what to commit)

Track: `README.md`, `.gitignore`, `preprocess.py`, `srnn/*.py`, `checkpoints/.gitkeep`.  
Ignore (see `.gitignore`): `wandb/`, `save/`, `log/`, `data/`, `__pycache__/`, `*.npz`, `*.tar`, `*.pkl`, `*.png`, and other generated artifacts.

## Citation

```bibtex
@inproceedings{VMO2018-SocialAttention,
   author = {Vemula, Anirudh and Muelling, Katharina and Oh, Jean},
   title = {Social Attention: Modeling Attention in Human Crowds},
   booktitle = {Proceedings of IEEE International Conference on Robotics and Automation (ICRA)},
   year = {2018}
}
```
