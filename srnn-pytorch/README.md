# Social Attention for Multi-Mouse Trajectory Prediction

Adapted from [Social Attention: Modeling Attention in Human Crowds](https://arxiv.org/abs/1710.04689) (Vemula et al., ICRA 2018).
Original code: [cmubig/socialAttention](https://github.com/cmubig/socialAttention/tree/master/social-attention/srnn-pytorch)

## Key Contributions

- **Multi-keypoint graph**: Extended single-node-per-agent to 4 keypoints per mouse (nose, ear, center_back, tail_base), enabling body pose prediction and fine-grained attention analysis
- **Dual-stream Bahdanau attention**: Separate softmax for intra-mouse (body structure) and inter-mouse (social interaction) edges, replacing dot-product attention to resolve gradient vanishing at LSTM zero-initialization
- **Keypoint type embeddings + direction/log-distance spatial encoding**: Enables the model to distinguish edge semantics (e.g., nose→tail vs cb→cb)
- **10.4% improvement** in center_back trajectory prediction over single-keypoint baseline (12.53 vs 13.98 px)
- **Biologically meaningful attention patterns**: center_back as primary social reference; nose as intra-mouse attention hub; differential tail attention during chase events
- **Behavioral representation learning**: Hidden states support light condition classification (AUC=0.832) and chase detection (AUC=0.840)

## Results

Evaluated on MABe 2022 mouse triplet data (2783 test windows, 10-frame observation → 10-frame prediction at 7.5 fps).

| Method | CB ADE (px) | vs Static Baseline |
|--------|-------------|-------------------|
| Static (repeat last) | 17.63 | — |
| Constant velocity | 21.77 | — |
| MouseSRNN 1kp | 13.98 | +20.7% |
| **MouseSRNN 4kp (ours)** | **12.53** | **+29.0%** |

4kp outperforms 1kp at every prediction timestep (p < 1e-50, win rate 62%), with the gap growing from 0.4 px at t=1 to 1.9 px at t=10.

## Reproduction

### Setup

```bash
conda create -n sa python=3.8 && conda activate sa
pip install torch numpy matplotlib scikit-learn scipy wandb
```

### Data Preprocessing

```bash
python preprocess_mice.py \
  --window_size 20 --stride 10 --frame_skip 4 \
  --output ../data/mice/dataset_r1_w20_s10_fs4.npz
```

### Training

```bash
cd srnn && python train_mice.py \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --n_keypoints 4 --graph_type full \
  --num_epochs 100 --batch_size 16 \
  --learning_rate 0.001 --weight_decay 0.0 \
  --warmup_epochs 0 --ss_max 0.0 \
  --grad_clip 10.0 --lambda_dist 0.5 --lambda_attn 0.01 \
  --exp_tag v3_4kp_full
```

### Evaluation

```bash
python evaluate_mice.py \
  --checkpoint ../save/mice/v3_4kp_full/best_model.tar \
  --split test --mode mean
```

### Visualization

```bash
python visualize_attn_pairwise.py \
  --checkpoint ../save/mice/v3_4kp_full/best_model.tar \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --auto_select --n_best 6 --n_active 3
```

Generates per sample: trajectory plot, 12×12 attention heatmap, mouse-level attention heatmap, and per-node intra/inter attention figures.

## Citation

```bibtex
@inproceedings{VMO2018-SocialAttention,
   author = {Vemula, Anirudh and Muelling, Katharina and Oh, Jean},
   title = {Social Attention: Modeling Attention in Human Crowds},
   booktitle = {Proceedings of IEEE International Conference on Robotics and Automation (ICRA)},
   year = {2018}
}
```
