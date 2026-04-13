# Social Attention for Mouse Trajectory Prediction

Adapted from [Social Attention: Modeling Attention in Human Crowds](https://arxiv.org/abs/1710.04689) (Vemula et al., ICRA 2018).
Original code: [BenJamesbabala/srnn-pytorch](https://github.com/BenJamesbabala/srnn-pytorch)

This repository extends the Structural RNN with social attention mechanism from **human crowd trajectory prediction** to **multi-mouse social behavior analysis**, with several architectural improvements for multi-keypoint body pose modeling.

## Overview

The model predicts future trajectories of 3 interacting mice (each represented by up to 4 body keypoints) given their observed trajectories. It learns a structured graph representation where:
- **Temporal edges** encode each keypoint's motion history
- **Spatial edges** encode pairwise relationships between keypoints (both within and across mice)
- **Dual-stream attention** separately models intra-mouse body structure and inter-mouse social interactions
- **Node RNN** integrates temporal, spatial, and attention information to predict future positions as bivariate Gaussian distributions

## Key Improvements over Original Social Attention

| Component | Original (Human Crowds) | Ours (Mouse Keypoints) |
|-----------|------------------------|----------------------|
| **Graph structure** | One node per person, fully connected | Configurable: 1/2/3/4 keypoints per mouse, `full` (intra+inter) or `inter`-only graph |
| **Spatial edge encoding** | Raw 2D displacement | Direction + log-distance + learnable keypoint type embeddings (src & dst) |
| **Attention mechanism** | Dot-product with non-standard temperature scaling | **Additive (Bahdanau) attention** with separate softmax for intra-mouse and inter-mouse edges |
| **Attention regularization** | None | Entropy regularization (`lambda_attn`) to prevent uniform attention collapse |
| **Loss function** | Bivariate Gaussian NLL | NLL + body distance loss (`lambda_dist`) + activity-based sample weighting + attention entropy |
| **Training** | Fixed teacher forcing | Support for scheduled sampling and learning rate warmup (disabled in best config) |
| **Evaluation** | ADE/FDE | ADE/FDE + center_back ADE (cross-config comparable) + body structure error + inter-mouse distance error + baselines (static/linear/constant-velocity) |

### Why These Changes?

1. **Dot-product → Additive attention**: The original dot-product attention suffers from gradient vanishing when LSTM hidden states are zero-initialized. Additive attention uses `score = v^T tanh(Q + K)` which provides gradient flow even at initialization, improving attention weight differentiation by 38-92x.

2. **Dual-stream attention**: In a full graph with 12 nodes (3 mice × 4 keypoints), a single softmax over all 11 neighbors causes inter-mouse attention signals to be drowned out by the more numerous intra-mouse edges. Separate softmax for intra (3 same-mouse neighbors) and inter (8 other-mouse neighbors) forces the model to learn both body structure and social interaction.

3. **Keypoint type embeddings**: The original spatial edge input is a 2D displacement vector, which is identical regardless of whether the edge connects two noses or a nose and a tail. Adding learnable embeddings for source and destination keypoint types allows the model to distinguish edge semantics.

4. **Direction + log-distance**: Replacing raw displacement with `[cos θ, sin θ, log(1 + d)]` decouples direction from distance, preventing close-range interactions from dominating the representation.

## Data Preprocessing

```bash
python preprocess_mice.py \
  --data_dir ../data/mice/raw \
  --window_size 20 --stride 10 --frame_skip 4 \
  --keypoints nose left_ear center_back tail_base \
  --output ../data/mice/dataset_r1_w20_s10_fs4.npz
```

- **Window**: 20 frames (10 observed + 10 predicted) at 7.5 fps (after 4× frame skip from 30 fps)
- **Stride**: 10 frames between windows
- **Keypoints**: 4 per mouse — nose, left_ear, center_back, tail_base
- **Normalization**: Coordinates normalized to [0, 1] by arena size (450 px)
- **Metadata**: Each window includes `activity`, `lights` (0/1), `chase` (0/1) labels
- **Splits**: Train / Val / Test with stratified sampling

## Training

### Best Configuration (v3_4kp_full)

```bash
cd srnn && python train_mice.py \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --n_keypoints 4 --graph_type full \
  --num_epochs 100 --batch_size 16 \
  --learning_rate 0.001 --weight_decay 0.0 \
  --warmup_epochs 0 --ss_max 0.0 \
  --grad_clip 10.0 --lambda_dist 0.5 --lambda_attn 0.01 \
  --eval_every 10 --eval_batches 10 --save_every 20 \
  --num_workers 0 --arena_px 450 \
  --exp_tag v3_4kp_full --wandb_project MS_mice
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_keypoints` | 4 | nose, left_ear, center_back, tail_base |
| `graph_type` | full | Both intra-mouse and inter-mouse edges |
| `batch_size` | 16 | |
| `learning_rate` | 1e-3 | Adam optimizer |
| `weight_decay` | 0.0 | No L2 regularization |
| `warmup_epochs` | 0 | No warmup |
| `ss_max` | 0.0 | No scheduled sampling |
| `grad_clip` | 10.0 | Gradient norm clipping |
| `lambda_dist` | 0.5 | Body distance loss weight |
| `lambda_attn` | 0.01 | Attention entropy regularization weight |

## Evaluation

```bash
python evaluate_mice.py \
  --checkpoint ../save/mice/v3_4kp_full/best_model.tar \
  --split test --mode mean --arena_px 450
```

### Results (test set, `evaluate_mice.py --mode mean`)

**v3_4kp_full (best configuration):**

| Metric | Value |
|--------|-------|
| **CB ADE** | **12.53 px** (1.39 cm) |
| CB FDE | 23.84 px (2.65 cm) |
| All-node ADE | 14.65 px (1.63 cm) |
| All-node FDE | 26.87 px (2.99 cm) |


### Baseline Comparison (center_back ADE)

| Method | CB ADE (px) | Improvement |
|--------|-------------|-------------|
| Static (repeat last frame) | 17.63 | — |
| Constant velocity | 21.77 | — |
| Linear extrapolation | 22.02 | — |
| **MouseSRNN 4kp (ours)** | **12.53** | **+29.0% vs static** |
| MouseSRNN 1kp | 13.98 | +20.7% vs static |

### 1kp vs 4kp Comparison

Both models trained with identical hyperparameters, evaluated with the same `evaluate_mice.py` pipeline (calls `net.predict()`). Static baselines are identical (17.63 px), confirming fair comparison on the same test data.

| Metric | 1kp (3 nodes) | **4kp (12 nodes)** | Δ |
|--------|--------------|-------------------|---|
| **CB ADE (px)** | 13.98 | **12.53** | **-1.45 (-10.4%)** |
| **CB FDE (px)** | 25.77 | **23.84** | **-1.93 (-7.5%)** |
| All-node ADE (px) | 13.98 | 14.65 | +0.67 |

The 4-keypoint model improves center_back prediction by 10.4% over the single-keypoint baseline, demonstrating that body pose context (head direction, body curvature) helps predict the body center trajectory. The all-node ADE is slightly higher for 4kp because it additionally predicts noisier peripheral keypoints (nose, tail_base), but the primary CB metric and inter-mouse distance error both improve. Beyond prediction accuracy, the 4kp model uniquely provides complete body pose prediction and fine-grained attention analysis for social behavior interpretation.

## Visualization

### Attention & Trajectory Figures

```bash
# Auto-select best, active, and chase samples
python visualize_attn_pairwise.py \
  --checkpoint ../save/mice/v3_4kp_full/best_model.tar \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --auto_select --n_best 6 --n_active 3

# Specific samples
python visualize_attn_pairwise.py \
  --checkpoint ../save/mice/v3_4kp_full/best_model.tar \
  --data ../data/mice/dataset_r1_w20_s10_fs4.npz \
  --sample_idx 1085 65 960
```

Per sample generates:
- `trajectory.png` — CB trajectory prediction (GT solid + pred dashed, original Social Attention style)
- `heatmap_12x12.png` — Full 12×12 node-level attention matrix
- `heatmap_mouse.png` — 3×3 mouse-level mean attention
- `intra/M{i}_{kp}.png` — 12 figures: each node's attention to same-mouse keypoints
- `inter/M{i}_{kp}.png` — 12 figures: each node's attention to other-mouse keypoints

### Representation Analysis

Hidden state embeddings from the trained model encode behaviorally meaningful information:

| Task | Best Feature | AUC-ROC |
|------|-------------|---------|
| Light condition (ON/OFF) classification | Global node state (128d) | 0.832 |
| Chase event detection | Temporal trajectory (640d) | 0.840 |
| Raw feature baseline (activity + speed + distance) | 3d | 0.623 |

### Attention Patterns

Key findings from attention analysis:
- **Inter-mouse attention** concentrates ~95% on target mice's **center_back** keypoint (body centroid)
- **Intra-mouse attention**: all keypoints strongly attend to **nose** (motion direction indicator), nose attends to **tail_base** (body axis)
- **Chase events** show distinct patterns: pursuing mouse allocates attention to target's **tail_base** (direction of escape), unlike normal interactions
- Light ON vs OFF: attention entropy is significantly higher during lights-on (p < 1e-12), indicating more distributed social attention

## File Structure

```
srnn-pytorch/
├── preprocess_mice.py          # MABe data → .npz sliding window dataset
├── data/mice/                  # Preprocessed datasets
│   └── dataset_r1_w20_s10_fs4.npz
├── srnn/
│   ├── model_mice.py           # MouseSRNN model (dual-stream Bahdanau attention)
│   ├── train_mice.py           # Training script
│   ├── evaluate_mice.py        # Evaluation with metrics & baselines
│   ├── visualize_attn_pairwise.py  # Attention & trajectory visualization
│   ├── mouse_dataset.py        # PyTorch Dataset & DataLoader
│   └── criterion_mice.py       # Bivariate Gaussian NLL + auxiliary losses
├── save/mice/
│   └── v3_4kp_full/            # Best model checkpoint & plots
│       ├── best_model.tar
│       └── plots/
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, Matplotlib, scikit-learn
- wandb (optional, for experiment tracking)

## Citation

If you use this code, please cite the original Social Attention paper:

```bibtex
@inproceedings{VMO2018-SocialAttention,
   author = {Vemula, Anirudh and Muelling, Katharina and Oh, Jean},
   title = {Social Attention: Modeling Attention in Human Crowds},
   booktitle = {Proceedings of IEEE International Conference on Robotics and Automation (ICRA)},
   year = {2018}
}
```

**License**: GPL v3
