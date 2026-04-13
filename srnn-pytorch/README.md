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

### Data source (MABe 2022)

This project expects **MABe 2022 Mouse Triplets** keypoint `.npy` archives (30 Hz tracking, three mice per clip).

- **Challenge hub (files, rules, Resources):** [MABe 2022: Mouse Triplets — AIcrowd](https://www.aicrowd.com/challenges/mabe-2022-mouse-triplets)  
  Download the training keypoint bundles from the challenge **Resources** tab (you may need an AIcrowd account). The starter materials also describe layout and `aicrowd` CLI downloads, e.g. [Getting Started — Round 1](https://www.aicrowd.com/showcase/getting-started-mabe-2022-mouse-triplets-round-1).
- **Related challenge (videos + larger file set):** [MABe 2022: Mouse Triplets — Video Data](https://www.aicrowd.com/challenges/mabe-2022-mouse-triplets-video-data)  
  Keypoint-only stages still refer to the same triplet task; use whichever release matches your filenames below.

### Data layout (raw → preprocessed)

All paths below are relative to the **`srnn-pytorch/`** directory (where `preprocess.py` lives).

**1. Raw inputs (you must place these; they are not downloaded by the scripts)**

| File | Purpose |
|------|---------|
| `data/mice/user_train_r1.npy` | Round‑1 / split “r1” source dict (`sequences`, keypoints, …) |
| `data/mice/user_train.npy` | “r2” / general user train source |

If a file is missing, preprocessing **skips** that split; if both are missing, `preprocess.py` exits with an error. Filenames are defined in `preprocess.py` (`DATASET_META`); adjust that table if your downloads use different names.

**2. Preprocessed outputs (directory is created automatically)**

Running `preprocess.py` creates `data/mice/` if needed (`mkdir(parents=True, exist_ok=True)`) and writes tagged `.npz` files, e.g.:

- `dataset_r1_w20_s10_fs4.npz`
- `dataset_r2_w20_s10_fs4.npz`
- `dataset_combined_w20_s10_fs4.npz`

Override the folder with `--out_dir` (still relative to `srnn-pytorch/` unless you pass an absolute path).

**3. Training / evaluation**

From the `srnn/` folder, relative `--data` is resolved against **`srnn/`**, so use e.g.:

`--data ../data/mice/dataset_r1_w20_s10_fs4.npz` → file at `srnn-pytorch/data/mice/dataset_r1_w20_s10_fs4.npz`.

`train.py` auto-creates `log/mice/<exp_tag>/` and `save/mice/<exp_tag>/` under `srnn-pytorch/`; it does **not** create the `.npz` file itself.

### Preprocess

From `srnn-pytorch/`:

```bash
python preprocess.py \
  --window_size 20 --stride 10 --fs 4 \
  --out_dir data/mice
```

This writes `dataset_*_w20_s10_fs4.npz` under `data/mice/`. Use the `r1` (or `combined`) file in `--data` for training to match the published experiments.

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
