"""
Mouse Trajectory Dataset Preprocessing

Loads raw MABe 2022 mouse tracking data, cleans it (completely discarding
invalid frames rather than interpolating), extracts selected keypoints,
and builds sliding-window datasets ready for SRNN training.

Produces three datasets: r1 (user_train_r1), r2 (user_train), and combined.

Usage:
    python preprocess_mice.py --window_size 20
    python preprocess_mice.py --window_size 15 --stride 7
    python preprocess_mice.py --analyze_only
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ─── Physical constants ─────────────────────────────────────────────
SELECTED_KPS = [0, 3, 6, 9]  # nose, neck, center_back, tail_base
KP_NAMES = ["nose", "neck", "center_back", "tail_base"]
N_MICE = 3
N_KPS = len(SELECTED_KPS)
N_NODES = N_MICE * N_KPS            # 12
FRAME_RATE = 30                       # original capture Hz
MOUSE_MAX_SPEED_CM_S = 60.0          # literature: max sprint ~60 cm/s

# Dataset-specific arena sizes (diameter in px) — estimated from P1/P99 range
DATASET_META = {
    "r1": {
        "file": "data/mice/user_train_r1.npy",
        "arena_px": 710,
        "arena_cm": 50.0,
    },
    "r2": {
        "file": "data/mice/user_train.npy",
        "arena_px": 450,
        "arena_cm": 50.0,  # same physical arena, different resolution
    },
}


# ─── Core helpers ────────────────────────────────────────────────────

def load_raw(path: str) -> dict:
    """Load a .npy MABe file and return the dict."""
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "item"):
        return data.item()
    return data


def frame_is_valid(kp_frame, kp_indices=SELECTED_KPS):
    """
    Check whether a single frame has valid tracking for all mice and
    selected keypoints. A coordinate of (0,0) signals tracking loss.

    kp_frame: (n_mice, n_all_kps, 2)
    """
    for m in range(N_MICE):
        for ki in kp_indices:
            if abs(kp_frame[m, ki, 0]) + abs(kp_frame[m, ki, 1]) < 1.0:
                return False
    return True


def compute_validity_mask(kp_seq, already_selected=False):
    """
    Return a boolean array (T,) indicating valid frames.
    If already_selected=True, kp_seq has shape (T, 3, n_selected_kps, 2)
    and we check all keypoints.  Otherwise uses SELECTED_KPS indices.
    """
    T = kp_seq.shape[0]
    mask = np.ones(T, dtype=bool)
    if already_selected:
        n_kps = kp_seq.shape[2]
        for t in range(T):
            for m in range(N_MICE):
                for ki in range(n_kps):
                    if abs(kp_seq[t, m, ki, 0]) + abs(kp_seq[t, m, ki, 1]) < 1.0:
                        mask[t] = False
                        break
                if not mask[t]:
                    break
    else:
        for t in range(T):
            mask[t] = frame_is_valid(kp_seq[t], SELECTED_KPS)
    return mask


def compute_window_activity(window, arena_px):
    """
    Mean per-frame displacement of center_back across 3 mice,
    normalised by arena size.  window: (win, 3, n_kps, 2)
    center_back is index 2 within the selected 4 keypoints.
    """
    cb_idx = KP_NAMES.index("center_back")
    disp = np.linalg.norm(
        np.diff(window[:, :, cb_idx, :], axis=0), axis=-1
    )  # (win-1, 3)
    return disp.mean() / arena_px


def build_windows(kp_seq, ann, validity, win, stride, arena_px,
                  static_thresh):
    """
    Extract sliding windows from one sequence.

    Returns list of dicts, each containing:
      - 'data':     (win, n_mice, n_kps, 2) float32 normalised coords
      - 'lights':   int  (0/1, majority vote within window)
      - 'chase':    int  (0/1, any chase frame within window)
      - 'activity': float (mean normalised displacement)
    """
    T = kp_seq.shape[0]
    windows = []
    for start in range(0, T - win, stride):
        end = start + win
        if not validity[start:end].all():
            continue

        raw_window = kp_seq[start:end].astype(np.float32)
        activity = compute_window_activity(raw_window, arena_px)

        if activity < static_thresh:
            continue

        norm_window = raw_window / arena_px

        chase_ann = ann[0][start:end]
        lights_ann = ann[1][start:end]

        windows.append({
            "data": norm_window,
            "lights": int(lights_ann.mean() > 0.5),
            "chase": int(chase_ann.sum() > 0),
            "activity": float(activity),
        })
    return windows


# ─── Dataset builder ─────────────────────────────────────────────────

def process_dataset(raw_dict, arena_px, fs, win, stride, static_thresh):
    """
    Process one raw dataset dict into a list of valid windows.

    Returns:
        windows: list[dict]
        stats: dict with counts and distributions
    """
    seqs = raw_dict["sequences"]
    keys = list(seqs.keys())

    all_windows = []
    n_valid_seqs = 0
    n_total_frames = 0
    n_valid_frames = 0
    n_discarded_static = 0
    n_discarded_invalid = 0

    for sid in keys:
        s = seqs[sid]
        kp_full = np.array(s["keypoints"]).astype(np.float32)
        ann_full = s["annotations"]  # (2, 1800)

        kp_sub = kp_full[::fs][:, :, SELECTED_KPS, :]
        ann_sub = np.stack([ann_full[0][::fs], ann_full[1][::fs]], axis=0)
        T = len(kp_sub)
        n_total_frames += T

        validity = compute_validity_mask(kp_sub, already_selected=True)
        n_valid_frames += validity.sum()

        if validity.sum() < win:
            continue
        n_valid_seqs += 1

        seq_windows = build_windows(
            kp_sub, ann_sub, validity,
            win, stride, arena_px, static_thresh,
        )
        all_windows.extend(seq_windows)

    n_total_windows = len(keys) * max(1, (450 - win) // stride)
    stats = {
        "n_sequences": len(keys),
        "n_valid_sequences": n_valid_seqs,
        "n_total_frames": n_total_frames,
        "n_valid_frames": n_valid_frames,
        "valid_frame_ratio": n_valid_frames / max(1, n_total_frames),
        "n_windows": len(all_windows),
    }
    return all_windows, stats


def split_by_index(windows, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Split windows into train / val / test.
    Uses a deterministic shuffle.
    """
    rng = np.random.RandomState(seed)
    n = len(windows)
    idx = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    pick = lambda idxs: [windows[i] for i in idxs]
    return pick(train_idx), pick(val_idx), pick(test_idx)


def windows_to_arrays(window_list):
    """Convert list of window dicts to numpy arrays for saving."""
    if not window_list:
        return {
            "data": np.empty((0, 0, N_MICE, N_KPS, 2), dtype=np.float32),
            "lights": np.empty(0, dtype=np.int32),
            "chase": np.empty(0, dtype=np.int32),
            "activity": np.empty(0, dtype=np.float32),
        }
    return {
        "data": np.stack([w["data"] for w in window_list]).astype(np.float32),
        "lights": np.array([w["lights"] for w in window_list], dtype=np.int32),
        "chase": np.array([w["chase"] for w in window_list], dtype=np.int32),
        "activity": np.array([w["activity"] for w in window_list], dtype=np.float32),
    }


def save_dataset(windows, out_path, name, win, stride, fs, split_ratios):
    """Split windows and save as .npz file."""
    train, val, test = split_by_index(windows, split_ratios)

    arrays = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        d = windows_to_arrays(split_data)
        for k, v in d.items():
            arrays[f"{split_name}_{k}"] = v

    arrays["meta_window_size"] = np.array(win)
    arrays["meta_stride"] = np.array(stride)
    arrays["meta_fs"] = np.array(fs)
    arrays["meta_hz"] = np.array(FRAME_RATE / fs)
    arrays["meta_n_nodes"] = np.array(N_NODES)
    arrays["meta_kp_indices"] = np.array(SELECTED_KPS)
    arrays["meta_kp_names"] = np.array(KP_NAMES)

    np.savez_compressed(out_path, **arrays)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved {name}: {out_path} ({size_mb:.1f} MB)")
    print(f"    train={len(train)}, val={len(val)}, test={len(test)}")


# ─── Analysis ────────────────────────────────────────────────────────

def analyze_window_sizes(raw_dict, arena_px, fs):
    """Print a recommendation table for different window sizes."""
    seqs = raw_dict["sequences"]
    keys = list(seqs.keys())

    # Pre-compute validity for all sequences
    valid_counts = {}
    for sid in keys:
        kp = np.array(seqs[sid]["keypoints"]).astype(np.float32)[::fs][:, :, SELECTED_KPS, :]
        valid_counts[sid] = compute_validity_mask(kp, already_selected=True)

    hz = FRAME_RATE / fs
    print(f"\n  {'Win':>4s} {'Time':>6s} {'Stride':>6s} {'#Windows':>9s} "
          f"{'obs/pred':>9s} {'Batches':>8s}")
    print("  " + "-" * 55)

    for win in [10, 15, 20, 25, 30, 40]:
        stride = win // 2
        n_win = 0
        for sid in keys:
            T = len(valid_counts[sid])
            for start in range(0, T - win, stride):
                if valid_counts[sid][start:start + win].all():
                    n_win += 1

        obs = int(win * 0.75)
        pred = win - obs
        time_s = win / hz
        batches = n_win // 64

        marker = " ★" if 2.0 <= time_s <= 4.0 else ""
        print(f"  {win:>4d} {time_s:>5.1f}s {stride:>6d} {n_win:>9d} "
              f"{obs:>4d}/{pred:<4d} {batches:>8d}{marker}")


def print_dataset_stats(name, windows, stats):
    """Pretty-print statistics for a processed dataset."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Sequences: {stats['n_sequences']} total, "
          f"{stats['n_valid_sequences']} with enough valid frames")
    print(f"  Frames:    {stats['n_valid_frames']}/{stats['n_total_frames']} "
          f"valid ({stats['valid_frame_ratio']*100:.1f}%)")
    print(f"  Windows:   {stats['n_windows']}")

    if windows:
        acts = np.array([w["activity"] for w in windows])
        lights = np.array([w["lights"] for w in windows])
        chase = np.array([w["chase"] for w in windows])

        print(f"  Activity:  mean={acts.mean()*50*7.5:.1f} cm/s, "
              f"median={np.median(acts)*50*7.5:.1f} cm/s")
        print(f"  Lights ON: {lights.sum()}/{len(lights)} "
              f"({lights.mean()*100:.1f}%)")
        print(f"  Has chase: {chase.sum()}/{len(chase)} "
              f"({chase.mean()*100:.2f}%)")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MABe mouse data into sliding-window datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--window_size", type=int, default=20,
        help="Sliding window length in subsampled frames. "
             "At fs=4 (7.5 Hz): 15→2.0s, 20→2.7s, 25→3.3s, 30→4.0s",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Window stride (default: window_size // 2)",
    )
    parser.add_argument(
        "--fs", type=int, default=4,
        help="Frame subsampling factor (30Hz → 30/fs Hz)",
    )
    parser.add_argument(
        "--static_thresh", type=float, default=2e-3,
        help="Discard windows where mean normalised displacement is below this "
             "(eliminates low-activity windows). "
             "Default 2e-3 ≈ 0.9 px/frame ≈ 0.75 cm/s; keeps ~76%% of data.",
    )
    parser.add_argument(
        "--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train / val / test split ratios",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/mice",
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--analyze_only", action="store_true",
        help="Only print analysis, do not create datasets",
    )
    args = parser.parse_args()

    if args.stride is None:
        args.stride = args.window_size // 2

    hz = FRAME_RATE / args.fs
    win_time = args.window_size / hz
    obs_len = int(args.window_size * 0.75)
    pred_len = args.window_size - obs_len

    print("=" * 60)
    print("  Mouse Data Preprocessing")
    print("=" * 60)
    print(f"  Window:    {args.window_size} frames = {win_time:.1f}s @ {hz:.1f}Hz")
    print(f"  Stride:    {args.stride} frames = {args.stride/hz:.1f}s")
    print(f"  Subsample: fs={args.fs} (30Hz → {hz:.1f}Hz)")
    print(f"  obs/pred:  {obs_len}/{pred_len}")
    print(f"  Split:     {args.split}")
    print(f"  Output:    {args.out_dir}")

    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent

    # Load raw data
    print("\nLoading raw data...")
    raw_data = {}
    for dset_name, meta in DATASET_META.items():
        fpath = script_dir / meta["file"]
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping {dset_name}")
            continue
        t0 = time.time()
        raw_data[dset_name] = load_raw(str(fpath))
        n_seq = len(raw_data[dset_name]["sequences"])
        print(f"  {dset_name}: {n_seq} sequences loaded ({time.time()-t0:.1f}s)")

    if not raw_data:
        print("ERROR: No data files found. Exiting.")
        sys.exit(1)

    # Analyze window sizes if requested
    if args.analyze_only:
        for dname, rdata in raw_data.items():
            meta = DATASET_META[dname]
            print(f"\n--- Window size analysis for {dname} ---")
            analyze_window_sizes(rdata, meta["arena_px"], args.fs)
        print("\n★ = recommended range (2.0s – 4.0s)")
        return

    # Process each dataset
    processed = OrderedDict()
    for dname, rdata in raw_data.items():
        meta = DATASET_META[dname]
        print(f"\nProcessing {dname}...")
        t0 = time.time()
        windows, stats = process_dataset(
            rdata, meta["arena_px"], args.fs,
            args.window_size, args.stride, args.static_thresh,
        )
        elapsed = time.time() - t0
        print_dataset_stats(dname, windows, stats)
        print(f"  (processed in {elapsed:.1f}s)")
        processed[dname] = windows

    # Build combined dataset
    combined = []
    for ws in processed.values():
        combined.extend(ws)
    print(f"\nCombined: {len(combined)} windows total")

    # Save
    out_dir = script_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"w{args.window_size}_s{args.stride}_fs{args.fs}"
    print(f"\nSaving datasets (tag: {tag})...")

    for dname, windows in processed.items():
        out_path = out_dir / f"dataset_{dname}_{tag}.npz"
        save_dataset(
            windows, str(out_path), dname,
            args.window_size, args.stride, args.fs, tuple(args.split),
        )

    out_path_combined = out_dir / f"dataset_combined_{tag}.npz"
    save_dataset(
        combined, str(out_path_combined), "combined",
        args.window_size, args.stride, args.fs, tuple(args.split),
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for dname, windows in processed.items():
        n_train = int(len(windows) * args.split[0])
        print(f"  {dname:>10s}: {len(windows):>6d} windows → "
              f"{n_train} train, batch@64 = {n_train//64} batches/epoch")
    n_train_c = int(len(combined) * args.split[0])
    print(f"  {'combined':>10s}: {len(combined):>6d} windows → "
          f"{n_train_c} train, batch@64 = {n_train_c//64} batches/epoch")

    print(f"\n  Window: {args.window_size} frames ({win_time:.1f}s), "
          f"obs={obs_len}, pred={pred_len}")
    print(f"  Data shape per window: "
          f"({args.window_size}, {N_MICE}, {N_KPS}, 2)")
    print(f"  Keypoints: {', '.join(KP_NAMES)}")
    print(f"  Coordinate range: [0, 1] (normalised by arena diameter)")

    print(f"\n  Output files:")
    for f in sorted(out_dir.glob(f"dataset_*_{tag}.npz")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")

    print(f"\n  To load:")
    print(f"    d = np.load('data/mice/dataset_combined_{tag}.npz')")
    print(f"    X_train = d['train_data']   # (N, {args.window_size}, 3, 4, 2)")
    print(f"    y_lights = d['train_lights'] # (N,)")
    print()


if __name__ == "__main__":
    main()
