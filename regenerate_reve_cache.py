"""Generate the pre-shuffled REVE shard cache used by `mask_sweep_full.py`.

Expects the REVE dataset (https://huggingface.co/datasets/brain-bzh/reve-dataset)
to already be downloaded locally. The dataset ships:
  - `csvs/df_big.csv` and `csvs/df_clean.csv` — recording / sub-recording metadata
  - `recordings*/recording_-_eeg_-_<i>.npy` — float32 (n_t, n_c) memmaps

Usage:
  python regenerate_reve_cache.py \\
      --split train \\
      --reve_dir /path/to/reve-dataset \\
      --out_dir /path/to/reve_shuffled/train \\
      --seed 42 \\
      --n_workers 32

The output directory will contain `shard_NNNNN.dat` / `shard_NNNNN.idx` pairs
plus a `manifest.json`. Point `mask_sweep_full.py` at the `train` directory via
the `REVE_TRAIN_CACHE_DIR` env var.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from eeg_fm_masking.reve_cache.generate import generate_cache_parallel
from eeg_fm_masking.reve_cache.splits import OPEN_TRAIN, OPEN_VAL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument(
        "--reve_dir",
        type=Path,
        required=True,
        help="Root of the downloaded REVE dataset (must contain csvs/ and "
             "the recording .npy files).",
    )
    p.add_argument(
        "--rec_subdirs",
        nargs="+",
        default=["recordings-1", "recordings-2"],
        help="Subdirectories of --reve_dir containing recording_-_eeg_-_*.npy.",
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=32)
    p.add_argument("--target_shard_bytes", type=int, default=1 * 2**30)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rec_ids = OPEN_TRAIN if args.split == "train" else OPEN_VAL
    rec_dirs = [args.reve_dir / sd for sd in args.rec_subdirs]
    csv_dir = args.reve_dir / "csvs"

    rec_df = pd.read_csv(csv_dir / "df_big.csv")
    subrec_df = pd.read_csv(csv_dir / "df_clean.csv", low_memory=False)

    rec_df = rec_df[rec_df["big_recording_index"].isin(rec_ids)].reset_index(drop=True)
    subrec_df = subrec_df[
        subrec_df["big_recording_index"].isin(rec_ids)
    ].reset_index(drop=True)
    print(
        f"[regen] split={args.split} seed={args.seed} n_workers={args.n_workers} "
        f"rec_df={len(rec_df)} subrec_df={len(subrec_df)}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = args.out_dir.parent / "GENERATING.lock"
    lock.write_text(f"started_at={time.time()}\n")

    try:
        manifest = generate_cache_parallel(
            rec_df=rec_df,
            subrec_df=subrec_df,
            rec_dirs=rec_dirs,
            out_dir=args.out_dir,
            seed=args.seed,
            target_shard_bytes=args.target_shard_bytes,
            n_workers=args.n_workers,
        )
        print(json.dumps(manifest, indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
