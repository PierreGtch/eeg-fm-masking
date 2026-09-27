"""Pre-shuffled cache generator.

Two layers:
  - `_window_iter(rec_df, subrec_df, rec_dirs)` — iterator over 30 s windows
  - `generate_cache(...)` / `generate_cache_parallel(...)` — multi-process
    orchestration that randomly routes each window to a shard, so a single
    sequential read per shard at training time already yields a fully
    diverse mini-batch.
"""
from __future__ import annotations

import json
import random
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from eeg_fm_masking.reve_cache.format import WINDOW_SAMPLES
from eeg_fm_masking.reve_cache.writer import ShardWriter


def _find_source(big_rec_idx: int, rec_dirs: list[Path]) -> Path:
    for d in rec_dirs:
        p = d / f"recording_-_eeg_-_{big_rec_idx}.npy"
        if p.is_file():
            return p
    raise FileNotFoundError(f"no source file for big_recording_index={big_rec_idx}")


def _window_iter(
    rec_df: pd.DataFrame,
    subrec_df: pd.DataFrame,
    rec_dirs: list[Path],
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield (window_float32[n_chans, WINDOW_SAMPLES], big_recording_index, sub_idx).

    Windows are non-overlapping 30 s (stride = window length). Subrecording
    durations that are not an integer multiple of the window length have
    their tail dropped (at most WINDOW_SAMPLES - 1 samples lost).
    """
    for r_i in rec_df["big_recording_index"].astype(int):
        row = rec_df.loc[rec_df["big_recording_index"] == r_i].iloc[0]
        n_c = int(row["n_chans"])
        n_t = int(row["duration"])
        src = _find_source(r_i, rec_dirs)
        mm = np.memmap(src, dtype=np.float32, mode="r", shape=(n_t, n_c))

        subs = subrec_df[subrec_df["big_recording_index"] == r_i]
        for sub_pos, sub in enumerate(subs.itertuples(index=False)):
            start, end = int(sub.start), int(sub.end)
            n_win = max(0, (end - start) // WINDOW_SAMPLES)
            for w in range(n_win):
                t0 = start + w * WINDOW_SAMPLES
                # .copy() materialises (mm slice would otherwise stay lazy).
                yield mm[t0:t0 + WINDOW_SAMPLES].T.copy(), r_i, sub_pos
        del mm


def generate_cache(
    rec_df: pd.DataFrame,
    subrec_df: pd.DataFrame,
    rec_dirs: list[Path],
    out_dir: Path,
    seed: int,
    target_shard_bytes: int = 1 * 2**30,
    n_writer_groups: int = 32,
) -> dict:
    """Generate a pre-shuffled cache of REVE windows into `out_dir`.

    Sequential source reads + random routing to worker-owned shard pools.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estimate total bytes to pre-create the right number of shards.
    mask = subrec_df["big_recording_index"].isin(rec_df["big_recording_index"])
    spanned = subrec_df.loc[mask]
    n_chans_map = dict(
        zip(rec_df["big_recording_index"].astype(int), rec_df["n_chans"].astype(int))
    )
    total_bytes = int(sum(
        ((int(s.end) - int(s.start)) // WINDOW_SAMPLES)
        * n_chans_map[int(s.big_recording_index)]
        * WINDOW_SAMPLES * 4
        for s in spanned.itertuples(index=False)
    ))
    # +2% slack to absorb the per-shard cap variance from random routing.
    n_shards = max(1, int(-(-total_bytes * 102 // (target_shard_bytes * 100))))
    group_shards = [
        list(range(g, n_shards, n_writer_groups)) for g in range(n_writer_groups)
    ]

    writers = [
        ShardWriter(
            dat_path=out_dir / f"shard_{i:05d}.dat",
            idx_path=out_dir / f"shard_{i:05d}.idx",
            cap_bytes=target_shard_bytes,
        )
        for i in range(n_shards)
    ]
    available = [list(s) for s in group_shards]
    rng = random.Random(seed)
    idx = 0
    for window, big_rec_idx, sub_idx in _window_iter(rec_df, subrec_df, rec_dirs):
        group = idx % n_writer_groups
        pool = available[group]
        if not pool:
            # All shards in this group are full — spill into any other group.
            flat = [s for g in available for s in g]
            if not flat:
                raise RuntimeError("no shards available — target_shard_bytes too small")
            pool = flat
        j = rng.randrange(len(pool))
        shard_id = pool[j]
        writers[shard_id].append(window, big_rec_idx, sub_idx)
        if writers[shard_id].full and shard_id in available[group]:
            available[group].remove(shard_id)
        idx += 1

    for w in writers:
        w.close()

    shard_sizes = [
        (out_dir / f"shard_{i:05d}.dat").stat().st_size for i in range(n_shards)
    ]
    manifest = {
        "seed": seed,
        "generated_at": time.time(),
        "n_shards": n_shards,
        "n_records": idx,
        "target_shard_bytes": target_shard_bytes,
        "window_samples": WINDOW_SAMPLES,
        "shard_sizes": shard_sizes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _worker_generate(args) -> dict:
    worker_id, rec_df, subrec_df, rec_dirs, worker_dir, seed, target_shard_bytes = args
    return generate_cache(
        rec_df=rec_df,
        subrec_df=subrec_df,
        rec_dirs=rec_dirs,
        out_dir=worker_dir,
        seed=seed + worker_id,
        target_shard_bytes=target_shard_bytes,
        n_writer_groups=1,
    )


def generate_cache_parallel(
    rec_df: pd.DataFrame,
    subrec_df: pd.DataFrame,
    rec_dirs: list[Path],
    out_dir: Path,
    seed: int,
    target_shard_bytes: int = 1 * 2**30,
    n_workers: int = 32,
) -> dict:
    """Parallel variant: split rec_df across `n_workers`, merge shards into `out_dir`.

    Each worker calls `generate_cache` into a private sub-directory; after all
    workers finish, shards are renamed into `out_dir` with a unified numbering.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_base = out_dir / "_workers"
    worker_base.mkdir(exist_ok=True)

    rec_ids = rec_df["big_recording_index"].astype(int).tolist()
    partitions: list[list[int]] = [[] for _ in range(n_workers)]
    for i, r_i in enumerate(rec_ids):
        partitions[i % n_workers].append(r_i)

    args_list = []
    for w, rids in enumerate(partitions):
        if not rids:
            continue
        w_dir = worker_base / f"w{w:03d}"
        w_dir.mkdir(exist_ok=True)
        sub_rec_df = rec_df[rec_df["big_recording_index"].isin(rids)].reset_index(drop=True)
        sub_subrec_df = subrec_df[subrec_df["big_recording_index"].isin(rids)].reset_index(drop=True)
        args_list.append(
            (w, sub_rec_df, sub_subrec_df, rec_dirs, w_dir, seed, target_shard_bytes)
        )

    ctx = get_context("spawn")
    with ctx.Pool(min(n_workers, len(args_list))) as pool:
        sub_manifests = pool.map(_worker_generate, args_list)

    shard_sizes: list[int] = []
    next_id = 0
    for w, sm in enumerate(sub_manifests):
        w_dir = worker_base / f"w{w:03d}"
        for i in range(sm["n_shards"]):
            src_dat = w_dir / f"shard_{i:05d}.dat"
            src_idx = w_dir / f"shard_{i:05d}.idx"
            dst_dat = out_dir / f"shard_{next_id:05d}.dat"
            dst_idx = out_dir / f"shard_{next_id:05d}.idx"
            src_dat.rename(dst_dat)
            src_idx.rename(dst_idx)
            shard_sizes.append(dst_dat.stat().st_size)
            next_id += 1
        (w_dir / "manifest.json").unlink(missing_ok=True)
        w_dir.rmdir()
    worker_base.rmdir()

    total_records = sum(sm["n_records"] for sm in sub_manifests)
    manifest = {
        "seed": seed,
        "generated_at": time.time(),
        "n_shards": next_id,
        "n_records": total_records,
        "target_shard_bytes": target_shard_bytes,
        "window_samples": WINDOW_SAMPLES,
        "shard_sizes": shard_sizes,
        "n_workers": n_workers,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
