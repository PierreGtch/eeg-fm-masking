"""Torch IterableDataset over pre-shuffled REVE shards.

Per-epoch shuffle strategy:
  1. Sort all shard paths and shuffle with a seed derived from (seed, epoch).
  2. Every effective-worker (= rank * num_workers + worker_id, out of
     world_size * num_workers) takes every `eff_nw`-th shard from the shuffled
     list, starting at its own index. DDP ranks get disjoint shard slices; same
     worker_id across ranks sees DIFFERENT shards.
  3. The effective-worker permutes its own slice with a seed derived from the
     epoch seed and its effective id.
  4. For each shard, record indices are permuted with yet another derived seed.
  5. Each effective-worker emits exactly `min_records_per_worker` records
     (computed deterministically across all ranks using cached shard sizes).
     This gives every DDP rank exactly the same per-rank record budget — and,
     once the DataLoader is built with `drop_last=True`, the same number of
     batches per epoch. Without this, IterableDataset + DDP causes
     rank-divergent batch counts (shards have variable record counts), and
     one rank exits its training loop a few steps early while the other is
     still issuing gradient ALLREDUCEs — producing an NCCL collective
     mismatch.

Each shard is read fully into RAM before yielding records, which guarantees
sequential disk reads regardless of the in-memory shuffle order.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from eeg_fm_masking.reve_cache.format import WINDOW_SAMPLES
from eeg_fm_masking.reve_cache.reader import read_shard_flat


def _get_dist_info() -> tuple[int, int]:
    """Return (rank, world_size). Falls back to (0, 1) outside DDP."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _list_shards(cache_dir: Path) -> list[tuple[Path, Path]]:
    """Find (dat, idx) pairs in cache_dir, sorted by shard id."""
    dats = sorted(cache_dir.glob("shard_*.dat"))
    pairs: list[tuple[Path, Path]] = []
    for d in dats:
        i = d.with_suffix(".idx")
        if i.is_file():
            pairs.append((d, i))
    return pairs


def _shard_record_counts(shards: list[tuple[Path, Path]]) -> list[int]:
    """Read record count from every .idx file (`size_bytes // 20`).

    Reads only file metadata (one stat() per shard) — no I/O on shard contents.
    For ~4k shards on a typical shared filesystem this is <1 s.
    """
    from eeg_fm_masking.reve_cache.format import INDEX_ENTRY_SIZE
    return [idx.stat().st_size // INDEX_ENTRY_SIZE for _, idx in shards]


def _min_records_per_worker(
    sizes: list[int],
    epoch_seed: int,
    world_size: int,
    num_workers: int,
) -> int:
    """Smallest per-effective-worker record count across all (rank, worker_id).

    Computed deterministically from the shard sizes + epoch seed, mirroring the
    partition logic in `ShuffledShardDataset.__iter__`. Every rank can compute
    the same value without any cross-rank communication, so this is the budget
    each worker truncates to. Truncating to `min` (rather than `mean` or `max`)
    is what makes per-rank batch counts identical without resampling/replay.
    """
    eff_nw = world_size * num_workers
    pool_indices = list(range(len(sizes)))
    random.Random(epoch_seed).shuffle(pool_indices)
    # For each effective-worker, sum its slice of shard sizes.
    per_worker_totals = [
        sum(sizes[i] for i in pool_indices[eff::eff_nw])
        for eff in range(eff_nw)
    ]
    return min(per_worker_totals)


def _derive_seed(*parts: int) -> int:
    """Hash a tuple of ints into a uint32 seed in a stable way."""
    h = 0
    for p in parts:
        h = (h * 2654435761 + int(p)) & 0xFFFFFFFF
    return h


class ShuffledShardDataset(torch.utils.data.IterableDataset):
    """Streams pre-shuffled REVE windows from a shard directory.

    Yields raw records `{X, n_chans, big_recording_index, sub_idx}`. The training
    transforms (scaling, ch_pos lookup, channel padding) are applied downstream
    by `ReveCacheDataModule`.
    """

    def __init__(self, cache_dir: Path | str, seed: int = 42) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.seed = int(seed)
        self._shards = _list_shards(self.cache_dir)
        if not self._shards:
            raise FileNotFoundError(f"no shards found in {self.cache_dir}")
        # Cache record-count per shard so worker processes don't re-stat at
        # __iter__ time. ~4k stats; runs once at construction in the main proc.
        self._shard_sizes = _shard_record_counts(self._shards)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        rank, world_size = _get_dist_info()
        # Effective worker index across all ranks — ranges [0, world_size * nw).
        # Same worker_id on different ranks maps to different eff_wid, so the
        # shard slicing below gives each rank a disjoint portion of the shards.
        eff_wid = rank * nw + wid
        eff_nw = world_size * nw

        epoch_seed = _derive_seed(self.seed, self._epoch)
        pool = list(self._shards)
        random.Random(epoch_seed).shuffle(pool)
        my_shards = pool[eff_wid::eff_nw]
        random.Random(_derive_seed(epoch_seed, eff_wid)).shuffle(my_shards)

        # DDP rank-equalization budget (see module docstring step 5):
        # truncate this worker's output to the global min per-worker record
        # count so every rank yields exactly the same total — required for
        # NCCL collective ordering at the epoch boundary. With world=1 this
        # is a no-op (min == this worker's count, modulo shuffle).
        budget = _min_records_per_worker(
            self._shard_sizes, epoch_seed, world_size, nw
        )
        emitted = 0

        for dat, idx in my_shards:
            if emitted >= budget:
                break
            # Next-shard prefetch (nice-to-have; best-effort).
            self._maybe_prefetch_next(my_shards, dat)

            flat, index = read_shard_flat(dat, idx)
            if len(index) == 0:
                continue

            perm_seed = _derive_seed(epoch_seed, eff_wid, hash(dat.name) & 0xFFFF)
            order = np.arange(len(index))
            np.random.default_rng(perm_seed).shuffle(order)

            # Yield torch.Tensors wrapping zero-copy slices of `flat`.
            # With a multi-process DataLoader, returning a torch.Tensor lets the
            # runtime move the data through shared memory (mmap-backed) instead
            # of pickling the bytes across the IPC boundary — a >10x speedup
            # for large per-record payloads compared to raw numpy ndarrays.
            for k in order:
                if emitted >= budget:
                    break
                entry = index[int(k)]
                n_c = int(entry["n_chans"])
                start = int(entry["offset"]) // 4  # byte offset -> float32 index
                count = n_c * WINDOW_SAMPLES
                view = flat[start:start + count].reshape(n_c, WINDOW_SAMPLES)
                yield {
                    "X": torch.from_numpy(view),
                    "n_chans": n_c,
                    "big_recording_index": int(entry["big_rec_idx"]),
                    "sub_idx": int(entry["sub_idx"]),
                }
                emitted += 1

            del flat
            self._drop_cache(dat)

    @staticmethod
    def _maybe_prefetch_next(shards: list[tuple[Path, Path]], current: Path) -> None:
        """Best-effort POSIX_FADV_WILLNEED on the next shard in the list.

        posix_fadvise is Linux-only; on macOS/BSD the attributes don't exist
        (AttributeError, not OSError), so we guard with hasattr.
        """
        if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_WILLNEED"):
            return
        for i, (d, _) in enumerate(shards):
            if d == current and i + 1 < len(shards):
                next_dat = shards[i + 1][0]
                try:
                    fd = os.open(next_dat, os.O_RDONLY)
                    try:
                        sz = os.fstat(fd).st_size
                        os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_WILLNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass
                return

    @staticmethod
    def _drop_cache(path: Path) -> None:
        """Best-effort POSIX_FADV_DONTNEED on a shard we are done with (Linux-only)."""
        if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
            return
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                sz = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass
