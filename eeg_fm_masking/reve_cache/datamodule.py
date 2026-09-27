"""Lightning DataModule for the pre-shuffled REVE cache.

Pipeline per-worker:
    ShuffledShardDataset (raw X + big_recording_index + n_chans)
        -> _RecordTransform (scale + ch_pos lookup + channel pad/subsample)
        -> DataLoader collate (stacks into batch)
        -> yields {X, ch_pos, ch_padding_mask}

Notes:
  * No DataLoader shuffle: the shard dataset shuffles at two levels (shard
    order per worker, records per shard) — see ShuffledShardDataset.
  * Set `reload_dataloaders_every_n_epochs=1` in the trainer if you train for
    more than one epoch; otherwise the shard-shuffle seed is frozen on the
    first epoch (the dataset's internal epoch counter is only bumped when we
    rebuild the DataLoader in `train_dataloader()`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from eeg_fm_masking.functions import scale_signal
from eeg_fm_masking.reve_cache.config import ReveCacheDataModuleConfig
from eeg_fm_masking.reve_cache.dataset import ShuffledShardDataset

_POSITIONS_FILENAME_RE = re.compile(r"recording_-_positions_-_(\d+)\.npy$")


def _load_positions_map(positions_dir: Path) -> dict[int, np.ndarray]:
    """Eagerly load every `recording_-_positions_-_<i>.npy` into a dict.

    ~6500 recordings × ~64 channels × 3 floats ≈ 5 MB total — cheap enough to
    load fully at setup() rather than wiring up a lazy per-worker cache.
    Positions are returned as float32 (xyz head coordinates).
    """
    out: dict[int, np.ndarray] = {}
    for p in positions_dir.glob("recording_-_positions_-_*.npy"):
        m = _POSITIONS_FILENAME_RE.match(p.name)
        if m is None:
            continue
        bri = int(m.group(1))
        out[bri] = np.load(p).astype(np.float32, copy=False)
    if not out:
        raise FileNotFoundError(
            f"no recording_-_positions_-_*.npy files found in {positions_dir}"
        )
    return out


class _RecordTransform(IterableDataset):
    """Wraps `ShuffledShardDataset` and produces model-ready records.

    Runs inside each DataLoader worker. Applies (in order):
      1. `scale_signal` — per-window normalisation; same semantics as
         `DatasetWrapper._transform_X` in the standard datamodule.
      2. ch_pos lookup: joins on `big_recording_index` to get the channel xyz
         positions (first `n_chans` rows of the recording-level positions file).
      3. Channel pad-or-subsample to `cfg.dataset_wrapper.n_chans`, mirroring
         `DatasetWrapper._subsample_channels` so that batches can be collated.

    Yields `{X, ch_pos, ch_padding_mask}` — the full set consumed by
    `ContextualEncoder.forward` in models.py.
    """

    def __init__(
        self,
        shard_ds: ShuffledShardDataset,
        positions_map: dict[int, np.ndarray],
        cfg: ReveCacheDataModuleConfig,
    ) -> None:
        super().__init__()
        self.shard_ds = shard_ds
        self.positions_map = positions_map
        self.cfg = cfg
        # Channel-subsampling RNG. The wrapper config's `seed` is None-able in
        # the upstream schema; treat None as "pick an OS-random seed".
        wrapper_seed = cfg.dataset_wrapper.seed
        self._rng = np.random.default_rng(wrapper_seed)
        self._target_n_chans = int(cfg.dataset_wrapper.n_chans)
        self._dtype = cfg.dataset_wrapper.dtype

    def __iter__(self):
        w = self.cfg.dataset_wrapper
        for rec in self.shard_ds:
            X = rec["X"]  # (n_c, T) float32 torch.Tensor
            n_c = rec["n_chans"]
            bri = rec["big_recording_index"]

            # Scale: scale_signal accepts torch.Tensor directly, so we stay in
            # torch and avoid a round-trip to numpy.
            X = scale_signal(X, w.factor, w.scaler, w.clip_sigma)
            # `dtype` is a string ("float32", "bfloat16", ...). torch dtypes
            # aren't keyword-accessible by arbitrary string, so use getattr.
            X = X.to(dtype=getattr(torch, self._dtype))

            # ch_pos lookup. The positions file stores all channels of the
            # recording; we take the first `n_c` rows (shards store channels
            # in the original recording order, same as the positions file).
            ch_pos_np = self.positions_map[bri][:n_c]
            ch_pos = torch.from_numpy(ch_pos_np).clone()

            # Pad or subsample to uniform channel count so default_collate works.
            X, ch_pos, ch_padding_mask = self._fix_n_chans(X, ch_pos, n_c)

            yield {
                "X": X,
                "ch_pos": ch_pos,
                "ch_padding_mask": ch_padding_mask,
            }

    def _fix_n_chans(
        self, X: torch.Tensor, ch_pos: torch.Tensor, n_c: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad with zeros (and mark via `ch_padding_mask=True`) or subsample.

        Mirrors `DatasetWrapper._subsample_channels`; kept as a self-contained
        method so this wrapper doesn't depend on the standard datamodule.
        """
        target = self._target_n_chans
        n_pad = target - n_c
        if n_pad > 0:
            # Pad last dim of X is time (keep it intact) → pad channels axis (=0).
            # F.pad counts from the last dim: (time_l, time_r, chan_l, chan_r).
            X = F.pad(X, (0, 0, 0, n_pad), mode="constant", value=0.0)
            ch_pos = F.pad(ch_pos, (0, 0, 0, n_pad), mode="constant", value=0.0)
            ch_padding_mask = torch.zeros(target, dtype=torch.bool)
            ch_padding_mask[n_c:] = True
        elif n_pad < 0:
            indices = self._rng.choice(n_c, size=target, replace=False)
            indices.sort()
            indices_t = torch.from_numpy(indices)
            X = X[indices_t]
            ch_pos = ch_pos[indices_t]
            ch_padding_mask = torch.zeros(target, dtype=torch.bool)
        else:
            ch_padding_mask = torch.zeros(target, dtype=torch.bool)
        return X, ch_pos, ch_padding_mask


class ReveCacheDataModule(LightningDataModule):
    """Lightning DataModule that streams from a pre-shuffled REVE shard cache."""

    def __init__(self, cfg: ReveCacheDataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._positions_map: dict[int, np.ndarray] | None = None

    def __getattribute__(self, item):
        # Mirror DataModule in datamodule.py: hide val_dataloader when no
        # val cache is configured, so Lightning's hasattr() skips validation.
        if item == "val_dataloader":
            cfg = object.__getattribute__(self, "cfg")
            if cfg.val_cache_dir is None:
                raise AttributeError
        return object.__getattribute__(self, item)

    def setup(self, stage: str | None = None) -> None:
        if self._positions_map is None:
            self._positions_map = _load_positions_map(self.cfg.positions_dir)

    def _make_dataloader(self, cache_dir: Path, epoch: int) -> DataLoader:
        assert self._positions_map is not None, "setup() must be called first"
        shard_ds = ShuffledShardDataset(cache_dir, seed=self.cfg.shuffle_seed)
        shard_ds.set_epoch(epoch)
        wrapped = _RecordTransform(shard_ds, self._positions_map, self.cfg)
        return DataLoader(
            wrapped,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            # IterableDataset: no shuffle flag; shuffling happens inside the dataset.
            # drop_last=True is required under DDP: combined with the per-worker
            # `budget` truncation in ShuffledShardDataset, it guarantees every
            # rank yields exactly `(budget * num_workers) // batch_size` batches
            # per epoch. Without it, a partial last batch would survive on each
            # rank and reintroduce a 0/1-batch divergence at the epoch boundary
            # (NCCL collective mismatch on the next allreduce).
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        # current_epoch is 0 before the first fit-call; trainer may be None
        # during testing/introspection, so fall back to 0.
        epoch = getattr(self.trainer, "current_epoch", 0) if self.trainer else 0
        return self._make_dataloader(self.cfg.train_cache_dir, epoch=epoch)

    def val_dataloader(self) -> DataLoader:
        # Fixed epoch=0 for reproducible validation ordering across runs.
        assert self.cfg.val_cache_dir is not None
        return self._make_dataloader(self.cfg.val_cache_dir, epoch=0)
