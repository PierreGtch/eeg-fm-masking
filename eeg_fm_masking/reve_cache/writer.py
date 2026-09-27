"""Append-only writer for a single shard (two files: .dat + .idx)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from eeg_fm_masking.reve_cache.format import (
    BlockBufferedWriter,
    WRITE_BLOCK_SIZE,
    WINDOW_SAMPLES,
)


class ShardWriter:
    """Writes records to (dat_path, idx_path). Pure append, no finalisation phase.

    Order of writes per record: data first, then index entry. A crash mid-write
    leaves at most one orphan record in .dat that is invisible because not
    referenced in .idx.
    """

    def __init__(self, dat_path: Path, idx_path: Path, cap_bytes: int) -> None:
        self._dat = BlockBufferedWriter(dat_path, block_size=WRITE_BLOCK_SIZE)
        self._idx = open(idx_path, "wb", buffering=1 * 2**20)
        self._bytes_written = 0
        self._cap = cap_bytes

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def full(self) -> bool:
        return self._bytes_written >= self._cap

    def append(self, window: np.ndarray, big_rec_idx: int, sub_idx: int) -> None:
        assert window.dtype == np.float32
        assert window.ndim == 2 and window.shape[1] == WINDOW_SAMPLES
        offset = self._bytes_written
        self._dat.write(window.tobytes())
        self._bytes_written += window.nbytes
        # Index AFTER data so a crash between the two leaves a data orphan
        # rather than a dangling idx entry pointing past EOF.
        self._idx.write(
            struct.pack("<QIII", offset, window.shape[0], int(big_rec_idx), int(sub_idx))
        )

    def close(self) -> None:
        self._dat.close()
        self._idx.close()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
