"""Cache on-disk format: constants, lookup dtype, block-aligned writer.

A shard is two files:
  shard_NNNNN.dat  — float32 records concatenated, no header
  shard_NNNNN.idx  — packed entries (offset, n_chans, big_rec_idx, sub_idx), 20 B each
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WRITE_BLOCK_SIZE: int = 16 * 2**20
"""Per-write OS chunk size (16 MiB). Matches a typical HPC block size and
avoids block-allocation amplification on shared filesystems."""

WINDOW_SAMPLES: int = 6000
"""30 seconds × 200 Hz — the model's training window length."""

INDEX_ENTRY_SIZE: int = 20
"""(uint64 offset) + (uint32 n_chans) + (uint32 big_rec_idx) + (uint32 sub_idx)."""

INDEX_DTYPE = np.dtype([
    ("offset", "<u8"),
    ("n_chans", "<u4"),
    ("big_rec_idx", "<u4"),
    ("sub_idx", "<u4"),
])
assert INDEX_DTYPE.itemsize == INDEX_ENTRY_SIZE


class BlockBufferedWriter:
    """Append-only file writer that issues aligned block-sized writes.

    The standard `io.BufferedWriter` flushes when the buffer is about to
    overflow, so for a 16 MiB buffer + 3 MiB chunks the actual OS writes
    are 15 MiB each — below the filesystem block, and slow. This writer
    flushes exactly `block_size` bytes each time, keeping the tail
    < block_size in memory until close.
    """

    def __init__(self, path: Path, block_size: int = WRITE_BLOCK_SIZE) -> None:
        self._fp = open(path, "wb", buffering=0)  # bypass the standard buffer
        self._block_size = block_size
        self._buf = bytearray()

    def write(self, data: bytes | bytearray | memoryview) -> None:
        self._buf.extend(data)
        while len(self._buf) >= self._block_size:
            self._fp.write(bytes(self._buf[: self._block_size]))
            del self._buf[: self._block_size]

    def close(self) -> None:
        if self._buf:
            self._fp.write(bytes(self._buf))
            self._buf.clear()
        self._fp.close()

    def __enter__(self) -> "BlockBufferedWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
