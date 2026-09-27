"""Shard reader: loads a shard into RAM via two sequential reads.

For the dataset we use `read_shard_flat`: it issues one `np.fromfile` per file,
which goes straight to a numpy-owned buffer and bypasses Python's `BufferedReader`
(8 KiB default buffer — kills throughput on multi-GiB files). Record accesses
are then zero-copy slices on the flat array.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_fm_masking.reve_cache.format import INDEX_DTYPE, WINDOW_SAMPLES


def read_shard_flat(dat_path: Path, idx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read .dat as a single flat float32 array + .idx as a structured array.

    `np.fromfile` issues one fread-loop directly into a numpy-owned buffer,
    bypassing Python's BufferedReader. Record accesses are then plain slices
    on the flat array (zero-copy views) + reshape (zero-copy). The per-record
    `.copy()` is paid only at the DataLoader IPC boundary (pickle of the view).
    """
    index = np.fromfile(idx_path, dtype=INDEX_DTYPE)
    flat = np.fromfile(dat_path, dtype=np.float32)
    if len(index) > 0:
        last = index[-1]
        last_end_bytes = int(last["offset"]) + int(last["n_chans"]) * WINDOW_SAMPLES * 4
        assert last_end_bytes <= flat.nbytes, f"orphan data in {dat_path}"
    return flat, index
