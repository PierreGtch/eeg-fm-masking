"""Pre-shuffled REVE cache.

The cache is a directory of paired shard files (`shard_NNNNN.dat` +
`shard_NNNNN.idx`). Windows are shuffled across shards at generation time,
so a simple sequential read per shard already yields a fully diverse
mini-batch while keeping I/O sequential.

  - Reader (used at training time): `dataset.ShuffledShardDataset`
  - Writer (used by `regenerate_reve_cache.py` at the repo root):
    `generate.generate_cache_parallel` + `writer.ShardWriter`
"""
from eeg_fm_masking.reve_cache.dataset import ShuffledShardDataset

__all__ = ["ShuffledShardDataset"]
