"""Config for the pre-shuffled REVE cache datamodule.

Reads pre-shuffled shards off the cluster's scratch space and yields
`{X, ch_pos, ch_padding_mask}` — the minimal set of fields the SSL/MAE
frameworks actually consume.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, DirectoryPath
from typing_extensions import Annotated

from eeg_fm_masking.configs.utils import InstantiatorConfig


class DatasetWrapperConfig(BaseModel):
    """Per-window scaling / channel-padding parameters for the REVE cache.

    Used by `ReveCacheDataModuleConfig.dataset_wrapper` and by the OEB wrapper
    `eeg_fm_masking.oeb.wrapper.ContextualEncoderBenchmarkWrapper` (via the
    fields `factor`, `scaler`, `clip_sigma`, `n_chans`).
    """

    model_config = ConfigDict(extra="forbid")

    ch_pos: bool = True
    ch_names: bool = False
    crop_inds: bool = False
    target: bool = False
    description_cols: list[str] = ["dataset", "subject", "session", "run", "task"]
    factor: float = 1e6
    scaler: Literal["none", "median_std", "median_std_clip", "chan_std", "std"] = "none"
    clip_sigma: float | None = None
    dtype: str = "float32"
    n_chans: int | None = 32
    seed: int | None = 12


class ReveCacheDataModuleConfig(InstantiatorConfig):
    """DataModule reading pre-shuffled REVE shards.

    `dataset_wrapper` is reused (rather than redeclaring its fields) so that
    the scaling/padding semantics stay identical to the standard datamodule.
    A validator enforces that only the `ch_pos=True` + `n_chans` + scaling
    parameters are used — every other wrapper field (target, ch_names,
    description_cols, etc.) must be disabled, because this cache only carries
    raw EEG windows and channel positions.
    """

    data_module_type: Literal["reve_cache"] = "reve_cache"

    # Cache directories (produced by the offline regeneration tooling).
    train_cache_dir: DirectoryPath
    val_cache_dir: DirectoryPath | None = None

    # Directory with `recording_-_positions_-_<big_rec_idx>.npy` files.
    # Positions are loaded eagerly at setup() (small: ~5 MB total).
    positions_dir: DirectoryPath

    # Reused for scaling / channel padding / dtype / rng seed — see validator.
    dataset_wrapper: Annotated[
        DatasetWrapperConfig, Field(default_factory=DatasetWrapperConfig)
    ]

    # DataLoader params. Defaults benchmarked at 3150 windows/s at 8 workers
    # on a single H100 with the cache on a fast shared filesystem.
    batch_size: int = 600
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    # Kept False so every epoch starts from a fresh DataLoader; combined with
    # PLTrainerConfig.reload_dataloaders_every_n_epochs=1 this ensures each
    # epoch gets a new deterministic shuffle seeded by (shuffle_seed, epoch).
    persistent_workers: bool = False

    # Seed for the per-epoch shard shuffle (separate from channel-subsampling
    # seed, which lives in dataset_wrapper.seed).
    shuffle_seed: int = 42

    @model_validator(mode="after")
    def _check_wrapper(self):
        w = self.dataset_wrapper
        # We only carry raw X + channel positions in the shard; metadata,
        # targets, channel names and crop indices are not stored. Enforce it
        # explicitly so misconfiguration fails loud at load time rather than
        # producing silently-missing batch keys.
        if not w.ch_pos:
            raise ValueError("dataset_wrapper.ch_pos must be True (ch_pos is required)")
        if w.ch_names:
            raise ValueError("dataset_wrapper.ch_names must be False (not stored in cache)")
        if w.crop_inds:
            raise ValueError("dataset_wrapper.crop_inds must be False (not stored in cache)")
        if w.target:
            raise ValueError("dataset_wrapper.target must be False (pre-training is label-free)")
        if w.description_cols:
            raise ValueError(
                "dataset_wrapper.description_cols must be empty "
                "(metadata is not stored in cache)"
            )
        # Collation requires a fixed channel count.
        if w.n_chans is None:
            raise ValueError("dataset_wrapper.n_chans must be set (required for collation)")
        return self

    def create_instance(self):
        from eeg_fm_masking.reve_cache.datamodule import ReveCacheDataModule

        return ReveCacheDataModule(self)
