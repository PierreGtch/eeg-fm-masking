"""Mask-strategy sweep: 2 models × 29 masking configurations, 10 epochs each.

Full grid sweep over (radius_blocks, length_blocks):

  radius_blocks  ∈ {0.0, 0.06, 0.09, 0.12, math.inf}
  length_blocks  ∈ {1, 2, 4, 8, 16, N_TIME_PATCHES=33}

Full Cartesian product = 30 combinations. One cell is removed:

  * (radius_blocks=math.inf, length_blocks=N_TIME_PATCHES): would mask the
    whole example (all channels × all time), no learning possible. -1.

Net = 29 mask configurations × 2 models = 58 jobs.

Training setup:
  - 2 GPUs DDP (find_unused for JEPA's ema_model/zero-weight losses).
  - max_epochs=10, seed=42, bf16-mixed, gradient clip 3.0 (norm) for both.
  - Warmup = 1 epoch (STEPS_PER_EPOCH=3080); lr_scheduler_T_0 = MAX_EPOCHS *
    STEPS_PER_EPOCH (exactly; a smaller T_0 would let CosineAnnealingLR pass
    its minimum and spike the LR back up at the end).
  - ModelCheckpoint save_top_k=-1, every_n_epochs=1 (keep all 10 epoch ckpts).
  - pct_unmasked + scalp_surface, n_target_blocks=None (inclusion-exclusion
    K formula). All masks vectorized=True.
  - All 58 jobs share SLURM resources (2 GPUs, 16 cpus/task, 10h) and go in
    a single `infra.job_array()` submission.

Mask geometry (computed for our pipeline, sfreq_features = 10/9 Hz, 30 s
window, 32 channels):
  - MASK_RATIO = 0.55, PCT_UNMASKED = 0.45
  - N_TIME_PATCHES = int(sfreq × 30 s) = 33
  - N_CHANNELS = 32
  - SCALP_SURFACE = 4π × (0.1)² × 3/4

Cluster paths (REVE shard cache, channel positions, wandb/lightning logs)
are read from environment variables — see the constants section below.
"""

import os
import math
import subprocess
from pathlib import Path

os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
    Path("~/.cache/torchinductor").expanduser().as_posix()
)

# Override via WANDB_CACHE_ROOT to keep wandb / Lightning state off a
# low-quota home filesystem.
WANDB_CACHE_ROOT = os.environ.get(
    "WANDB_CACHE_ROOT",
    str(Path("~/.cache/wandb_cache").expanduser()),
)
os.environ.setdefault("WANDB_DIR",          f"{WANDB_CACHE_ROOT}/runs")
os.environ.setdefault("WANDB_CACHE_DIR",    f"{WANDB_CACHE_ROOT}/cache")
os.environ.setdefault("WANDB_ARTIFACT_DIR", f"{WANDB_CACHE_ROOT}/artifacts")
os.environ.setdefault("WANDB_LOCK_DIR",     f"{WANDB_CACHE_ROOT}/locks")
for _d in (
    os.environ["WANDB_DIR"],
    os.environ["WANDB_CACHE_DIR"],
    os.environ["WANDB_ARTIFACT_DIR"],
    os.environ["WANDB_LOCK_DIR"],
):
    Path(_d).mkdir(parents=True, exist_ok=True)
# Lightning's local mirror (`<save_dir>/<project>/<run_id>/checkpoints/`)
# is NOT controlled by WANDB_DIR — it's a WandbLogger init_arg, set in
# make_config() below.
WANDB_LIGHTNING_SAVE_DIR = f"{WANDB_CACHE_ROOT}/lightning_logs"
Path(WANDB_LIGHTNING_SAVE_DIR).mkdir(parents=True, exist_ok=True)

from eeg_fm_masking.configs.train import TrainingConfig
from eeg_fm_masking.configs.pl_trainer import PLTrainerConfig
from eeg_fm_masking.configs.models_defaults import (
    get_reve_small_config_config,  # MAE
    get_reve_small_jepa_config,    # JEPA
)
from eeg_fm_masking.configs.slurm import get_slurm_clf_training_config
from eeg_fm_masking.configs.utils import PathInstantiatorConfig
from eeg_fm_masking.configs.architectures import MaskMakerConfig
from eeg_fm_masking.reve_cache.config import (
    DatasetWrapperConfig,
    ReveCacheDataModuleConfig,
)


# ============================================================
# CONSTANTS
# ============================================================

DIM = 512
N_GPUS = 2
BATCH_SIZE = 600   # per-GPU
NUM_WORKERS = 8    # per-GPU
MAX_EPOCHS = 10
TIMEOUT_MIN = 600  # 10h SLURM budget
SCALER = "median_std_clip"
CLIP_SIGMA = 15

# Verified by reading every .idx file under TRAIN_CACHE_DIR (see reader.py,
# 20-byte INDEX_DTYPE entries).
N_TRAIN_EXAMPLES = 3_695_009
STEPS_PER_EPOCH = math.ceil(N_TRAIN_EXAMPLES / (BATCH_SIZE * N_GPUS))  # = 3080
WARMUP_STEPS = STEPS_PER_EPOCH  # 1 epoch of warmup
# T_0 must match the total number of optimizer steps EXACTLY. If T_0 < actual
# steps, CosineAnnealingLR passes its minimum and the LR rises again over the
# last steps, wrecking the final weights.
LR_SCHEDULER_T_0 = MAX_EPOCHS * STEPS_PER_EPOCH  # = 30800

# Mask-strategy geometry (computed for our pipeline)
SFREQ_FEATURES = 200 / 180  # = 10/9, matches models_defaults._SFREQ_FEATURES
N_CHANNELS = 32
N_TIME_PATCHES = int(SFREQ_FEATURES * 30)  # = 33
MASK_RATIO = 0.55
PCT_UNMASKED = 1.0 - MASK_RATIO            # = 0.45
SCALP_SURFACE = 4 * math.pi * (0.1**2) * 3 / 4  # 10cm hemisphere area

# JEPA-specific defaults (no-reg variant)
VAR_LOSS_WEIGHT = 0.0
COV_LOSS_WEIGHT = 0.0
VAR_LOSS_TARGET = 1.0
EMA_DECAY = 0.999
EMA_END_DECAY = 1.0
EMA_ANNEAL_END_STEP = 30000
AVG_TOP_K = 1
GRADIENT_CLIP = 3.0

# REVE pre-shuffled shard cache (training data) and channel-position lookup
# directory. Both must be on a fast shared filesystem on the cluster nodes.
# Set the corresponding env vars before launching.
TRAIN_CACHE_DIR = Path(
    os.environ.get("REVE_TRAIN_CACHE_DIR", "PLEASE_SET_REVE_TRAIN_CACHE_DIR")
)
POSITIONS_DIR = Path(
    os.environ.get("REVE_POSITIONS_DIR", "PLEASE_SET_REVE_POSITIONS_DIR")
)


# ============================================================
# MASK STRATEGIES — full grid (29 configs)
# ============================================================

# Cartesian product of these axes, minus 1 exclusion (see below).
RADIUS_BLOCKS_GRID = [0.0, 0.06, 0.09, 0.12, math.inf]
LENGTH_BLOCKS_GRID = [1, 2, 4, 8, 16, N_TIME_PATCHES]

# (radius, length) combinations to skip:
#   - (math.inf, N_TIME_PATCHES): masks the whole example, no learning possible.
SKIP_COMBINATIONS = {
    (math.inf, N_TIME_PATCHES),  # whole example
}


def _radius_label(r: float) -> str:
    if r == 0.0:
        return "r0"
    if r == math.inf:
        return "rinf"
    # 0.06 -> "r006", 0.09 -> "r009", 0.12 -> "r012"
    return f"r{int(round(r * 100)):03d}"


def _length_label(length: int) -> str:
    return f"l{length:02d}"


# All configs use n_target_blocks=None: MaskMaker computes K via the
# inclusion-exclusion formula from pct_unmasked + the per-block coverage f.
MASK_STRATEGIES = []
for _r in RADIUS_BLOCKS_GRID:
    for _length in LENGTH_BLOCKS_GRID:
        if (_r, _length) in SKIP_COMBINATIONS:
            continue
        MASK_STRATEGIES.append(
            dict(
                name=f"{_radius_label(_r)}_{_length_label(_length)}",
                radius_blocks=_r,
                length_blocks=_length,
                pct_unmasked=PCT_UNMASKED,
                scalp_surface=SCALP_SURFACE,
                n_target_blocks=None,
                vectorized=True,
            )
        )

assert len(MASK_STRATEGIES) == 29, (
    f"Expected 29 mask configs, got {len(MASK_STRATEGIES)}"
)


# ============================================================
# CONFIG BUILDER
# ============================================================


def make_framework(model_kind: str):
    """Return a framework config for 'mae' or 'jepa_noreg'."""
    if model_kind == "mae":
        cfg = get_reve_small_config_config(DIM=DIM)
    elif model_kind == "jepa_noreg":
        cfg = get_reve_small_jepa_config(DIM=DIM)
        cfg.var_loss_weight = VAR_LOSS_WEIGHT
        cfg.var_loss_target = VAR_LOSS_TARGET
        cfg.cov_loss_weight = COV_LOSS_WEIGHT
        cfg.ema_decay = EMA_DECAY
        cfg.ema_end_decay = EMA_END_DECAY
        cfg.ema_anneal_end_step = EMA_ANNEAL_END_STEP
        cfg.average_top_k_outputs = AVG_TOP_K
    else:
        raise ValueError(f"Unknown model_kind: {model_kind!r}")

    cfg.compile_mode = None  # standard inductor
    cfg.lr_scheduler_T_0 = LR_SCHEDULER_T_0
    cfg.warmup_steps = WARMUP_STEPS
    return cfg


def make_config(model_kind: str, mask: dict):
    """Build a TrainingConfig for one (model × mask) combination."""
    framework_cfg = make_framework(model_kind)
    framework_cfg.model.masker = MaskMakerConfig(
        radius_blocks=mask["radius_blocks"],
        length_blocks=mask["length_blocks"],
        pct_unmasked=mask["pct_unmasked"],
        scalp_surface=mask["scalp_surface"],
        n_target_blocks=mask["n_target_blocks"],
        vectorized=mask["vectorized"],
    )

    trainer_kwargs = dict(
        max_epochs=MAX_EPOCHS,
        precision="bf16-mixed",
        check_val_every_n_epoch=0,
        strategy="ddp_find_unused_parameters_true",  # JEPA needs it; harmless for MAE
        devices=N_GPUS,
        num_nodes=1,
        gradient_clip_val=GRADIENT_CLIP,
        gradient_clip_algorithm="norm",
    )

    config = TrainingConfig.model_construct(
        seed=42,
        trainer=PLTrainerConfig(**trainer_kwargs),
        framework=framework_cfg,
        datamodule=ReveCacheDataModuleConfig(
            train_cache_dir=TRAIN_CACHE_DIR,
            val_cache_dir=None,
            positions_dir=POSITIONS_DIR,
            dataset_wrapper=DatasetWrapperConfig(
                ch_pos=True,
                ch_names=False,
                crop_inds=False,
                target=False,
                description_cols=[],
                n_chans=N_CHANNELS,
                scaler=SCALER,
                clip_sigma=CLIP_SIGMA,
            ),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        ),
    )

    config = get_slurm_clf_training_config(config)
    config.infra.timeout_min = TIMEOUT_MIN
    config.infra.job_name = f"mask_sweep_full_{model_kind}_{mask['name']}"
    config.infra.tasks_per_node = N_GPUS
    config.infra.slurm_use_srun = True
    config.infra.cpus_per_task = 16
    config.infra.slurm_additional_parameters = {
        **(config.infra.slurm_additional_parameters or {}),
        "ntasks": N_GPUS,
        "gpus": N_GPUS,
    }

    wandb_logger = config.trainer.logger
    # Lightning's local checkpoint mirror is not covered by the WANDB_*
    # env vars; force it under WANDB_CACHE_ROOT too.
    wandb_logger.init_args["save_dir"] = WANDB_LIGHTNING_SAVE_DIR
    csv_logger = PathInstantiatorConfig(
        class_path="lightning.pytorch.loggers.CSVLogger",
        init_args={"save_dir": "PLACEHOLDER", "name": "PLACEHOLDER"},
    )
    config.trainer.logger = [wandb_logger, csv_logger]

    callbacks = [
        PathInstantiatorConfig(
            class_path="lightning.pytorch.callbacks.ModelCheckpoint",
            init_args={
                "save_top_k": -1,
                "every_n_epochs": 1,
                "save_on_train_epoch_end": True,
            },
        ),
        PathInstantiatorConfig(
            class_path="lightning.pytorch.callbacks.LearningRateMonitor",
        ),
    ]
    if model_kind == "jepa_noreg":
        callbacks.append(
            PathInstantiatorConfig(
                class_path="eeg_fm_masking.collapse_detector.CollapseDetectorCallback",
            )
        )
    config.trainer.callbacks = callbacks

    config = config.model_validate(config)
    return config


# ============================================================
# SUBMIT
# ============================================================

MODELS = ("mae", "jepa_noreg")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    retry = "--retry" in sys.argv  # re-submit only failed/missing tasks

    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()

    configs = []
    for model_kind in MODELS:
        for mask in MASK_STRATEGIES:
            cfg = make_config(model_kind, mask)
            if force:
                cfg.infra.mode = "force"
            elif retry:
                cfg.infra.mode = "retry"

            name = f"mask_sweep_full_{model_kind}_{mask['name']}_{commit_hash}"
            for logger in cfg.trainer.logger:
                if isinstance(logger, PathInstantiatorConfig):
                    logger.init_args["name"] = name
                    if "CSVLogger" in str(logger.class_path):
                        folder = cfg.infra.uid_folder()
                        assert folder is not None
                        logger.init_args["save_dir"] = str(folder / "csv_logs")
            configs.append((name, cfg))

    leader = configs[0][1]
    with leader.infra.job_array() as array:
        array.extend(cfg for _, cfg in configs)

    for name, cfg in configs:
        job = cfg.infra.job()
        folder = cfg.infra.uid_folder()
        print(f"\n--- Submitted: {name} ---")
        print(f"  job={job}")
        print(f"  folder={folder}")
