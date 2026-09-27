import os
from pathlib import Path

from eeg_fm_masking.configs.train import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    TrainingConfig,
)
from eeg_fm_masking.configs.utils import PathInstantiatorConfig


def get_slurm_config(config: TrainingConfig) -> TrainingConfig:
    # Cluster-specific bits (partition, account, qos, etc.) are left empty
    # here — set them on `config.infra` before launching, e.g. via env vars
    # or per-call overrides.
    config.infra.folder = Path("~/.cache/exca/").expanduser()
    config.infra.cluster = "slurm"
    config.infra.job_name = "EMPTY-JOB-NAME"
    config.infra.mode = "force"  # cache is ignored
    config.infra.nodes = 1
    config.infra.slurm_partition = os.environ.get("SLURM_PARTITION", "")
    config.infra.timeout_min = 1
    config.infra._uid_string = "{method}_ver-{version}/{uid}"
    config.infra.slurm_additional_parameters = {
        "ntasks": 1,
        "gpus": 1,
        "signal": "SIGUSR1@300",  # for lightning to push the logs before the job is killed
    }
    config.trainer.accelerator = "gpu"
    return config


def get_slurm_clf_training_config(config: TrainingConfig) -> TrainingConfig:
    config = get_slurm_config(config)

    config.datamodule.num_workers = 16
    config.datamodule.pin_memory = True
    config.datamodule.persistent_workers = True

    config.trainer.callbacks = [
        PathInstantiatorConfig(
            class_path="lightning.pytorch.callbacks.ModelCheckpoint",
        ),
        PathInstantiatorConfig(
            class_path="lightning.pytorch.callbacks.ModelCheckpoint",
            init_args={
                "train_time_interval": "00:10:00",
                "enable_version_counter": False,
            },
        ),
        PathInstantiatorConfig(
            class_path="lightning.pytorch.callbacks.LearningRateMonitor"
        ),
    ]
    config.trainer.plugins = PathInstantiatorConfig(
        class_path="lightning.pytorch.plugins.environments.SLURMEnvironment",
        init_args={
            "auto_requeue": False,
            "requeue_signal": "SIGUSR1",
        },
        check_types=False,  # issue with enum
    )
    config.trainer.logger = PathInstantiatorConfig(
        class_path="lightning.pytorch.loggers.WandbLogger",
        init_args={
            "project": DEFAULT_PROJECT,
            "log_model": "all",  # log best https://docs.wandb.ai/guides/integrations/lightning#model-checkpointing
        },
        dict_kwargs={
            "entity": DEFAULT_ENTITY,
        },
    )

    return config.model_validate(config)
