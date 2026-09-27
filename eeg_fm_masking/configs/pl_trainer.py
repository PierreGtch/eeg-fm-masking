from datetime import timedelta
from typing import ClassVar

from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,  # type: ignore
)
from lightning.fabric.utilities.types import _PATH
import pydantic

from eeg_fm_masking.configs.utils import (
    PathInstantiatorConfig,
    InstantiatorConfig,
    instantiate_optional_list,
)


class PLTrainerConfig(InstantiatorConfig):
    model_config = pydantic.ConfigDict(extra="forbid")
    _exclude_from_cls_uid: ClassVar[tuple[str]] = ("logger",)

    accelerator: str = "auto"
    strategy: str = "auto"
    devices: list[int] | str | int = "auto"
    num_nodes: int = 1
    precision: _PRECISION_INPUT | None = None
    logger: list[PathInstantiatorConfig] | PathInstantiatorConfig | None = None
    callbacks: list[PathInstantiatorConfig] | PathInstantiatorConfig | None = None
    fast_dev_run: int | bool = False
    max_epochs: int | None = None
    min_epochs: int | None = None
    max_steps: int = -1
    min_steps: int | None = None
    max_time: str | timedelta | dict[str, int] | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    overfit_batches: int | float = 0.0
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    num_sanity_val_steps: int | None = None
    log_every_n_steps: int | None = 1
    enable_checkpointing: bool | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: int | float | None = None
    gradient_clip_algorithm: str | None = None
    deterministic: bool | _LITERAL_WARN | None = None
    benchmark: bool | None = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: str | None = None
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: list[PathInstantiatorConfig] | PathInstantiatorConfig | None = None
    sync_batchnorm: bool = False
    # Default to 1 so the DataModule's `train_dataloader()` is called at the
    # start of every epoch. Required by datamodules that rely on epoch-indexed
    # shuffle seeds (e.g. ReveCacheDataModule) — otherwise the epoch counter
    # in DataLoader workers stays frozen at fit-start.
    reload_dataloaders_every_n_epochs: int = 1
    default_root_dir: _PATH | None = None

    def create_instance(self):
        from lightning.pytorch import Trainer

        return Trainer(
            accelerator=self.accelerator,
            strategy=self.strategy,
            devices=self.devices,
            num_nodes=self.num_nodes,
            precision=self.precision,
            logger=instantiate_optional_list(self.logger),
            callbacks=instantiate_optional_list(self.callbacks),
            fast_dev_run=self.fast_dev_run,
            max_epochs=self.max_epochs,
            min_epochs=self.min_epochs,
            max_steps=self.max_steps,
            min_steps=self.min_steps,
            max_time=self.max_time,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            limit_test_batches=self.limit_test_batches,
            limit_predict_batches=self.limit_predict_batches,
            overfit_batches=self.overfit_batches,
            val_check_interval=self.val_check_interval,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            num_sanity_val_steps=self.num_sanity_val_steps,
            log_every_n_steps=self.log_every_n_steps,
            enable_checkpointing=self.enable_checkpointing,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=self.enable_model_summary,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            deterministic=self.deterministic,
            benchmark=self.benchmark,
            inference_mode=self.inference_mode,
            use_distributed_sampler=self.use_distributed_sampler,
            profiler=self.profiler,
            detect_anomaly=self.detect_anomaly,
            barebones=self.barebones,
            plugins=instantiate_optional_list(self.plugins),
            sync_batchnorm=self.sync_batchnorm,
            reload_dataloaders_every_n_epochs=self.reload_dataloaders_every_n_epochs,
            default_root_dir=self.default_root_dir,
        )
