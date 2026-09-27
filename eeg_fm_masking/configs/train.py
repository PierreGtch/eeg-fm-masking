import os
from typing_extensions import Annotated
from pathlib import Path


from pydantic import Field, model_validator
from exca import TaskInfra

from eeg_fm_masking.configs.pl_trainer import PLTrainerConfig
from eeg_fm_masking.configs.pl_ssl import SSLFrameworkConfig
from eeg_fm_masking.configs.pl_mae import MAEFrameworkConfig
from eeg_fm_masking.configs.utils import InstantiatorConfig
from eeg_fm_masking.reve_cache.config import ReveCacheDataModuleConfig

FrameworkConfig = Annotated[
    SSLFrameworkConfig | MAEFrameworkConfig,
    Field(discriminator="framework_type"),
]

# Wandb entity / project. Override via env vars before launching to avoid
# accidental writes to the wrong workspace.
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "eeg-fm-masking")


class TrainingConfig(InstantiatorConfig):
    infra: TaskInfra = TaskInfra(version="0")

    trainer: Annotated[
        PLTrainerConfig,
        Field(default_factory=lambda: PLTrainerConfig(max_steps=10000)),
    ]
    framework: FrameworkConfig
    datamodule: ReveCacheDataModuleConfig

    seed: int = 42
    debug: bool = False

    wandb_entity: str = DEFAULT_ENTITY
    wandb_project: str = DEFAULT_PROJECT
    wandb_run_id: str | None = None
    wandb_version: str = "latest"

    @property
    def checkpoint_ref(self):
        if self.wandb_run_id is None:
            return None
        return f"{self.wandb_entity}/{self.wandb_project}/model-{self.wandb_run_id}:{self.wandb_version}"

    @model_validator(mode="after")
    def check_args(self):
        # late validation of the datamodule (because of model_construct)
        self.datamodule = self.datamodule.model_validate(self.datamodule)
        self.framework = self.framework.model_validate(self.framework)
        return self

    def create_instance(self):
        from lightning.pytorch.loggers import WandbLogger

        trainer = self.trainer.create_instance()

        ckpt_path = None
        if self.checkpoint_ref is not None:
            print(f"Resuming from checkpoint {self.checkpoint_ref}")
            if isinstance(logger := trainer.logger, WandbLogger):
                artifact = logger.experiment.use_artifact(
                    self.checkpoint_ref, use_as="resume"
                )
            else:
                import wandb

                api = wandb.Api()
                artifact = api.artifact(self.checkpoint_ref, type="model")
            artifact_dir = artifact.download()
            assert artifact_dir is not None
            ckpt_path = Path(artifact_dir) / "model.ckpt"

        framework = self.framework.create_instance()
        datamodule = self.datamodule.create_instance()

        return framework, datamodule, trainer, ckpt_path

    @infra.apply
    def train(self):
        config_str = self.infra.config(exclude_defaults=True).to_yaml()
        print(config_str)

        # pylint: disable=import-outside-toplevel
        from mne import set_log_level  # type: ignore
        import lightning.pytorch as pl
        from lightning.pytorch.loggers import WandbLogger

        set_log_level("WARNING")

        pl.seed_everything(self.seed, workers=True)

        framework, datamodule, trainer, ckpt_path = self.create_instance()

        for logger in trainer.loggers:
            logger.log_hyperparams(self.infra.config(uid=True, exclude_defaults=False))

        if isinstance(trainer.logger, WandbLogger) and self.debug:
            trainer.logger.watch(framework, log="all", log_freq=1, log_graph=False)

        trainer.fit(model=framework, datamodule=datamodule, ckpt_path=ckpt_path)
        return None  # avoid filling-up wandb cache with checkpoints already stored on wandb

    @classmethod
    def from_wandb(
        cls,
        run_id: str,
        entity=DEFAULT_ENTITY,
        project=DEFAULT_PROJECT,
        version="latest",
    ):
        import wandb
        import copy
        from lightning.pytorch.loggers import WandbLogger

        api = wandb.Api()
        run_ref = f"{entity}/{project}/{run_id}"
        run = api.run(run_ref)

        config_dict = copy.deepcopy(run.config)

        config = cls.model_validate(config_dict)
        config.wandb_entity = entity
        config.wandb_project = project
        config.wandb_run_id = run_id
        config.wandb_version = version
        return config


