from typing import Literal
from typing_extensions import Annotated

from pydantic import Field, model_validator

from eeg_fm_masking.configs.architectures import (
    EEGTransformerDecoderConfig,
    ContextualEncoderConfig,
)
from eeg_fm_masking.configs.utils import InstantiatorConfig


class SSLFrameworkConfig(InstantiatorConfig):
    framework_type: Literal["ssl"] = "ssl"

    model: Annotated[
        ContextualEncoderConfig,
        Field(default_factory=ContextualEncoderConfig),
    ]
    predictor: Annotated[
        EEGTransformerDecoderConfig,
        Field(default_factory=EEGTransformerDecoderConfig),
    ]

    lr: float = 1e-3
    final_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 0
    lr_scheduler_T_0: int | None = None  # if None, uses estimated_stepping_batches

    # number of teacher transformer layer outputs to average for creating the targets
    average_top_k_outputs: int = 1

    main_loss_weight: float = 1.0

    var_loss_weight: float = 1.0
    var_loss_target: float = 1.0

    cov_loss_weight: float = 1.0

    ema_decay: float = 0.999  # initial ema decay rate
    ema_end_decay: float = 0.9999  # final ema decay rate
    ema_anneal_end_step: int = 75000  # when to finish annealing ema decay rate

    log_grad_norms_every_n_steps: int | None = None

    debug: bool = False

    modelName: str = ""

    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None
    compile_backend: str = "inductor"
    compile_mode: str | None = None
    compile_options: dict[str, str | int | bool] | None = None
    compile_disable: bool = False

    def set_signal_length(self, signal_length: float):
        pass

    @model_validator(mode="after")
    def check_args(self):
        self.model = self.model.model_validate(self.model)
        assert self.average_top_k_outputs >= 1
        return self

    def create_instance(self):
        import torch
        import torch._dynamo
        # Raise dynamo's recompile cap (default 8): the ReveCache pipeline
        # specializes on enough shape/guard combinations to exceed it.
        torch._dynamo.config.cache_size_limit = 128
        from eeg_fm_masking.pl_ssl import SSLFramework

        compile_kwargs = {
            "fullgraph": self.compile_fullgraph,
            "dynamic": self.compile_dynamic,
            "backend": self.compile_backend,
            "mode": self.compile_mode,
            "options": self.compile_options,
            "disable": self.compile_disable,
        }
        SSLFramework._step_train = torch.compile(
            SSLFramework._step_train, **compile_kwargs
        )

        model, feature_encoder = self.model.create_instance()
        framework = SSLFramework(
            self,
            feature_encoder=feature_encoder,
            model=model,
            predictor=self.predictor.create_instance(),
        )
        return framework
