import math
from typing import Callable, Literal
from typing_extensions import Annotated

import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, ConfigDict, model_validator

from eeg_fm_masking.configs.utils import InstantiatorConfig


NormName = Literal["layer_norm", "rms_norm"]
_NORM_LAYERS: dict[NormName, type[nn.Module]] = {
    "layer_norm": nn.LayerNorm,
    "rms_norm": nn.RMSNorm,
}

ActivationName = Literal["relu", "gelu", "silu"]
_ACTIVATIONS: dict[ActivationName, Callable] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}

_DIM = 64


class LinearPatchEmbeddingConfig(InstantiatorConfig):
    """Config for :class:`~eeg_fm_masking.modules.LinearPatchEmbedding`.

    Implements the REVE-style patch tokenisation: each channel is unfolded into
    overlapping temporal patches which are then linearly projected to
    ``embed_dim``.
    """

    modelName: Literal["LinearPatchEmbedding"] = "LinearPatchEmbedding"
    dim: int = _DIM
    patch_size: int = 200  # 1 second at 200 Hz
    patch_overlap: int = 20  # 0.1 second overlap

    def create_instance(self):
        from eeg_fm_masking.modules import LinearPatchEmbedding

        return LinearPatchEmbedding(
            embed_dim=self.dim,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
        )


class EEGTransformerEncoderConfig(InstantiatorConfig):
    d_model: int = _DIM
    nhead: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    bias: bool = False
    norm: NormName = "layer_norm"
    activation: ActivationName = "relu"
    glu: bool = False
    num_layers: int = 2

    @model_validator(mode="after")
    def check_args(self):
        assert self.d_model % self.nhead == 0, (
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        )
        return self

    def create_instance(self):
        from eeg_fm_masking.transformer import (
            TransformerEncoderLayer,
            TransformerEncoder,
        )

        layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            bias=self.bias,
            norm_layer=_NORM_LAYERS[self.norm],
            activation=_ACTIVATIONS[self.activation],
            glu=self.glu,
        )
        return TransformerEncoder(
            encoder_layer=layer,
            num_layers=self.num_layers,
        )


class EEGTransformerDecoderConfig(InstantiatorConfig):
    modelName: Literal["TransformerDecoder"] = "TransformerDecoder"

    d_model: int = _DIM
    nhead: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    bias: bool = False
    norm: NormName = "layer_norm"
    activation: ActivationName = "relu"
    glu: bool = False
    num_layers: int = 2
    patch_size: int = 8

    # cross-attention input dim (encoder d_model); None = same as d_model
    memory_dim: int | None = None

    @model_validator(mode="after")
    def check_args(self):
        assert self.d_model % self.nhead == 0, (
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        )
        return self

    def create_instance(self):
        from eeg_fm_masking.modules import EEGTransformerDecoder

        return EEGTransformerDecoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            bias=self.bias,
            activation=_ACTIVATIONS[self.activation],
            glu=self.glu,
            norm_layer=_NORM_LAYERS[self.norm],
            num_layers=self.num_layers,
            patch_size=self.patch_size,
            memory_dim=self.memory_dim,
        )


class PositionalEncoderConfig(InstantiatorConfig):
    modelName: Literal["AdditivePositionalEncoder"] = "AdditivePositionalEncoder"
    spat_dim: int = _DIM // 4 * 3
    time_dim: int = _DIM - _DIM // 4 * 3
    sfreq_features: float = 16.0

    max_seconds: float = 600  # 10min
    max_x: float = 2  # using head_size=1

    @model_validator(mode="after")
    def check_args(self):
        assert self.spat_dim % 3 == 0, (
            f"spat_dim ({self.spat_dim}) must be divisible by 3 (x, y, z coordinates)"
        )
        return self

    def create_instance(self):
        from eeg_fm_masking.modules import PositionalEncoder

        return PositionalEncoder(
            spat_dim=self.spat_dim,
            time_dim=self.time_dim,
            sfreq_features=self.sfreq_features,
            max_seconds=self.max_seconds,
            max_x=self.max_x,
            init_buffer_n_times=33,
        )


class MaskMakerConfig(InstantiatorConfig):
    """Config for ``MaskMaker`` / ``MaskMakerVectorized``.

    ``radius_blocks`` supports two special values that bypass the
    spherical-cap formula:
      - ``0.0``: only the center channel is masked (single-channel block).
      - ``math.inf``: all (non-padding) channels are masked
        (full-spatial block).
    For both, ``scalp_surface`` is ignored when computing ``n_target_blocks``
    from ``pct_unmasked`` (any positive value satisfies the validator).
    """
    # for sfreq_features=16, at least 16 channels, and 30 seconds of data
    radius_blocks: float = 0.06  # 6cm radius
    length_blocks: int = 16 * 4
    pct_unmasked: float | None = 0.45
    scalp_surface: float | None = 4 * math.pi * (0.1**2) * 3 / 4  # 10cm radius
    n_target_blocks: int | None = None
    vectorized: bool = False

    @model_validator(mode="after")
    def check_args(self):
        assert (self.pct_unmasked is None) == (
            self.scalp_surface is None
        ), "pct_unmasked and scalp_surface must be both set or both None"
        assert (self.n_target_blocks is None) != (
            self.pct_unmasked is None
        ), "Can not specify both n_target_blocks and pct_unmasked+scalp_surface"
        if self.pct_unmasked is not None and self.scalp_surface is not None:
            assert 0 <= self.pct_unmasked <= 1, "pct_unmasked should be in [0, 1]"
            radius_is_special = (
                self.radius_blocks == 0.0 or math.isinf(self.radius_blocks)
            )
            if not radius_is_special:
                covered_surface = 2 * math.pi * self.radius_blocks**2
                assert covered_surface < self.scalp_surface, (
                    "The mask can not cover more than 100% of the scalp. "
                    "Decrease radius_blocks or increase scalp_surface. Possible unit mismatch"
                )
        return self

    def create_instance(self):
        from eeg_fm_masking.modules import MaskMaker, MaskMakerVectorized

        cls = MaskMakerVectorized if self.vectorized else MaskMaker
        return cls(
            n_target_blocks=self.n_target_blocks,
            pct_unmasked=self.pct_unmasked,
            radius_blocks=self.radius_blocks,
            length_blocks=self.length_blocks,
            scalp_surface=self.scalp_surface,
        )


class ContextualEncoderConfig(InstantiatorConfig):
    feature_encoder: Annotated[
        LinearPatchEmbeddingConfig,
        Field(default_factory=LinearPatchEmbeddingConfig),
    ]
    transformer: Annotated[
        EEGTransformerEncoderConfig,
        Field(default_factory=EEGTransformerEncoderConfig),
    ]
    masker: Annotated[MaskMakerConfig | None, Field(default_factory=MaskMakerConfig)]
    pos_encoder: Annotated[
        PositionalEncoderConfig,
        Field(default_factory=PositionalEncoderConfig),
    ]
    shared_feature_encoder: bool = True

    def set_dim(self, dim: int):
        """Set the number of dimensions of the models embeddings."""
        self.feature_encoder.dim = dim
        self.transformer.d_model = dim
        self.pos_encoder.spat_dim = dim // 4 * 3
        self.pos_encoder.time_dim = dim - dim // 4 * 3

    @property
    def dim(self):
        return self.feature_encoder.dim

    @model_validator(mode="after")
    def check_args(self):
        dim = self.feature_encoder.dim
        assert self.pos_encoder.spat_dim + self.pos_encoder.time_dim == dim, (
            f"spat_dim={self.pos_encoder.spat_dim}, "
            f"time_dim={self.pos_encoder.time_dim}"
        )
        assert (
            self.transformer.d_model == dim
        ), f"transformer.d_model={self.transformer.d_model}"
        return self

    def create_instance(self):
        from eeg_fm_masking.models import ContextualEncoder

        feature_encoder = self.feature_encoder.create_instance()
        if self.shared_feature_encoder:
            shared_feature_encoder, feature_encoder = feature_encoder, None
        else:
            shared_feature_encoder = None
        masker = self.masker.create_instance() if self.masker is not None else None
        contextual_encoder = ContextualEncoder(
            feature_encoder=feature_encoder,
            pos_encoder=self.pos_encoder.create_instance(),
            transformer=self.transformer.create_instance(),
            masker=masker,
        )
        return contextual_encoder, shared_feature_encoder
