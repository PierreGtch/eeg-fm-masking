"""REVE-Small default configs (MAE and JEPA, both with additive PE).

Architecture and hyperparameters follow the REVE paper (Table 5 + Table 6,
REVE-Small row): 4 layers, 8 heads, dim=512, GEGLU FFN with 8/3 expansion
ratio, RMSNorm, peak LR=2.4e-4, no auxiliary var/cov regularization.
The positional encoder is the local AdditivePositionalEncoder, not the
REVE 4D Fourier PE.
"""
import math

from eeg_fm_masking.configs.architectures import (
    EEGTransformerEncoderConfig,
    EEGTransformerDecoderConfig,
    ContextualEncoderConfig,
    LinearPatchEmbeddingConfig,
    MaskMakerConfig,
    PositionalEncoderConfig,
)
from eeg_fm_masking.configs.pl_mae import MAEFrameworkConfig
from eeg_fm_masking.configs.pl_ssl import SSLFrameworkConfig

# Shared constants for the REVE dataset
_SFREQ = 200
_PS = 200  # patch size: 1s
_OVERLAP = 20  # overlap between patches: 0.1s
_SFREQ_FEATURES = _SFREQ / (_PS - _OVERLAP)

# REVE-Small architecture (REVE paper, Table 6)
_NUM_LAYERS = 4
_NHEAD = 8
_NUM_DEC_LAYERS = 2

# REVE-Small training (REVE paper, Table 5)
_LR = 2.4e-4


def _ffn_dim(dim: int) -> int:
    """LLaMA-style 8/3 expansion ratio used by REVE."""
    return dim * 8 // 3


def _make_encoder(DIM: int, pos_encoder) -> ContextualEncoderConfig:
    """REVE-Small contextual encoder, shared across MAE/JEPA/RoPE variants."""
    return ContextualEncoderConfig(
        feature_encoder=LinearPatchEmbeddingConfig(
            dim=DIM,
            patch_size=_PS,
            patch_overlap=_OVERLAP,
        ),
        pos_encoder=pos_encoder,
        masker=MaskMakerConfig(
            pct_unmasked=0.45,  # 55% masked
            scalp_surface=4 * math.pi * (0.1**2) * 3 / 4,  # 10cm radius
            radius_blocks=0.06,  # 6cm (REVE used 3cm but seems too small)
            length_blocks=int(_SFREQ_FEATURES * 3.0),  # 3s
            vectorized=True,
        ),
        transformer=EEGTransformerEncoderConfig(
            num_layers=_NUM_LAYERS,
            nhead=_NHEAD,
            d_model=DIM,
            dim_feedforward=_ffn_dim(DIM),
            dropout=0.0,
            criss_cross_attn=False,
            bias=False,
            norm="rms_norm",
            activation="gelu",
            glu=True,
        ),
    )


def _make_decoder(DIM: int, patch_size: int) -> EEGTransformerDecoderConfig:
    """REVE-Small decoder/predictor, shared across MAE/JEPA variants.

    For MAE, ``patch_size`` is the raw signal patch length (so the head
    reconstructs the EEG signal). For JEPA, ``patch_size=DIM`` so the head
    predicts a latent embedding of the same dimension as the encoder output.
    """
    return EEGTransformerDecoderConfig(
        d_model=DIM,
        nhead=_NHEAD,
        dim_feedforward=_ffn_dim(DIM),
        num_layers=_NUM_DEC_LAYERS,
        patch_size=patch_size,
        dropout=0.0,
        bias=False,
        norm="rms_norm",
        activation="gelu",
        glu=True,
    )


def _additive_pe(DIM: int) -> PositionalEncoderConfig:
    return PositionalEncoderConfig(
        spat_dim=DIM // 4 * 3,
        time_dim=DIM - DIM // 4 * 3,
        sfreq_features=_SFREQ_FEATURES,
        max_x=0.15,  # 15cm
    )


def get_reve_small_config_config(DIM=512) -> MAEFrameworkConfig:
    """REVE-Small MAE config with additive PE."""
    return MAEFrameworkConfig(
        model=_make_encoder(DIM, _additive_pe(DIM)),
        predictor=_make_decoder(DIM, patch_size=_PS),
        lr=_LR,
        var_loss_weight=0.0,
        cov_loss_weight=0.0,
        compile_fullgraph=True,
        compile_dynamic=False,
        compile_disable=False,
        compile_mode=None,
    )


def get_reve_small_jepa_config(DIM=512) -> SSLFrameworkConfig:
    """REVE-Small JEPA config (EMA teacher) with additive PE.

    Reuses the same encoder, masking, optimizer and loss weights as the
    REVE-Small MAE config. The MAE reconstruction head is replaced by a
    predictor that targets the EMA teacher's latent embeddings, so its
    output dimension is ``DIM`` (latent) instead of ``_PS`` (raw signal).
    """
    return SSLFrameworkConfig(
        model=_make_encoder(DIM, _additive_pe(DIM)),
        predictor=_make_decoder(DIM, patch_size=DIM),
        lr=_LR,
        var_loss_weight=0.0,
        cov_loss_weight=0.0,
        compile_fullgraph=True,
        compile_dynamic=False,
        compile_disable=False,
        compile_mode=None,
    )


