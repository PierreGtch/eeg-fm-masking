"""OpenEEGBench wrapper for ContextualEncoder (MAE pretrained).

Adapts our pretrained ``ContextualEncoder`` to the OpenEEGBench model interface::

    model = ContextualEncoderBenchmarkWrapper(
        n_chans, n_times, n_outputs, sfreq, chs_info,
        scaler="chan_std", factor=1e6,
        model_kwargs={...},
    )
    logits = model(x)  # x: (B, C, T) -> (B, n_outputs)
    # model.final_layer is replaced by OpenEEGBench per finetuning strategy
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from eeg_fm_masking.functions import scale_signal


class ContextualEncoderBenchmarkWrapper(nn.Module):
    """OpenEEGBench-compatible wrapper for a pretrained ContextualEncoder.

    The architecture is reconstructed from ``model_kwargs``
    (``run.config['framework']['model']`` from wandb).  Weight loading is
    performed by OEB's ``load_pretrained()`` after ``__init__`` — the wrapper's
    ``self.feature_encoder`` and ``self.model`` attribute names match the
    checkpoint key prefixes.  Extra keys (predictor, masker) are silently
    skipped by OEB's shape-filtered loading.

    Args:
        n_chans: Number of EEG channels in the dataset.
        n_times: Number of time samples per window.
        n_outputs: Number of output classes.
        sfreq: Sampling frequency in Hz.
        chs_info: List of MNE channel-info dicts (``loc[:3]`` = x/y/z in metres).
        scaler: Normalization method matching the pretraining
            ``DatasetWrapperConfig.scaler``.  Must be passed explicitly.
        factor: Multiplicative factor applied before the scaler, matching
            ``DatasetWrapperConfig.factor``.
        clip_sigma: Clip threshold (in std units) for ``median_std_clip``.
        **model_config: The ``ContextualEncoderConfig`` fields from wandb,
            unpacked by OpenEEGBench's ``PretrainedBackbone`` from
            ``model_kwargs``.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info: list[dict] | None = None,
        *,
        scaler: Literal["none", "median_std", "median_std_clip", "chan_std", "std"],
        factor: float = 1e6,
        clip_sigma: float | None = None,
        **model_config: Any,
    ):
        super().__init__()

        self._scaler = scaler
        self._factor = factor
        self._clip_sigma = clip_sigma

        ch_pos = self._extract_ch_pos(chs_info, n_chans)  # (n_chans, 3)
        self.register_buffer("_ch_pos", ch_pos)

        from eeg_fm_masking.configs.architectures import ContextualEncoderConfig
        from eeg_fm_masking.models import ContextualEncoder

        model_cfg = ContextualEncoderConfig.model_validate(model_config)

        encoder, feature_encoder = model_cfg.create_instance()
        encoder = ContextualEncoder.to_maskless_arch(encoder)

        self.feature_encoder = feature_encoder
        self.model = encoder

        # OpenEEGBench replaces self.final_layer (via setattr) per finetuning
        # strategy. We still create a valid Linear so the model is usable as-is.
        d_model = model_cfg.dim
        patch_size = feature_encoder.linear.in_features
        n_patches = n_times // patch_size
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_chans * n_patches * d_model, n_outputs),
        )

    @staticmethod
    def _extract_ch_pos(chs_info: list[dict] | None, n_chans: int) -> torch.Tensor:
        """Return ``(n_chans, 3)`` float32 XYZ positions from MNE ``chs_info``."""
        if not chs_info:
            raise ValueError(
                "Channel info list is empty or None; cannot extract positions."
            )

        if len(chs_info) != n_chans:
            raise ValueError(
                f"Length of chs_info ({len(chs_info)}) does not match n_chans ({n_chans})."
            )

        positions = [ch["loc"][:3] for ch in chs_info[:n_chans]]
        if not all(len(pos) == 3 for pos in positions):
            raise ValueError(
                "All channel 'loc' entries must have at least 3 elements for XYZ positions."
            )
        if any(any(np.isnan(pos)) for pos in positions):
            raise ValueError("Channel positions contain NaN values; cannot proceed.")

        return torch.from_numpy(np.array(positions, dtype=np.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an EEG batch and return class logits.

        Args:
            x: ``(B, C, T)`` raw EEG signal (float32).

        Returns:
            ``(B, n_outputs)`` logits.
        """
        x = scale_signal(x, self._factor, self._scaler, self._clip_sigma)  # type: ignore[assignment]

        B = x.shape[0]
        ch_pos = self._ch_pos.unsqueeze(0).expand(B, -1, -1)  # (B, C, 3)
        local_features = self.feature_encoder(x)  # (B, C, n_patches, d_model)

        batch = {"X": x, "ch_pos": ch_pos, "local_features": local_features}
        out = self.model(batch, apply_mask=False)

        return self.final_layer(out["ctx_features"])
