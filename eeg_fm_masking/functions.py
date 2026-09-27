from typing import Any
import math

import torch
import numpy as np
from numpy.typing import NDArray


def scale_signal(
    X: NDArray | torch.Tensor,
    factor: float,
    scaler: str,
    clip_sigma: float | None = None,
) -> NDArray | torch.Tensor:
    """Apply factor scaling and per-window normalization to an EEG signal.

    Works with numpy arrays (used in the dataloader) and torch tensors
    (used in the OEB wrapper).

    Parameters
    ----------
    X : array (..., channels, time)
    factor : multiplicative factor (e.g. 1e6 for V→µV).
    scaler : one of ``"none"``, ``"chan_std"``, ``"median_std"``,
        ``"median_std_clip"``, ``"std"``.
    clip_sigma : required iff ``scaler == "median_std_clip"``.  After
        dividing by the median-of-channel-stds, values are clipped to
        ``[-clip_sigma, +clip_sigma]`` (units of the per-window median std).
    """
    X = X * factor
    if scaler == "none":
        return X
    is_torch = isinstance(X, torch.Tensor)
    eps = 1e-6
    if is_torch:
        std = X.std(dim=-1, keepdim=True, correction=0)
    else:
        std = np.std(X, axis=-1, keepdims=True)
    if scaler == "chan_std":
        return X / (std + eps)
    if is_torch:
        if scaler == "median_std":
            return X / (std.median(dim=-2, keepdim=True).values + eps)
        if scaler == "median_std_clip":
            if clip_sigma is None:
                raise ValueError("median_std_clip requires clip_sigma.")
            X = X / (std.median(dim=-2, keepdim=True).values + eps)
            return X.clamp(-clip_sigma, clip_sigma)
        if scaler == "std":
            return X / ((std**2).mean(dim=-2, keepdim=True).sqrt() + eps)
    else:
        if scaler == "median_std":
            return X / (np.median(std, axis=-2, keepdims=True) + eps)
        if scaler == "median_std_clip":
            if clip_sigma is None:
                raise ValueError("median_std_clip requires clip_sigma.")
            X = X / (np.median(std, axis=-2, keepdims=True) + eps)
            return np.clip(X, -clip_sigma, clip_sigma)
        if scaler == "std":
            return X / ((std**2).mean(axis=-2, keepdims=True) ** 0.5 + eps)
    raise ValueError(f"Unknown scaler: {scaler!r}")


def pos_encode_continuous_batched(
    x, x_min, x_max, n_dim, out: torch.Tensor | None = None
) -> torch.Tensor:
    """1-dimensional positional encoding (batched).

    Args:
        x: torch.Tensor of shape (*xdims).
        x_min: minimum possible value of x.
        x_max: maximum possible value of x.
        n_dim: number of dimensions of the positional encoding (must be even).
        out: output tensor of shape (*xdims, n_dim). If None, allocated.
    """
    out = torch.empty(x.shape + (n_dim,), dtype=torch.float32) if out is None else out
    if not torch.compiler.is_compiling():
        assert out.shape == x.shape + (n_dim,)
        assert n_dim % 2 == 0
    div_term = torch.exp(
        (1 - torch.arange(0, n_dim, 2, device=out.device) / n_dim) * 2 * math.pi
    )
    xx = torch.as_tensor((x - x_min) / (x_max - x_min)).unsqueeze(-1)
    out[..., 0::2] = torch.sin(xx * div_term)
    out[..., 1::2] = torch.cos(xx * div_term)
    return out


def pos_encode_time(n_times, n_dim, max_n_times, out: torch.Tensor | None = None):
    """1-dimensional positional encoding for time samples.

    Args:
        n_times: number of time samples to encode.
        n_dim: number of dimensions of the positional encoding (must be even).
        max_n_times: largest possible number of time samples (used to scale).
        out: output tensor of shape (n_times, n_dim). If None, allocated.
    """
    out = torch.empty((n_times, n_dim), dtype=torch.float32) if out is None else out
    assert out.shape == (n_times, n_dim)
    assert n_dim % 2 == 0
    position = torch.arange(n_times, device=out.device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_dim, 2, device=out.device) * (-math.log(max_n_times) / n_dim)
    )
    out[:, 0::2] = torch.sin(position * div_term)
    out[:, 1::2] = torch.cos(position * div_term)
    return out


def get_distance_matrix(channels_positions: torch.Tensor):
    """Pairwise Euclidean distances between channel positions.

    Args:
        channels_positions: ``(n_channels, 3)``.

    Returns:
        ``(n_channels, n_channels)`` distance matrix.
    """
    return (
        (channels_positions.unsqueeze(0) - channels_positions.unsqueeze(1)) ** 2
    ).sum(dim=2) ** 0.5


def channels_block_masking(
    channels_positions: torch.Tensor,
    n_masks: int,
    radius: float,
    n_blocks: int,
    min_unmasked=None,
    return_centers: bool = False,
    padding_mask: torch.Tensor | None = None,
    out=None,
):
    """Mask all channels within ``radius`` of randomly chosen centers.

    ``True`` where masked (i.e. should be ignored).

    Two special values of ``radius`` are supported:
      - ``0.0``: only the center channel itself is masked.
      - ``math.inf``: all non-padding channels are masked.

    Returns:
        output: ``(n_masks, n_channels)``.
    """
    n_channels, _ = channels_positions.size()
    if n_blocks == 0:  # nothing masked
        return channels_positions.new_zeros((n_masks, n_channels), dtype=torch.bool)
    if padding_mask is not None:
        channels_positions = channels_positions.masked_fill(
            padding_mask.unsqueeze(-1), math.inf
        )
    center_ranks = channels_positions.new_empty(
        (n_masks, n_channels), dtype=torch.float32
    ).uniform_()
    if padding_mask is not None:
        center_ranks = center_ranks.masked_fill(padding_mask.unsqueeze(0), math.inf)
    centers = center_ranks.argsort(dim=1)[:, :n_blocks]  # (n_masks, n_blocks)
    dist_mat = get_distance_matrix(channels_positions)  # (n_channels, n_channels)
    # ``radius == 0.0`` is special-cased: ``dist < 0.0`` would mask nothing,
    # but we want the center channel (distance 0 to self) masked.
    # ``radius == math.inf`` works with the generic ``<`` because padding
    # positions are set to inf and ``inf < inf`` is False.
    if radius == 0.0:
        mask_mat = dist_mat <= 0.0
    else:
        mask_mat = dist_mat < radius
    mask = (
        out
        if out is not None
        else channels_positions.new_empty((n_masks, n_channels), dtype=torch.bool)
    )

    mask[:] = mask_mat[centers].any(dim=1)
    if min_unmasked is not None:
        unmasked = dist_mat[centers].min(dim=1).values.argsort(dim=1)[:, -min_unmasked:]
        selected = torch.zeros_like(mask, dtype=torch.bool)
        selected.scatter_(1, unmasked, True)
        mask[selected] = False
    if return_centers:
        return mask, centers
    return mask


def to_fp32(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: v.float() for k, v in state_dict.items()}


def update_ema_params(
    model: torch.nn.Module, new_model: torch.nn.Module, decay: float, fp32: bool = True
):
    """In-place EMA update of ``model`` parameters towards ``new_model``.

    Inspired by
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/ema_module.py
    """
    ema_state_dict = {}
    ema_params = model.state_dict()
    if fp32:
        ema_params = to_fp32(ema_params)

    for key, param in new_model.named_parameters():
        if isinstance(param, dict):
            continue
        if key not in ema_params:
            continue

        ema_param = ema_params[key]

        if param.shape != ema_param.shape:
            raise ValueError(
                f"incompatible tensor shapes between model param and ema param"
                f"{param.shape} vs. {ema_param.shape}"
            )

        if "version" in key:
            # Do not decay a model.version pytorch param
            continue

        lr = 1 - decay
        if not param.requires_grad:
            ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
            ema_param = ema_params[key]
        else:
            ema_param.mul_(1 - lr)
            ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=lr)

        ema_state_dict[key] = ema_param

    for key, param in new_model.named_buffers():
        ema_state_dict[key] = param

    model.load_state_dict(ema_state_dict)
