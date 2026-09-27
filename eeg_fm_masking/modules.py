import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
import einops

from eeg_fm_masking.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from eeg_fm_masking.functions import (
    pos_encode_continuous_batched,
    pos_encode_time,
    channels_block_masking,
)


class LinearPatchEmbedding(nn.Module):
    """Linear patch embedding for EEG, as used in REVE.

    Segments each channel into overlapping patches along the time dimension and
    projects each patch linearly to an embedding dimension.

    Args:
        embed_dim: Output embedding dimension.
        patch_size: Size of each patch in samples (default 200 = 1 s at 200 Hz).
        patch_overlap: Overlap between consecutive patches in samples (default 20).
    """

    def __init__(
        self, *, embed_dim: int, patch_size: int = 200, patch_overlap: int = 20
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.linear = nn.Linear(patch_size, embed_dim)

    def forward(
        self, x: torch.Tensor, return_patches: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(b, c, t)`` raw EEG signal.
        Returns:
            ``(b, c, n_patches, embed_dim)`` patch embeddings.
        """
        # (b, c, t) -> (b, c, n_patches, patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size - self.patch_overlap)
        output = self.linear(patches)
        if return_patches:
            return output, patches
        return output


class EEGTransformerDecoder(nn.Module):
    """Transformer decoder for MAE reconstruction.

    Uses cross-attention to attend to the encoder output (memory) and
    reconstructs raw patches at masked positions via a final linear projection.

    The decoder input (tgt) is ``mask_token`` (+ positional encoding) at every
    position.  A ``tgt_key_padding_mask`` restricts self-attention to masked
    positions only, while a ``memory_key_padding_mask`` restricts cross-attention
    to unmasked encoder positions.  This avoids feeding the encoder output into
    both tgt and memory, while keeping tensor shapes constant for
    ``torch.compile``.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout probability.
    bias : bool
        Whether linear layers use bias.
    num_layers : int
        Number of decoder layers.
    patch_size : int
        Output dimension per token (raw patch length).
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        bias: bool = False,
        activation: Callable = F.relu,
        glu: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        num_layers: int,
        patch_size: int,
        memory_dim: int | None = None,
    ):
        super().__init__()
        if memory_dim is not None and memory_dim != d_model:
            self.memory_proj = nn.Linear(memory_dim, d_model, bias=bias)
        else:
            self.memory_proj = None
        layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
            activation=activation,
            glu=glu,
            norm_layer=norm_layer,
        )
        self.transformer = TransformerDecoder(
            layer,
            num_layers=num_layers,
        )
        self.mask_token = nn.Parameter(torch.empty(d_model))
        self.linear = nn.Linear(d_model, patch_size, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.mask_token)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.linear.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        pos_encoding: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        memory : torch.Tensor
            Encoder output, shape ``(b, c, t, d)``.
        pos_encoding : torch.Tensor
            Positional encoding, shape ``(b, c, t, d)``.
        mask : torch.Tensor
            Boolean mask, shape ``(b, c, t)``. ``True`` = masked (to reconstruct).

        Returns
        -------
        torch.Tensor
            Predictions, shape ``(b, c, t, patch_size)``.
        """
        if self.memory_proj is not None:
            memory = self.memory_proj(memory)
            pos_encoding = self.memory_proj(pos_encoding)

        tgt = self.mask_token + pos_encoding  # (b, c, t, d)

        tgt_key_padding_mask = einops.rearrange(~mask, "b c t -> b (c t)")
        memory_key_padding_mask = einops.rearrange(mask, "b c t -> b (c t)")

        decoded = self.transformer(
            tgt, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.linear(decoded)


class PositionalEncoder(nn.Module):
    """
    Positional encoder for EEG data.

    Args:
        spat_dim: int | None
            Number of dimensions to use to encode the spatial position of the patch,
            i.e. the EEG channel. If None, the spatial position is not encoded.
        time_dim: int | None
            Number of dimensions to use to encode the temporal position of the patch.
            If None, the temporal position is not encoded.
        sfreq_features: float
            The "downsampled" sampling frequency returned by the feature encoder.
        max_seconds: float
            Maximum number of seconds to encode. The default is 600s = 10 minutes.
        max_x: float
            Maximum value of the spatial position. Should be in the same unit as the
            channel locations.
        n_spat_coord: int
            Number of spatial coordinates. Typically 3 for x, y, z coordinates.
        init_buffer_n_times: int | None
            If not None, the buffer for the temporal positional encoding will be
            initialised to encode at least this number of time steps.
    """

    def __init__(
        self,
        spat_dim: int | None = None,
        time_dim: int | None = None,
        sfreq_features: float | None = None,
        max_seconds: float = 600,
        max_x: float = 2,
        n_spat_coord: int = 3,
        init_buffer_n_times: int | None = None,
    ):
        super().__init__()
        self.spat_dim = spat_dim
        self.time_dim = time_dim
        self.max_seconds = max_seconds
        self.max_x = max_x
        self.n_spat_coord = n_spat_coord
        if spat_dim:
            assert (
                spat_dim % n_spat_coord == 0
            ), "spat_dim must be a multiple of n_spat_coord"
        if self.time_dim:
            assert sfreq_features is not None, "sfreq_features must be set"
            self.max_n_times = int(self.max_seconds * sfreq_features)
            self.register_buffer(
                "encoding_time",
                torch.zeros((0, self.time_dim), dtype=torch.float32),
            )
            if init_buffer_n_times is not None:
                self._check_encoding_time(init_buffer_n_times)
        self.fixed_ch_names = None

    def _check_encoding_time(self, n_times):
        assert (
            self.time_dim is not None
        ), "time_dim must be set to call _check_encoding_time"
        if self.encoding_time.size(0) < n_times:
            self.encoding_time = self.encoding_time.new_empty((n_times, self.time_dim))
            _ = pos_encode_time(
                n_times=n_times,
                n_dim=self.time_dim,
                max_n_times=self.max_n_times,
                out=self.encoding_time,
            )

    def _check_input(self, local_features):
        b, c, t, d = local_features.shape
        shapes = {"b": b, "c": c, "t": t, "d": d}
        assert shapes["d"] >= (self.spat_dim or 0) + (self.time_dim or 0)
        return shapes

    def forward(self, local_features, ch_pos, split=False):
        """
        Args:
            local_features
            ch_pos
            split: bool
                If True, the output positional encoding will be split into two tensors

        Returns:
            (pos_encoding_spat, pos_encoding_time)
                If split=True.
            pos_encoding
                If split=False. ``pos_encoding = pos_encoding_spat + pos_encoding_time``
        """
        shapes = self._check_input(local_features)
        pos_encoding_spat = torch.zeros_like(local_features[..., :1, :])  # t=1
        pos_encoding_time = torch.zeros(
            (shapes["b"], 1, shapes["t"], shapes["d"]),
            device=local_features.device,
            dtype=local_features.dtype,
        )

        if self.spat_dim:
            out_spat = ch_pos.new_empty(
                ch_pos.shape + (self.spat_dim // self.n_spat_coord,)
            )
            _ = pos_encode_continuous_batched(
                ch_pos,
                x_min=-self.max_x,
                x_max=self.max_x,
                n_dim=self.spat_dim // self.n_spat_coord,
                out=out_spat,
            )
            pos_encoding_spat[:, :, :, : self.spat_dim] = einops.rearrange(
                out_spat, "b c nsc d -> b c 1 (nsc d)"
            )

        if self.time_dim:
            self._check_encoding_time(shapes["t"])
            start_dim = self.spat_dim or 0
            _ = pos_encoding_time[:, :, :, start_dim : start_dim + self.time_dim].copy_(
                self.encoding_time[None, None, : shapes["t"], :],
            )
        if split:
            return pos_encoding_spat, pos_encoding_time
        return pos_encoding_spat + pos_encoding_time


class MaskMaker(nn.Module):
    """Same mask maker as in V-JEPA.

    Only difference: instead of rectangular targets over pixels,
    we define a sphere and mask all channels within the sphere.

    Parameters
    ----------
    radius_blocks: float
        Spatial radius of the target blocks to mask (same unit as the channels positions).
        For a mask covering 100% of the head, with ``head_size=1``,
        use ``radius_blocks=1``.

        Two special values are supported and bypass the spherical-cap formula:

        - ``0.0``: only the center channel of each block is masked
          (single-channel block).
        - ``math.inf``: all (non-padding) channels are masked
          (full-spatial block).

        For both special values ``scalp_surface`` is ignored when computing
        ``n_target_blocks`` from ``pct_unmasked``.
    length_blocks: int
        Temporal length of the target blocks to mask.
    n_target_blocks: int | None
        Number of target to generate
    pct_unmasked: float | None
        minimum percentage of non-masked elements.
        Used in combination with scalp_surface as alternative way of computing n_target_blocks.
    scalp_surface: float | None
        The surface of the scalp covered by electrodes.
        Used in combination with pct_unmasked as alternative way of computing n_target_blocks.
        Rough formula: scalp_surface = 4 * pi * head_radius**2 * 3/4,
        where head_radius is in the same unit as the channels positions and radius_blocks.
        Ignored when ``radius_blocks`` is one of the special values
        (``0.0`` or ``math.inf``); pass any positive number in that case.

    Notes
    -----
    When ``ch_padding_mask`` is provided to ``forward``, the effective
    number of channels can vary across batch elements. In that case the
    mask-density formula is applied per element (``n_target_blocks``
    becomes a 1-D tensor of shape ``(batch_size,)`` internally) so the
    target mask ratio is honoured per element. See
    ``get_n_target_blocks`` for details.
    """

    def __init__(
        self,
        radius_blocks: float,
        length_blocks: int,
        n_target_blocks: int | None = None,
        pct_unmasked: float | None = None,
        scalp_surface: float | None = None,
    ):
        assert (pct_unmasked is None) == (
            scalp_surface is None
        ), "pct_unmasked and scalp_surface must be both set or both None"
        assert (n_target_blocks is None) != (
            pct_unmasked is None
        ), "Can not specify both n_target_blocks and pct_unmasked+scalp_surface"
        super().__init__()
        self.radius_blocks = radius_blocks
        self.length_blocks = length_blocks
        self._n_target_blocks = n_target_blocks
        self.pct_unmasked = pct_unmasked
        self.scalp_surface = scalp_surface
        if self.scalp_surface is not None and not self._radius_is_special:
            assert self.covered_surface < self.scalp_surface, (
                "The mask can not cover more than 100% of the scalp. "
                "Decrease radius_blocks or increase scalp_surface. Possible unit mismatch"
            )

    @property
    def _radius_is_special(self) -> bool:
        return self.radius_blocks == 0.0 or math.isinf(self.radius_blocks)

    @property
    def covered_surface(self):
        """Spherical cap area for Euclidean radius ``radius_blocks``.

        For two points at Euclidean distance ``r`` on a sphere of radius
        ``R``, the spherical cap of points within distance ``r`` of one
        of them has height ``h = r²/(2R)`` and area ``2πRh = πr²`` —
        independent of ``R`` in the small-cap regime. See
        https://en.wikipedia.org/wiki/Spherical_cap.
        """
        return math.pi * self.radius_blocks**2

    def get_n_target_blocks(
        self,
        n_times: int,
        n_channels: int | torch.Tensor | None = None,
    ) -> int | torch.Tensor:
        """Number of target blocks required to reach ``pct_unmasked``.

        ``n_channels`` may be:
          - ``None`` / ``int``: returns an ``int``.
          - 1-D tensor of shape ``(batch_size,)`` with the number of
            non-padding channels per batch element: returns a 1-D tensor
            of the same shape (one value per element). Useful when batch
            elements have a variable effective channel count due to
            ``ch_padding_mask``.

        ``n_channels`` is required only when ``radius_blocks == 0.0``
        (the single-channel-block formula needs it); for the other radii
        it merely controls the output shape.
        """
        if self._n_target_blocks is not None:
            return self._n_target_blocks
        if not torch.compiler.is_compiling():
            assert self.pct_unmasked is not None and self.scalp_surface is not None

        # Per-block token coverage fraction f.
        if self.radius_blocks == 0.0:
            assert n_channels is not None, (
                "n_channels is required when radius_blocks=0.0"
            )
            f = self.length_blocks / (n_channels * n_times)
        elif math.isinf(self.radius_blocks):
            f = self.length_blocks / n_times
        else:
            f = (
                self.covered_surface * self.length_blocks / self.scalp_surface / n_times
            )

        # Inclusion-exclusion correction: block centers are sampled with
        # replacement, so blocks overlap. Under the uniform-iid-centers
        # approximation, E[mask_ratio] = 1 - (1-f)^K, giving
        # K = ceil(log1p(-rho) / log1p(-f)). The naive no-overlap formula
        # K = ceil(rho/f) systematically under-masks.
        rho = 1.0 - self.pct_unmasked

        if isinstance(f, torch.Tensor):
            return (
                math.log1p(-rho) / torch.log1p(-f)
            ).ceil().long().clamp(min=1)

        if rho <= 0.0:
            n_target_blocks = 0
        elif f >= 1.0:
            n_target_blocks = 1
        else:
            n_target_blocks = max(
                1, math.ceil(math.log1p(-rho) / math.log1p(-f))
            )
        if isinstance(n_channels, torch.Tensor):
            return torch.full(
                n_channels.shape,
                n_target_blocks,
                dtype=torch.long,
                device=n_channels.device,
            )
        return n_target_blocks

    def _sample_block_masks(self, ch_pos, mask, padding_mask, n_times, n_target_blocks):
        channels_masks = channels_block_masking(
            channels_positions=ch_pos,
            n_masks=n_target_blocks,
            radius=self.radius_blocks,
            n_blocks=1,
            min_unmasked=None,
            padding_mask=padding_mask,
        )  # (n_masks, n_channels)

        high = max(1, n_times - self.length_blocks + 1)
        starts = torch.randint(
            low=0, high=high, size=(n_target_blocks,), device=mask.device
        )
        time_indices = (
            starts[:, None]
            + torch.arange(self.length_blocks, device=mask.device)[None, :]
        )  # (n_masks, L)
        time_masks = torch.zeros(
            n_target_blocks, n_times, dtype=torch.bool, device=mask.device
        )
        time_masks.scatter_(1, time_indices, True)  # (n_masks, n_times)

        mask |= (channels_masks.unsqueeze(2) & time_masks.unsqueeze(1)).any(dim=0)
        return mask

    def forward(
        self, ch_pos: torch.Tensor, ch_padding_mask: torch.Tensor | None, n_times: int
    ):
        # ``n_target_blocks`` varies per batch element only for radius=0.0
        # with padding (the other formulas don't depend on n_channels).
        if self.radius_blocks == 0.0 and ch_padding_mask is not None:
            n_channels_in: int | torch.Tensor = (~ch_padding_mask).sum(dim=1)
        else:
            n_channels_in = ch_pos.shape[1]
        n_target_blocks = self.get_n_target_blocks(n_times, n_channels=n_channels_in)
        cpm = ch_padding_mask
        masks = ch_pos.new_zeros(ch_pos.shape[:2] + (n_times,), dtype=torch.bool)
        if cpm is not None:
            masks |= cpm[:, :, None]
        for i, ch_pos_i in enumerate(ch_pos):
            cmp_i = cpm[i] if cpm is not None else None
            n_tb_i = (
                int(n_target_blocks[i].item())
                if isinstance(n_target_blocks, torch.Tensor)
                else n_target_blocks
            )
            masks[i] = self._sample_block_masks(
                ch_pos_i, masks[i], cmp_i, n_times, n_tb_i
            )
        return masks


class MaskMakerVectorized(MaskMaker):
    """Vectorized version of MaskMaker that operates on the full batch at once.

    Replaces the per-sample Python loop with batched tensor operations,
    making it compatible with torch.compile without excessive graph unrolling.

    Assumes n_blocks=1 (single center per target block), which is the only
    mode used in the original MaskMaker code.

    Supports the same ``radius_blocks`` special values as the parent
    ``MaskMaker`` (``0.0`` → only the center channel is masked,
    ``math.inf`` → all non-padding channels are masked).

    When ``ch_padding_mask`` is not None, the per-batch effective channel
    count varies across batch elements; ``n_target_blocks`` is then a
    1-D tensor of shape ``(batch_size,)`` and the output respects the
    per-element block count via a block-validity mask (excess blocks are
    suppressed). The allocation size is the worst-case ``n_target_blocks``
    at the full padded channel count, which keeps shapes static for
    ``torch.compile``.

    When ``return_blocks=True``, ``forward`` additionally returns the
    per-block ``centers`` (channel indices, ``(b, n_blocks_alloc)``) and
    ``starts`` (temporal offsets in token units, ``(b, n_blocks_alloc)``)
    used to generate the mask — useful for visualising which spatial cap
    / temporal window each block corresponds to.
    """

    def forward(
        self, ch_pos: torch.Tensor, ch_padding_mask: torch.Tensor | None, n_times: int,
        return_blocks: bool = False,
    ):
        b, c, _ = ch_pos.shape
        device = ch_pos.device

        if self.radius_blocks == 0.0 and ch_padding_mask is not None:
            n_channels_in: int | torch.Tensor = (~ch_padding_mask).sum(dim=1)
        else:
            n_channels_in = c
        n_target_blocks = self.get_n_target_blocks(n_times, n_channels=n_channels_in)
        n_target_blocks_alloc: int = self.get_n_target_blocks(n_times, n_channels=c)  # type: ignore[assignment]
        if isinstance(n_target_blocks, torch.Tensor):
            block_validity = (
                torch.arange(n_target_blocks_alloc, device=device).unsqueeze(0)
                < n_target_blocks.unsqueeze(1)
            )  # (b, n_target_blocks_alloc)
        else:
            block_validity = None

        positions = ch_pos
        if ch_padding_mask is not None:
            positions = positions.masked_fill(ch_padding_mask.unsqueeze(-1), math.inf)

        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist_mat = (diff * diff).sum(-1).sqrt()
        if self.radius_blocks == 0.0:
            mask_mat = dist_mat <= 0.0  # (b, c, c) — only self per row
        else:
            mask_mat = dist_mat < self.radius_blocks  # (b, c, c)

        center_ranks = ch_pos.new_empty(b, n_target_blocks_alloc, c).uniform_()
        if ch_padding_mask is not None:
            center_ranks = center_ranks.masked_fill(ch_padding_mask.unsqueeze(1), math.inf)
        centers = center_ranks.argsort(dim=-1)[..., 0]  # (b, n_target_blocks_alloc)

        b_idx = torch.arange(b, device=device).unsqueeze(1).expand_as(centers)
        channels_masks = mask_mat[b_idx, centers]  # (b, n_target_blocks_alloc, c)

        if block_validity is not None:
            channels_masks = channels_masks & block_validity.unsqueeze(-1)

        high = max(1, n_times - self.length_blocks + 1)
        starts = torch.randint(0, high, (b, n_target_blocks_alloc), device=device)
        offsets = torch.arange(self.length_blocks, device=device)
        time_indices = starts.unsqueeze(-1) + offsets  # (b, n_target_blocks_alloc, L)
        time_masks = torch.zeros(
            b, n_target_blocks_alloc, n_times, dtype=torch.bool, device=device
        )
        time_masks.scatter_(2, time_indices, True)

        # Combine spatial & temporal via matmul (avoids 4D intermediate):
        # (b, c, n_target) @ (b, n_target, n_times) → (b, c, n_times)
        masks = torch.bmm(
            channels_masks.permute(0, 2, 1).float(),
            time_masks.float(),
        ).bool()

        if ch_padding_mask is not None:
            masks = masks | ch_padding_mask.unsqueeze(-1)

        if return_blocks:
            return masks, centers, starts
        return masks


class VarLoss(nn.Module):
    """Variance term of the loss from VICReg.

    The features dimension, along which to compute the average,
    should be the last one. Variance is computed along all
    other dimensions.

    Parameters
    ----------
    target: float
        gamma parameter for the hinge loss in the paper
    eps: float
        Small value for numerical stability.
    """

    def __init__(self, target: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.target = target
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x = einops.rearrange(x, "... d -> (...) d")
        x = torch.var(x, dim=0)
        x = F.relu(self.target - torch.sqrt(x + self.eps))
        return torch.mean(x)


def traceable_cov(x):
    # Standardize: Subtract the mean
    mu = x.mean(dim=1, keepdim=True)
    x = x - mu
    # Covariance formula: (X * X.T) / (N - 1)
    return (x @ x.T) / (x.shape[1] - 1)


class CovLoss(nn.Module):
    """Covariance term of the loss from VICReg.

    The features dimension, along which to compute the covariance,
    should be the last one.
    """

    def forward(self, x: torch.Tensor):
        d = x.shape[-1]
        x = einops.rearrange(x, "... d -> d (...)")
        x = traceable_cov(x)
        x = torch.triu(x, diagonal=1)
        x = torch.sum(x**2) * 2 / d
        return x


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Mask is true where elements should be compared"""
        d = input.shape[-1]
        loss = self.mse(input, target)
        loss = loss.masked_fill(~mask.unsqueeze(-1), 0.0)
        return loss.sum() / mask.sum() / d
