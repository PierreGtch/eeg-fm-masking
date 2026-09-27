"""Transformer modules.

Adapted from
https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment
See also the tutorial
https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
"""

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection.
            Each head has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask=None,
        is_causal=False,
    ) -> Tensor:
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )
        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask, dropout_p=self.dropout, is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        return self.out_proj(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: Callable = torch.nn.functional.relu,
        glu: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, d_model, d_model, d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        # When glu=True, linear1 produces (gate, up) packed along the last dim,
        # so we double its output. Total params stay close to a standard FFN
        # when dim_feedforward is set to (8/3)*d_model (LLaMA convention).
        ff_in = dim_feedforward * 2 if glu else dim_feedforward
        self.linear1 = nn.Linear(d_model, ff_in, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.glu = glu
        self.d_model = d_model

    def _sa_block(self, x, attn_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x):
        if self.glu:
            gate, up = self.linear1(x).chunk(2, dim=-1)
            h = self.activation(gate) * up
        else:
            h = self.activation(self.linear1(x))
        x = self.linear2(self.dropout(h))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src_key_padding_mask is not None:
            # (b, S) True=ignore -> (b, 1, 1, S) True=attend for SDPA
            kp = ~src_key_padding_mask[:, None, None, :]
            src_mask = kp if src_mask is None else (src_mask & kp)
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: Callable = torch.nn.functional.relu,
        glu: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        norm_first=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, d_model, d_model, d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = MultiHeadAttention(
            d_model, d_model, d_model, d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )

        ff_in = dim_feedforward * 2 if glu else dim_feedforward
        self.linear1 = nn.Linear(d_model, ff_in, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.norm3 = norm_layer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.glu = glu

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        if self.glu:
            gate, up = self.linear1(x).chunk(2, dim=-1)
            h = self.activation(gate) * up
        else:
            h = self.activation(self.linear1(x))
        x = self.linear2(self.dropout(h))
        return self.dropout3(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        # Convert key_padding_masks (b, S) True=ignore -> SDPA (b, 1, 1, S) True=attend
        if tgt_key_padding_mask is not None:
            kp = ~tgt_key_padding_mask[:, None, None, :]
            tgt_mask = kp if tgt_mask is None else (tgt_mask & kp)
        if memory_key_padding_mask is not None:
            kp = ~memory_key_padding_mask[:, None, None, :]
            memory_mask = kp if memory_mask is None else (memory_mask & kp)

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_is_causal)
            x = x + self._mha_block(
                self.norm2(x), memory, memory_mask, memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_is_causal))
            x = self.norm2(
                x + self._mha_block(x, memory, memory_mask, memory_is_causal)
            )
            x = self.norm3(x + self._ff_block(x))
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        is_causal=False,
        return_last_n_outputs: int = 1,
        **kwargs,
    ):
        assert len(src.shape) == 4, str(src.shape)
        src_key_padding_mask = None
        if mask is not None:
            assert len(mask.shape) == 3, str(mask.shape)
            src = src.masked_fill(mask.unsqueeze(-1), 0.0)
            # mask: (b, c, t) True=masked -> key_padding_mask: (b, S) True=ignore
            src_key_padding_mask = einops.rearrange(mask, "b c t -> b (c t)")
        _, c, t, _ = src.shape
        src = einops.rearrange(src, "b c t d -> b (c t) d")
        output = src
        outputs = []
        for i, mod in enumerate(self.layers):
            output = mod(
                output, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal,
            )
            if (
                i >= self.num_layers - return_last_n_outputs
                or return_last_n_outputs == -1
            ):
                outputs.append(output)
        if len(outputs) == 0:
            outputs = [output]
        if self.norm is not None:
            outputs = [self.norm(o) for o in outputs]
        outputs = [
            einops.rearrange(o, "b (c t) d -> b c t d", c=c, t=t) for o in outputs
        ]
        if return_last_n_outputs == 1:
            return outputs[0]
        return outputs


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal=False,
        memory_is_causal=False,
        return_last_n_outputs: int = 1,
    ):
        assert len(tgt.shape) == 4, str(tgt.shape)
        assert len(memory.shape) == 4, str(memory.shape)
        _, c, t, _ = tgt.shape
        tgt = einops.rearrange(tgt, "b c t d -> b (c t) d")
        memory = einops.rearrange(memory, "b c t d -> b (c t) d")

        output = tgt
        outputs = []
        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            if (
                i >= self.num_layers - return_last_n_outputs
                or return_last_n_outputs == -1
            ):
                outputs.append(output)

        if len(outputs) == 0:
            outputs = [output]
        if self.norm is not None:
            outputs = [self.norm(o) for o in outputs]
        outputs = [
            einops.rearrange(o, "b (c t) d -> b c t d", c=c, t=t) for o in outputs
        ]
        if return_last_n_outputs == 1:
            return outputs[0]
        return outputs
