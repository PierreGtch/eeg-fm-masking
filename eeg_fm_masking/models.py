from torch import nn
import einops


class ContextualEncoder(nn.Module):
    def __init__(self, *, feature_encoder=None, pos_encoder, transformer, masker=None):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.pos_encoder = pos_encoder
        self.transformer = transformer
        self.masker = masker

    def forward(
        self, batch, return_last_n_outputs: int = 1, apply_mask: bool = True
    ) -> dict:
        X = batch["X"]
        ch_pos = batch["ch_pos"]
        ch_padding_mask = batch.get("ch_padding_mask", None)
        out = {}
        masker = self.masker if apply_mask else None

        if self.feature_encoder is not None:
            local_features = self.feature_encoder(X)
        else:
            local_features = batch["local_features"]
        _, _, t, _ = local_features.shape
        out["local_features"] = local_features

        padding_mask = None
        if ch_padding_mask is not None:
            padding_mask = einops.repeat(ch_padding_mask, "b c -> b c t", t=t)
        out["padding_mask"] = padding_mask

        pos_encoding_spat, pos_encoding_time = self.pos_encoder(
            local_features=local_features, ch_pos=ch_pos, split=True
        )
        pos_encoding = pos_encoding_spat + pos_encoding_time
        assert pos_encoding.shape == local_features.shape
        out["pos_encoding"] = pos_encoding
        out["pos_encoding_spat"] = pos_encoding_spat
        out["pos_encoding_time"] = pos_encoding_time
        z = local_features + pos_encoding

        mask = None
        if masker is not None:
            mask = masker(
                ch_pos=ch_pos,
                ch_padding_mask=ch_padding_mask,
                n_times=t,
            )

        targets_mask = mask
        if targets_mask is not None and padding_mask is not None:
            targets_mask = mask & ~padding_mask
        out["mask"] = mask
        out["targets_mask"] = targets_mask

        ctx_features = self.transformer(
            z,
            mask=mask,
            return_last_n_outputs=return_last_n_outputs,
        )
        out["ctx_features"] = ctx_features
        return out

    @classmethod
    def to_maskless_arch(cls, obj: "ContextualEncoder") -> "ContextualEncoder":
        return ContextualEncoder(
            feature_encoder=obj.feature_encoder,
            pos_encoder=obj.pos_encoder,
            transformer=obj.transformer,
        )
