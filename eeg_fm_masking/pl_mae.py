import itertools

from torch import nn
import torch
import einops
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm

from eeg_fm_masking.modules import VarLoss, CovLoss, MaskedMSELoss
from eeg_fm_masking.models import ContextualEncoder
from eeg_fm_masking.configs.pl_mae import MAEFrameworkConfig


parse_shape = einops.parse_shape


class MAEFramework(LightningModule):
    """Masked Autoencoder (MAE) framework.

    Unlike :class:`SSLFramework`, this framework uses the local patch features
    directly as reconstruction targets instead of relying on an EMA teacher.
    The contextual encoder encodes the masked input and the predictor reconstructs
    the local features at masked (hidden) positions.
    """

    def __init__(
        self,
        cfg: MAEFrameworkConfig,
        feature_encoder: nn.Module,  # LinearPatchEmbedding
        model: ContextualEncoder,
        predictor: nn.Module,
    ):
        super().__init__()
        assert (
            model.feature_encoder is None
        ), "ContextualEncoder should not have its own feature encoder"
        self.cfg = cfg
        self.feature_encoder = feature_encoder
        self.model = model
        self.predictor = predictor
        self.loss_fn = MaskedMSELoss()
        self.var_loss_fn = VarLoss(target=self.cfg.var_loss_target)
        self.cov_loss_fn = CovLoss()

    def _get_loss(self, out):
        preds = out["preds"]
        targets = out["targets"]
        embedding = out["ctx_features"]
        main_loss = self.loss_fn(preds, targets, out["targets_mask"])
        var_loss = self.var_loss_fn(embedding)
        cov_loss = self.cov_loss_fn(embedding)
        loss = (
            self.cfg.main_loss_weight * main_loss
            + self.cfg.var_loss_weight * var_loss
            + self.cfg.cov_loss_weight * cov_loss
        )
        return {
            "loss": loss,
            "main_loss": main_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }

    def forward(
        self, batch, apply_mask=True
    ) -> dict:  # pylint: disable=arguments-differ
        X = batch["X"]

        local_features, patches = self.feature_encoder(X, return_patches=True)
        batch["local_features"] = local_features

        out = self.model(batch, apply_mask=apply_mask)
        out["patches"] = patches
        return out

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer):
        # WARNING: grad_norm can be very slow; only enable for debugging.
        if (
            n := self.cfg.log_grad_norms_every_n_steps
        ) is not None and self.global_step % n == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def _step_train(self, batch, batch_idx):
        out = self(batch)

        # We pass targets_mask (= mask & ~padding_mask) rather than out["mask"]
        # so that padded positions are excluded from prediction.
        targets_mask = out["targets_mask"]
        preds = self.predictor(
            out["ctx_features"],
            pos_encoding=out["pos_encoding"],
            mask=targets_mask,
        )  # (b, c, t, patch_size)
        out["preds"] = preds

        targets = out["patches"].detach()  # (b, c, t, patch_size)
        # zero out non-masked elements so loss only covers masked positions
        targets = targets.masked_fill(~targets_mask.unsqueeze(-1), 0.0)
        out["targets"] = targets

        losses = self._get_loss(out)
        return losses

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        losses = self._step_train(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in losses.items()},
            batch_size=len(batch["X"]),
            sync_dist=True,  # For multiple GPU
            on_step=True,
            on_epoch=True,
        )

        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.model.parameters(),
                self.predictor.parameters(),
                self.feature_encoder.parameters(),
            ),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        total_steps = (
            self.trainer.estimated_stepping_batches
            if self.cfg.lr_scheduler_T_0 is None
            else self.cfg.lr_scheduler_T_0
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - self.cfg.warmup_steps),
            eta_min=self.cfg.final_lr,
        )
        if self.cfg.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, total_iters=self.cfg.warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.cfg.warmup_steps],
            )
        else:
            scheduler = cosine_scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
