import itertools
from copy import deepcopy

from torch import nn
import torch
import einops
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm

from eeg_fm_masking.modules import VarLoss, CovLoss, MaskedMSELoss
from eeg_fm_masking.models import ContextualEncoder
from eeg_fm_masking.configs.pl_ssl import SSLFrameworkConfig
from eeg_fm_masking.functions import update_ema_params


parse_shape = einops.parse_shape


class SSLFramework(LightningModule):
    """JEPA (Joint-Embedding Predictive Architecture) framework.

    Uses an EMA teacher to produce prediction targets in latent space.
    The student encoder processes masked input; the predictor reconstructs
    the teacher's representations at masked positions.
    """

    def __init__(
        self,
        cfg: SSLFrameworkConfig,
        feature_encoder: nn.Module | None,
        model: ContextualEncoder,
        predictor: nn.Module,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = feature_encoder
        self.model = model
        self.predictor = predictor
        self.loss_fn = MaskedMSELoss()
        self.var_loss_fn = VarLoss(target=self.cfg.var_loss_target)
        self.cov_loss_fn = CovLoss()
        self.ema_model = deepcopy(ContextualEncoder.to_maskless_arch(self.model))
        self.ema_model.eval()

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
        # Actual embedding std for collapse monitoring (var_loss is a hinge
        # loss that equals 0 when variance is healthy, so it cannot detect
        # collapse on its own)
        flat = einops.rearrange(embedding, "... d -> (...) d")
        embedding_std = flat.std(dim=0).mean()
        return {
            "loss": loss,
            "main_loss": main_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "embedding_std": embedding_std,
        }

    def _get_ema_decay(self):
        if self.global_step >= self.cfg.ema_anneal_end_step:
            return self.cfg.ema_end_decay
        r = self.cfg.ema_end_decay - self.cfg.ema_decay
        pct_remaining = 1 - self.global_step / self.cfg.ema_anneal_end_step
        return self.cfg.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def _step_teacher(self):
        decay = self._get_ema_decay()
        self.log("trainer/ema_decay", decay, on_step=True, on_epoch=False)
        update_ema_params(
            model=self.ema_model,
            new_model=self.model,
            decay=decay,
            fp32=True,
        )

    @torch.no_grad()
    def _forward_teacher(self, batch):
        self.ema_model.eval()
        out = self.ema_model(
            batch, return_last_n_outputs=self.cfg.average_top_k_outputs
        )
        z_list = out["ctx_features"]
        if not isinstance(z_list, list):
            z_list = [z_list]
        targets = z_list[0]
        for z in z_list[1:]:
            targets.add_(z.float())
        targets = targets.div_(len(z_list))
        return targets

    def forward(
        self, batch, apply_mask=True
    ) -> dict:  # pylint: disable=arguments-differ
        X = batch["X"]

        # Same feature encoder is used for both teacher and student.
        if self.feature_encoder is not None:
            local_features = self.feature_encoder(X)
            batch["local_features"] = local_features

        return self.model(batch, apply_mask=apply_mask)

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer):
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
        )
        out["preds"] = preds

        targets = self._forward_teacher(batch).detach()
        targets.masked_fill_(~targets_mask.unsqueeze(-1), 0.0)
        out["targets"] = targets

        losses = self._get_loss(out)
        return losses

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        losses = self._step_train(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in losses.items()},
            batch_size=len(batch["X"]),
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self._step_teacher()
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.model.parameters(),
                self.predictor.parameters(),
                (
                    self.feature_encoder.parameters()
                    if self.feature_encoder is not None
                    else []
                ),
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
