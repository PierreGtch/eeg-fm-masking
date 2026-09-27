"""Lightning callback that detects JEPA representation collapse and stops training.

Monitors embedding variance and main_loss for signs of collapse:
- Embedding std dropping near zero (variance collapse)
- main_loss exploding (divergence post-collapse)
- main_loss suspiciously low early in training (trivial solution)

Note on var_loss vs actual variance:
  var_loss = ReLU(target - sqrt(variance)) is a hinge loss that equals 0
  when variance >= target (healthy). It is NOT a good collapse indicator
  because it stays at 0 for all healthy states. Instead, we monitor
  train_embedding_std which is the actual standard deviation of the
  embeddings across the batch.
"""
import logging

from lightning.pytorch import Callback

logger = logging.getLogger(__name__)


class CollapseDetectorCallback(Callback):
    """Stops training when JEPA representation collapse is detected.

    Three detection conditions (any one triggers a stop):
        1. embedding_std < std_epsilon for std_patience consecutive steps.
        2. main_loss > main_loss_max.
        3. main_loss < main_loss_min_early and global_step < early_steps.

    The embedding std is logged by the SSL framework as ``train_embedding_std``.

    Parameters
    ----------
    std_epsilon : float
        Threshold below which embedding std is considered collapsed.
    std_patience : int
        Number of consecutive steps with std < std_epsilon before stopping.
    main_loss_max : float
        Upper threshold for main_loss explosion detection.
    main_loss_min_early : float
        Lower threshold for suspiciously low main_loss early in training.
    early_steps : int
        Number of initial steps during which main_loss_min_early is checked.
    """

    def __init__(
        self,
        std_epsilon: float = 0.01,
        std_patience: int = 200,
        main_loss_max: float = 1e4,
        main_loss_min_early: float = 0.001,
        early_steps: int = 500,
    ):
        super().__init__()
        self.std_epsilon = std_epsilon
        self.std_patience = std_patience
        self.main_loss_max = main_loss_max
        self.main_loss_min_early = main_loss_min_early
        self.early_steps = early_steps
        self._consecutive_low_std = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        embedding_std = metrics.get("train_embedding_std")
        main_loss = metrics.get("train_main_loss")
        step = trainer.global_step

        # Check embedding std collapse
        if embedding_std is not None:
            val = float(embedding_std)
            if val < self.std_epsilon:
                self._consecutive_low_std += 1
                logger.warning(
                    "Step %d: embedding_std=%.6f < epsilon=%.4f "
                    "(%d/%d consecutive)",
                    step, val, self.std_epsilon,
                    self._consecutive_low_std, self.std_patience,
                )
                if self._consecutive_low_std >= self.std_patience:
                    logger.warning(
                        "COLLAPSE DETECTED: embedding_std has been below %.4f "
                        "for %d consecutive steps. Stopping training.",
                        self.std_epsilon, self.std_patience,
                    )
                    trainer.should_stop = True
                    return
            else:
                self._consecutive_low_std = 0

        # Check main_loss explosion
        if main_loss is not None:
            val = float(main_loss)
            if val > self.main_loss_max:
                logger.warning(
                    "COLLAPSE DETECTED: main_loss=%.2f > threshold=%.2f "
                    "at step %d. Stopping training.",
                    val, self.main_loss_max, step,
                )
                trainer.should_stop = True
                return

        # Check suspiciously low main_loss early in training
        if main_loss is not None and step < self.early_steps:
            val = float(main_loss)
            if val < self.main_loss_min_early:
                logger.warning(
                    "COLLAPSE DETECTED: main_loss=%.6f < %.4f at step %d "
                    "(< %d). Trivial solution likely. Stopping training.",
                    val, self.main_loss_min_early, step, self.early_steps,
                )
                trainer.should_stop = True
                return
