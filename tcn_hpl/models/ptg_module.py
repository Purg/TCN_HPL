from typing import Any, Dict, Optional, Tuple, Union, List

from pytorch_lightning import LightningModule
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import F1Score, Recall, Precision

import einops


class PTGLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        smoothing_loss: float,
        use_smoothing_loss: bool,
        num_classes: int,
        compile: bool,
        pred_frame_index: int = -1,
    ) -> None:
        """Initialize a `PTGLitModule`.

        :param net: The model to train.
        :param criterion: Loss Computation
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param pred_frame_index:
            Index of a frame in the window whose predicted class and
            probabilities should represent the window as a whole. Negative
            indices are valid. Must be a valid index into the window range
            specified by the dataset
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss functions
        self.criterion = criterion
        self.mse = nn.MSELoss(reduction="none")

        # We only want to validation metric logging if training has actually
        # started, i.e. not during the sanity checking phase.
        self.has_training_started = False

        # Metric objects for calculating and averaging accuracy across batches
        # Various metrics are reset at the **beginning** of epochs due to the
        # desire to access metrics by callbacks and the **end** of epochs,
        # which would be hard to do if we reset at the end before they get a
        # chance to use those values...
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(
                    task="multiclass", num_classes=num_classes, average="weighted"
                ),
                "f1": F1Score(
                    task="multiclass", num_classes=num_classes, average="weighted"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="weighted"
                ),
                "precsion": Precision(
                    task="multiclass", num_classes=num_classes, average="weighted"
                ),
            },
            prefix="train/"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        # Some metrics that output per-class vectors. These will have to be
        # logged manually in a loop since only scalars can be logged.
        self.train_vec_metrics = MetricCollection(
            {
                "acc": Accuracy(
                    task="multiclass", num_classes=num_classes, average="none"
                ),
                "f1": F1Score(
                    task="multiclass", num_classes=num_classes, average="none"
                ),
            },
            prefix="train/"
        )
        self.val_vec_metrics = self.train_vec_metrics.clone(prefix="val/")
        self.test_vec_metrics = self.train_vec_metrics.clone(prefix="test/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_f1_best = MaxMetric()
        self.val_f1_best_epoch = MaxMetric()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of features for frames.
        :param m: A tensor of mask of the valid frames.
        :return: A tensor of logits.
        """
        return self.net(x, m)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.has_training_started = True

    def compute_loss(self, p, y, mask):
        """Compute the total loss for a batch

        :param p: The prediction
        :param y: The target labels
        :param mask: Marks valid input data

        :return: The loss
        """
        # p shape: (batch_size, num_classes, window_size)
        probs = torch.softmax(p, dim=1)  # shape (batch size, self.hparams.num_classes)
        preds = torch.argmax(probs, dim=1).float()  # shape: batch size

        loss = torch.zeros((1)).to(p[0])

        # Compute loss every frame in every batch.
        # The criterion is expected to softmax along the last dimension.
        loss += self.criterion(
            p.transpose(2, 1).contiguous().view(-1, self.hparams.num_classes),
            y.view(-1),
        )

        ## Only compute the criterion loss for the configured index of the
        ## window.
        #loss += self.criterion(
        #    p[:, :, self.hparams.pred_frame_index],
        #    y[:, self.hparams.pred_frame_index],
        #)

        if self.hparams.use_smoothing_loss:
            # need to penalize high volatility of predictions within a window
            mode, _ = torch.mode(y, dim=-1)
            mode = einops.repeat(mode, "b -> b c", c=preds.shape[-1])
            variation_coef = torch.abs(preds - mode)
            variation_coef = torch.sum(variation_coef, dim=-1)
            gt_variation_coef = torch.zeros_like(variation_coef)
            loss += self.hparams.smoothing_loss * torch.mean(
                self.mse(
                    variation_coef,
                    gt_variation_coef,
                ),
            )

        # Looks like this might be assuming the "reduction" model of the
        # configured criterion.
        # It looks like this is punishing consecutive predictions that are
        # different. A zero loss situation with this term would be when every
        # frame in the window is predicted to have the same class
        # probabilities. This doesn't seem relevant in the context of PTG where
        # some activities may be shorter than the window length, or the window
        # may cover the transition from one activity to another.
        #loss += self.hparams.smoothing_loss * torch.mean(
        #    torch.clamp(
        #        self.mse(
        #            F.log_softmax(p[:, :, 1:], dim=1),
        #            F.log_softmax(p.detach()[:, :, :-1], dim=1),
        #        ),
        #        min=0,
        #        max=16,
        #    )
        #    * mask[:, None, 1:]
        #)

        return loss

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of features, target labels, and mask.
        :param compute_loss: Flag to enable or not the computation of the loss
            value. If this is False, the value of the loss term is undefined.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of probabilities per class
            - A tensor of predictions.
            - A tensor of target labels.
            - A tensor of [video name, frame idx]
        """
        x, y, m, source_vid, source_frame = batch
        # x shape: (batch size, window, feat dim)
        # y shape: (batch size, window)
        # m shape: (batch size, window)
        # source_vid shape: (batch size, window)
        # source_frame shape: (batch size, window)
        x = x.transpose(2, 1)  # shape (batch size, feat dim, window)
        logits = self.forward(
            x, m
        )  # shape (4, batch size, num_classes, window))
        loss = torch.zeros((1)).to(x)
        if compute_loss:
            # Why compute loss for *every* stage when the prediction is only
            # pulled from the final stage? To do this may require or be
            # improved by adding residual connections between the stages.
            # ASSUMES RESIDUAL IN STAGES
            loss += self.compute_loss(logits[-1], y, m)
            #for p in logits:
            #    loss += self.compute_loss(p, y, m)

        pred_frame_index = self.hparams.pred_frame_index
        probs = torch.softmax(
            logits[-1, :, :, pred_frame_index], dim=1
        )  # shape (batch size, self.hparams.num_classes)
        preds = torch.argmax(logits[-1, :, :, pred_frame_index], dim=1)  # shape: batch size

        return loss, probs, preds, y, source_vid, source_frame

    def on_train_epoch_start(self) -> None:
        # Reset relevant metric collections
        self.train_metrics.reset()
        self.train_vec_metrics.reset()

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)

        # update and log loss
        # Don't want to log this on step because it causes some loggers (CSV)
        # to create some pretty unreadable output.
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, prog_bar=True, on_step=False, on_epoch=True
        )

        # return loss or backpropagation will fail
        return {
            "loss": loss,
            "preds": preds,
            "probs": probs,
            "targets": targets[:, self.hparams.pred_frame_index],
            "source_vid": source_vid[:, self.hparams.pred_frame_index],
            "source_frame": source_frame[:, self.hparams.pred_frame_index],
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        all_preds = torch.cat([o["preds"] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])

        scalar_metrics = self.train_metrics(all_preds, all_targets)
        self.log_dict(scalar_metrics, prog_bar=True, on_epoch=True)

        vec_metrics = self.train_vec_metrics(all_preds, all_targets)
        for k, t in vec_metrics.items():
            for i, v in enumerate(t):
                self.log(f"{k}-class_{i}", v)

    def on_validation_epoch_start(self) -> None:
        # Reset relevant metric collections
        self.val_metrics.reset()
        self.val_vec_metrics.reset()

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Optional[STEP_OUTPUT]:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, prog_bar=True)

        # Only retain the truth and source vid/frame IDs for the final window
        # frame as this is the ultimately relevant result.
        return {
            "loss": loss,
            "preds": preds,
            "probs": probs,
            "targets": targets[:, self.hparams.pred_frame_index],
            "source_vid": source_vid[:, self.hparams.pred_frame_index],
            "source_frame": source_frame[:, self.hparams.pred_frame_index],
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if not self.has_training_started:
            return

        all_preds = torch.cat([o['preds'] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])

        scalar_metrics = self.val_metrics(all_preds, all_targets)
        self.log_dict(scalar_metrics, prog_bar=True)

        vec_metrics = self.val_vec_metrics(all_preds, all_targets)
        for k, t in vec_metrics.items():
            for i, v in enumerate(t):
                self.log(f"{k}-class_{i}", v)

        # log `val_f1_best` as a value through `.compute()` return, instead of
        # as a metric object otherwise metric would be reset by lightning after
        # each epoch.
        # Also record the epoch from which the best F1 is from to the progress
        # bar, helping the reader know how far away from patience we are.
        if self.val_metrics.f1.compute() > self.val_f1_best.compute():
            self.val_f1_best_epoch(self.current_epoch)
        self.val_f1_best(self.val_metrics.f1.compute())
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True, on_epoch=True)
        self.log("val/f1_best_epoch", self.val_f1_best_epoch.compute(), prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        # Reset relevant metric collections
        self.test_metrics.reset()
        self.test_vec_metrics.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Optional[STEP_OUTPUT]:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, prog_bar=True)

        # Only retain the truth and source vid/frame IDs for the final window
        # frame as this is the ultimately relevant result.
        return {
            "loss": loss,
            "preds": preds,
            "probs": probs,
            "targets": targets[:, self.hparams.pred_frame_index],
            "source_vid": source_vid[:, self.hparams.pred_frame_index],
            "source_frame": source_frame[:, self.hparams.pred_frame_index],
        }

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        all_preds = torch.cat([o['preds'] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])

        scalar_metrics = self.test_metrics(all_preds, all_targets)
        self.log_dict(scalar_metrics, prog_bar=True, on_epoch=True)

        vec_metrics = self.test_vec_metrics(all_preds, all_targets)
        for k, t in vec_metrics.items():
            for i, v in enumerate(t):
                self.log(f"{k}-class_{i}", v)

    def setup(self, stage: Optional[str] = None) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PTGLitModule(None, None, None, None)
