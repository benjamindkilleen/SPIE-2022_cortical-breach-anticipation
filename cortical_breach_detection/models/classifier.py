#!/usr/bin/env python3
import logging
from typing import Any
from typing import Dict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch import optim
from torchmetrics.functional import accuracy
from torchmetrics.functional import auroc
from torchvision import models
from vertview.augmentations import build_augmentation

from ..metrics import AggregateMetrics


log = logging.getLogger(__name__)


class BinaryClassifier(pl.LightningModule):
    """

    input_channels: Number of channels in input images (default 3)
    num_layers: Number of layers in each side of U-net (default 5)
    features_start: Number of features in first layer (default 64)
    bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    lr: The learning rate.
    patience: Passed to the scheduler.
    augmentation: The config for augmentations to use on the image observations during training. (None corresponds to the identity transform.) Defaults to None.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        momentum: float = 0.9,
        scheduler: Dict[str, Any] = dict(patience=3, threshold=1e-6),
        augmentation: Optional[Dict] = None,
        encoder: str = "resnet50",
        normalize: bool = False,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.momentum = momentum
        self.scheduler = scheduler
        self.normalize = (
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if normalize else lambda x: x
        )
        self.augmentation = build_augmentation(augmentation)

        log.info(f"running with {encoder} model")
        self.resnet = getattr(models, encoder)(pretrained=False, num_classes=1)
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.augmentation = build_augmentation(augmentation)
        self.classification_metrics = AggregateMetrics(
            torchmetrics.AUROC(pos_label=1),
            torchmetrics.Specificity(),
            torchmetrics.Precision(),
            torchmetrics.Accuracy(),
        )

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        x = self.normalize(x)
        x = self.resnet(x)
        return dict(cortical_breach_logits=x)

    def training_step(self, batch, batch_idx):
        image, target, _ = batch
        mode = "train"
        image = self.augmentation(image)
        output = self(image)
        loss = self.bce_loss(output["cortical_breach_logits"], target["cortical_breach_label"].float())
        loss = torch.mean(loss)
        self.log(f"loss/{mode}", loss)
        self.log(
            f"{mode}/accuracy",
            accuracy(output["cortical_breach_logits"], target["cortical_breach_label"]),
            prog_bar=True,
        )
        return loss

    def eval_step(self, batch, batch_idx, mode="val"):
        image, target, info = batch
        labels = target["cortical_breach_label"]

        output = self(image)
        logits = output["cortical_breach_logits"]

        loss = self.bce_loss(logits, labels.float()).mean()
        self.log(f"loss/{mode}", loss)

        # log.debug(output["cortical_breach_logits"])
        pred = torch.sigmoid(output["cortical_breach_logits"])
        for m, v in self.classification_metrics(pred, labels, mode=mode).items():
            self.log(m, v)

        return dict(
            insertion_depth=info["insertion_depth"],
            pred=pred,
            label=target["cortical_breach_label"],
            progress=info["progress"],
        )

    def eval_epoch_end(self, outputs, mode="val"):
        pred = torch.cat([o["pred"] for o in outputs])
        label = torch.cat([o["label"] for o in outputs])
        progress_values = torch.cat([o["progress"] for o in outputs]).cpu().numpy()

        num_breached = torch.sum(label)
        log.info(f"{mode} set contains {num_breached} breaches out of {len(label)} samples")

        ps = []
        aurocs = []
        accs = []
        for p in np.unique(progress_values):
            ps.append(p)
            indices = progress_values == p
            if not np.any(indices):
                log.debug(f"progress {p} is empty")
                aurocs.append(0)
                continue

            try:
                acc = accuracy(pred[indices], label[indices]).cpu().numpy()
                a = auroc(pred[indices], label[indices]).cpu().numpy()
            except ValueError:
                log.error(f"auroc failed on progress {p}")
                acc = 0
                a = 0
            aurocs.append(a)
            accs.append(acc)

        acc = np.array(accs)
        aurocs = np.array(aurocs)
        ps = np.array(ps)
        log.info(f"progress values: {ps}")
        log.info(f"auroc: {aurocs}")
        log.info(f"accuracy: {accs}")
        np.savez(f"insertion_depth_metrics.npz", auroc=aurocs, progress_values=ps)
        plt.figure(figsize=(4, 3))
        plt.plot(ps, aurocs, "go")
        plt.savefig(f"{mode}_auroc_vs_insertion_progress_epoch={self.current_epoch}.pdf")

        self.classification_metrics.reset()

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.scheduler)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="loss/train")

    def configure_callbacks(self):
        return [
            LearningRateMonitor(logging_interval="step"),
            GPUStatsMonitor(),
            ModelCheckpoint(save_last=True, every_n_train_steps=1),
        ]
