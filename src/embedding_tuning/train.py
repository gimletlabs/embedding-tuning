# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import lightning as L
import torch
from absl import app, flags
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from embedding_tuning.datasets.cropped_coco import CroppedCOCODataset
from embedding_tuning.model_registry import (
    MODEL_NAME_TO_DATASET,
    ModelType,
)
from embedding_tuning.models.angular_margin import (
    EmbeddingModelProjection,
)

flags.DEFINE_enum(
    "model_type",
    ModelType.OPEN_CLIP.name,
    [itm.name for itm in ModelType],
    "Features extractor model type to use for angular margin finetuning.",
)
flags.DEFINE_string("model_name", "ViT-B-16", "Model to use")
flags.DEFINE_string("output_dir", "output", "Output directory")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("lr", 1e-4, "Learning rate")
flags.DEFINE_integer("num_epochs", 55, "Number of epochs")
flags.DEFINE_float("margin", 0.5, "margin angle in radians, used in angular margin loss")
flags.DEFINE_float(
    "scale",
    64.0,
    "Scale parameter for angular margin loss. "
    "Controls the radius of the hypersphere where embeddings are projected.",
)
flags.DEFINE_bool("easy_margin", False, "Whether to use easy margin")

flags.DEFINE_string(
    "train_ann",
    None,
    "Path to the training annotation file",
    required=True,
)

flags.DEFINE_string(
    "validation_ann",
    None,
    "Path to the validation annotation file",
    required=True,
)

flags.DEFINE_string(
    "experiment_dir",
    "",
    "Directory to store experiment logs and checkpoints",
)

FLAGS = flags.FLAGS


class ExponentialLRWithWarmup(LRScheduler):
    """
    Learning Rate that first linearly warmups up to a desired learning rate, then
    exponentially decays.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        warmup_start_value: float,
        warmup_duration: int,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            gamma (float): Multiplicative factor of learning rate decay.
            warmup_start_value (float): Initial learning rate for warmup.
            warmup_duration (int): Number of steps for warmup.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.gamma = gamma
        self.warmup_start_value = warmup_start_value
        self.warmup_duration = warmup_duration
        self.warmup_step = 0
        self.warmup_end_value = optimizer.param_groups[0]["lr"]

        # Initialize base class
        super().__init__(optimizer, last_epoch)

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = warmup_start_value

    def get_lr(self):
        """Compute learning rate using chainable form of the scheduler."""
        if self.warmup_step < self.warmup_duration:
            # Linear warmup
            lr = (
                self.warmup_end_value - self.warmup_start_value
            ) * self.warmup_step / self.warmup_duration + self.warmup_start_value
            return [lr for _ in self.base_lrs]
        else:
            # Exponential decay
            return [
                base_lr * (self.gamma ** (self.last_epoch - self.warmup_duration))
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """Step the scheduler forward."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.warmup_step < self.warmup_duration:
            self.warmup_step += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LitAngularMarginModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
    ):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images, labels)
        loss = nn.functional.cross_entropy(logits, labels)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images, labels)
        loss = nn.functional.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=FLAGS.lr, momentum=0.9)
        scheduler = ExponentialLRWithWarmup(
            optimizer, gamma=0.98, warmup_start_value=0.0, warmup_duration=100
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FeatureExtractorCheckpoint(ModelCheckpoint):
    """Custom checkpoint callback that saves only feature extractor parameters."""

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        """Override the save checkpoint method to save only feature extractor parameters."""
        # Get FeatureExtractor model parameters
        model_params = {"state_dict": trainer.model.model.model.backbone.state_dict()}
        torch.save(model_params, filepath)


def main(_):
    for k, v in FLAGS._flags().items():
        print(f"{k}: {v.value}")

    model, model_config = ModelType[FLAGS.model_type].load_model(
        MODEL_NAME_TO_DATASET[FLAGS.model_name],
        FLAGS.model_name,
    )

    dataset = CroppedCOCODataset(
        FLAGS.train_ann,
        crop_transform=model_config.train_preprocess_transform,
    )

    val_dataset = CroppedCOCODataset(
        FLAGS.validation_ann,
        crop_transform=model_config.val_preprocess_transform,
    )

    model = EmbeddingModelProjection(
        model=model,
        n_hidden=model_config.hidden_dim,
        n_out=len(dataset.cat_ids),
        margin=FLAGS.margin,
        scale=FLAGS.scale,
        easy_margin=FLAGS.easy_margin,
    )
    lit_model = LitAngularMarginModel(model)

    # Use both checkpoint callbacks
    default_checkpoint = ModelCheckpoint(
        dirpath=FLAGS.experiment_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="best-checkpoint",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    feature_extractor_checkpoint = FeatureExtractorCheckpoint(
        dirpath=FLAGS.experiment_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="best-feature-extractor",
    )
    logger = TensorBoardLogger(
        save_dir=FLAGS.experiment_dir,
    )
    trainer = L.Trainer(
        max_epochs=FLAGS.num_epochs,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        callbacks=[default_checkpoint, feature_extractor_checkpoint, lr_monitor],
        logger=logger,
    )
    trainer.validate(
        model=lit_model,
        dataloaders=DataLoader(val_dataset, batch_size=FLAGS.batch_size),
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=32
        ),
        val_dataloaders=DataLoader(val_dataset, batch_size=FLAGS.batch_size),
    )


if __name__ == "__main__":
    app.run(main)
