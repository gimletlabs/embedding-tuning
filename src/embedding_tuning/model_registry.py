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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T

MODEL_NAME_TO_DATASET = {
    # OpenCLIP models
    "ViT-B-32": "laion2b_s34b_b79k",
    "EVA02-L-14-336": "merged2b_s6b_b61k",
    "ViT-H-14-378-quickgelu": "dfn5b",
    "ViT-B-16-SigLIP-384": "webli",
    "ViT-B-16": "datacomp_xl_s13b_b90k",
}


class OpenCLIPFeatureExtractor(nn.Module):
    """
    Wrapper for OpenCLIP image encoder. Used to extract image features.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone.encode_image(x)


def load_open_clip_model(model_name, model_weights):
    # Model weights can represent a path, or a label to a pretrained model.
    # Figure out which one we are dealing with here.
    ckpt_path = None
    pretrained = None
    if model_weights and Path(model_weights).exists():
        ckpt_path = model_weights
    else:
        pretrained = model_weights
    # Load the OpenCLIP model and preprocessing transforms
    open_clip_model, train_preprocess_fn, val_preprocess_fn = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )

    if ckpt_path is not None:
        print(f"Loading model weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        open_clip_model.load_state_dict(ckpt["state_dict"])
    return open_clip_model, train_preprocess_fn, val_preprocess_fn


@dataclass
class ModelConfig:
    hidden_dim: int
    train_preprocess_transform: T.Compose = T.Compose([])
    val_preprocess_transform: T.Compose = T.Compose([])


class ModelType(Enum):
    """Model Type enum."""

    OPEN_CLIP = ModelConfig(
        hidden_dim=512,
    )

    def load_model(self, model_weights: str, model_name: str):
        """Load model and corresponding model config."""

        if self.name == "OPEN_CLIP":
            model, train_preprocess_fn, val_preprocess_fn = load_open_clip_model(
                model_name,
                model_weights,
            )
            model = OpenCLIPFeatureExtractor(model)
            self.value.train_preprocess_transform = cast(T.Compose, train_preprocess_fn)
            self.value.val_preprocess_transform = cast(T.Compose, val_preprocess_fn)

        else:
            raise ValueError(f"Unknown model type: {self.name}")
        return model, self.value
