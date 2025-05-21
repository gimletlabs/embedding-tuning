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

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class AngularMarginProjection(nn.Module):
    """
    This module defines the angular margin projection layer.
    And follows ideas introduces in: https://arxiv.org/pdf/1801.07698

    The learned embeddings and class prototypes are normalized to lie on a hypersphere
    with radius R = \\sqrt{s}, where s is the scale parameter.

    For a candidate embedding c and prototype p (both L2-normalized),
    the cosine similarity is scaled as:
    \\sqrt{s} ⋅ p ⋅ c ⋅ \\sqrt{s} = s ⋅ cos(θ)
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        margin: float = 0.2,
        scale: float = 64,
        easy_margin: bool = False,
    ):
        """
        Args:
            n_in (int): input dimension
            n_out (int): output dimension (number of classes)
            margin (float): margin angle in radians
            scale (float): scaling factor, equals to sqrt(radius) of the hypersphere R = \\sqrt{s}.
            easy_margin (bool): if True, use easy margin
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # hard margin hyper-parameters
        self.th = math.cos(math.pi - margin)
        # scaling by margin allows smoother optimization
        self.mm = math.sin(math.pi - margin) * margin

        self.prototypes = nn.Parameter(torch.Tensor(n_in, n_out))
        torch.nn.init.xavier_normal_(self.prototypes)

    def forward(self, logits, labels):
        norm_logits = F.normalize(logits, dim=1, eps=1e-7)
        norm_prototypes = F.normalize(self.prototypes, dim=0, eps=1e-7)

        cos_theta = norm_logits @ norm_prototypes
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        # cos(theta + margin)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.n_out)
        output = torch.where(one_hot == 1, cos_theta_m, cos_theta)

        return output * self.scale


class EmbeddingModelProjection(nn.Module):
    """
    Wrapper for embedding model with projection layer. Used for AngularMargin training.
    """

    def __init__(self, model, n_hidden, n_out, margin=0.5, scale=64.0, easy_margin=False):  # noqa: PLR0913
        super().__init__()
        self.model = model
        self.projection = AngularMarginProjection(n_hidden, n_out, margin, scale, easy_margin)

    def forward(self, x, labels=None):
        logits = self.model(x)
        logits = self.projection(logits, labels)
        return logits
