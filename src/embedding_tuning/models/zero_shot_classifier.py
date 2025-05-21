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

from itertools import groupby
from random import shuffle

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from embedding_tuning.datasets.tensor_dataset import TensorDataset
from embedding_tuning.model_registry import ModelType
from embedding_tuning.utils import load_json


def group_annotations(annotations, key_fn=lambda x: x["category_id"]) -> dict[str, list]:
    annotations = sorted(annotations, key=key_fn)
    category_groups = dict()
    for category_id, category_data in groupby(annotations, key=key_fn):
        category_groups[category_id] = list(category_data)
    return category_groups


class ZeroShotClassifier(nn.Module):
    def __init__(
        self,
        model_type,
        model_weight: str,
        model_name: str,
        k=5,
    ):
        super().__init__()

        self.k = k
        self.weight = None
        self.categories = None

        self.model, self.model_config = ModelType[model_type].load_model(
            model_weight,
            model_name,
        )

    def forward(self, x):
        x /= x.norm(dim=-1, keepdim=True)
        return x @ self.weight

    @property
    def device(self):
        return next(self.model.parameters()).device

    def build_zero_shot_classifier(self, catalog_path: str):
        catalog_data = load_json(catalog_path)

        annotations = catalog_data["annotations"]
        cats_dict = {itm["id"]: itm["name"] for itm in catalog_data["categories"]}
        image_paths = []
        image_ids = []
        img_id_to_emb_id = {}

        for idx, itm in enumerate(catalog_data["images"]):
            image_paths.append(itm["file_name"])
            image_ids.append(itm["id"])
            img_id_to_emb_id[itm["id"]] = idx

        embeddings = self.extract_embeddings(image_paths)

        image_category_groups = group_annotations(annotations)

        cat_names = []
        clsf_ids = []
        weights = []

        for cat_id, cat_name in cats_dict.items():
            cat_emb = []
            if cat_id not in image_category_groups:
                continue
            clsf_ids.append(cat_id)
            cat_names.append(cat_name)
            anno_group = image_category_groups[cat_id]
            # k randomly sampled images are used for average embedding prototype
            shuffle(anno_group)
            for anno in anno_group[: self.k]:
                img_id = anno["image_id"]
                emb_id = img_id_to_emb_id[img_id]
                emb = embeddings[emb_id]
                cat_emb.append(emb)
            cat_emb = torch.stack(cat_emb)
            cat_emb = torch.mean(cat_emb, dim=0)
            cat_emb = cat_emb / cat_emb.norm(dim=-1, keepdim=True)
            weights.append(cat_emb)

        weights = torch.stack(weights)
        self.weight = torch.nn.Parameter(weights.T)  # "n c -> c n"
        self.categories = cat_names
        self.catalog_cat_to_clsf_cat = {cat: i for i, cat in enumerate(clsf_ids)}
        self.clsf_cat_to_catalog_cat = {i: cat for i, cat in enumerate(clsf_ids)}

    @torch.no_grad()
    def extract_embeddings(self, image_paths: list[str]):
        embeddings = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path)
            image = self.model_config.val_preprocess_transform(image)
            image = image.to(self.device)
            image = image.unsqueeze(0)
            embeddings.append(self.model(image))
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def validate(
        self,
        catalog_path: str,
        batch_size: int = 10,
    ) -> MulticlassAccuracy:
        catalog_data = load_json(catalog_path)

        img_id_to_filename = {itm["id"]: itm["file_name"] for itm in catalog_data["images"]}

        targets = [itm["category_id"] for itm in catalog_data["annotations"]]

        image_paths = [img_id_to_filename[itm["image_id"]] for itm in catalog_data["annotations"]]
        embeddings = self.extract_embeddings(image_paths).cpu()

        dataset = TensorDataset(embeddings, targets)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        metric = MulticlassAccuracy(
            num_classes=len(self.categories),
            average=None,
            top_k=1,
        ).to(self.device)

        for batch in dataloader:
            emb, tgt = batch
            emb = emb.to(self.device)
            tgt = tgt.apply_(lambda x: self.catalog_cat_to_clsf_cat[x])
            tgt = tgt.to(self.device)
            pred = self.forward(emb).softmax(dim=-1)
            metric.update(pred, tgt)

        return metric.to("cpu")
