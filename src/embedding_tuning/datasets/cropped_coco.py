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

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from PIL import Image
from pycocotools.coco import COCO as _COCO
from torchvision.datasets.vision import StandardTransform, VisionDataset

from embedding_tuning.utils import load_json


@dataclass
class BoundingBox:
    coords: tuple[int, int, int, int]
    class_id: int
    label: str

    @classmethod
    def from_coco(
        cls,
        ann: dict,
        category: str,
    ):
        return BoundingBox(coords=ann["bbox"], class_id=ann["category_id"], label=category)


def COCO_from_dict(coco_dict: Dict) -> _COCO:  # noqa: N802
    coco = _COCO(None)
    coco.dataset = coco_dict
    coco.createIndex()
    return coco


class CroppedCOCODataset(VisionDataset):
    def __init__(  # noqa: PLR0913
        self,
        anno_path: str,
        root: str = "/",
        precrop_transform: Optional[Callable] = None,
        precrop_transforms: Optional[Callable] = None,
        crop_transform: Optional[Callable] = None,
        crop_transforms: Optional[Callable] = None,
        crop_target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            transform=precrop_transform,
            target_transform=None,
            transforms=precrop_transforms,
        )
        self.anno_path = anno_path
        self.root = root
        self.coco = COCO_from_dict(load_json(anno_path))
        self.ann_ids = self.coco.getAnnIds()

        self.cat_ids = sorted(self.coco.cats.keys())
        self.cat_id_to_name = {cat_id: self.coco.cats[cat_id]["name"] for cat_id in self.cat_ids}

        self.crop_transforms = crop_transforms
        self.crop_transform = crop_transform
        if self.crop_transforms is None:
            self.crop_transforms = StandardTransform(crop_transform, crop_target_transform)

    def __getitem__(self, index: int):
        ann = self.coco.anns[self.ann_ids[index]]
        img = self.coco.imgs[ann["image_id"]]
        path = os.path.join(self.root, img["file_name"])
        image = Image.open(path)
        label = self.cat_ids.index(ann["category_id"])

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        bbox = BoundingBox.from_coco(ann, self.cat_id_to_name[ann["category_id"]])
        cropped = image.crop(bbox.coords)

        if self.crop_transforms is not None:
            cropped, label = self.crop_transforms(cropped, label)

        return cropped, label

    def __len__(self) -> int:
        return len(self.ann_ids)
