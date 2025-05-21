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

import random
from itertools import groupby
from pathlib import Path
from typing import Union

import torchvision
from absl import app, flags, logging
from tqdm import tqdm

from embedding_tuning.utils import dump_json, load_json, seed_all

DATASET_MAP = {
    "flowers102": torchvision.datasets.Flowers102,
    "food101": torchvision.datasets.Food101,
}

flags.DEFINE_enum(
    "dataset",
    "flowers102",
    DATASET_MAP.keys(),
    "The dataset to prepare the data for.",
)

flags.DEFINE_float(
    "holdout_ratio",
    0.2,
    "The ratio of holdout categories.",
)
flags.DEFINE_integer(
    "seed",
    42,
    "The seed for the random number generator.",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "The directory to save the data.",
    required=True,
)

FLAGS = flags.FLAGS


def group_annotations(annotations, key_fn=lambda x: x["category_id"]) -> dict[str, list]:
    annotations = sorted(annotations, key=key_fn)
    category_groups = dict()
    for category_id, category_data in groupby(annotations, key=key_fn):
        category_groups[category_id] = list(category_data)
    return category_groups


def split_annotations_pub_holdout_categories(  # noqa: PLR0913
    coco_json_path: Union[str, Path],
    holdout_json_path: Union[str, Path],
    public_json_path: Union[str, Path],
    holdout_cats: set[int],
    public_cats: set[int],
    verbose: bool = False,
):
    """
    Simple function to split COCO annotations into training and test sets, with test_split portion of images allocated to the test set.
    Split is performed based on boxes, assuming that each image contains a single box.
    """

    coco_data = load_json(coco_json_path)
    image_category_groups = group_annotations(coco_data["annotations"])

    pub_annotations, hold_annotations = [], []
    pub_image_ids, hold_image_ids = [], []

    for category_id, category_data in image_category_groups.items():
        if verbose:
            logging.info(f"{category_id=} has {len(category_data)} annotations.")

        if category_id in public_cats:
            pub_image_ids.extend([anno["image_id"] for anno in category_data])
            pub_annotations.extend(category_data)

        elif category_id in holdout_cats:
            hold_image_ids.extend([anno["image_id"] for anno in category_data])
            hold_annotations.extend(category_data)

    pub_images = [img for img in coco_data["images"] if img["id"] in pub_image_ids]
    hold_images = [img for img in coco_data["images"] if img["id"] in hold_image_ids]

    # Write training set annotations to file.
    coco_data["images"] = pub_images
    coco_data["annotations"] = pub_annotations

    dump_json(coco_data, public_json_path)

    logging.info(f"Public set annotations saved to: {public_json_path}")

    # Write test set annotations to file.
    coco_data["images"] = hold_images
    coco_data["annotations"] = hold_annotations

    dump_json(coco_data, holdout_json_path)

    logging.info(f"Holdout set annotations saved to: {holdout_json_path}")


def to_coco(dataset: torchvision.datasets.Flowers102, output_dir: Path, split: str) -> dict:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    data["categories"] = [{"id": idx, "name": name} for idx, name in enumerate(dataset.classes)]

    for idx in tqdm(range(len(dataset))):
        image, label = dataset[idx]
        img_path = image_dir / f"image_{split}_{idx:05d}.jpg"
        image.save(img_path)

        data["images"].append(
            {
                "id": idx,
                "width": image.size[0],
                "height": image.size[1],
                "file_name": str(img_path.resolve()),
            }
        )

        data["annotations"].append(
            {
                "id": idx,
                "image_id": idx,
                "category_id": label,
                "bbox": [0, 0, image.size[0], image.size[1]],
                "area": image.size[0] * image.size[1],
                "iscrowd": False,
            }
        )

    return data


def main(_):
    FLAGS.output_dir = Path(FLAGS.output_dir)

    anno_dir = FLAGS.output_dir / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = []
    dataset_splits = ["train", "test", "val", "valid"]
    for split_name in dataset_splits:
        try:
            dataset = DATASET_MAP[FLAGS.dataset](
                root=FLAGS.output_dir,
                split=split_name,
                download=True,
            )
        except Exception as e:
            logging.error(f"Dataset {FLAGS.dataset} does not have split {split_name}: {e}")
            continue

        coco = to_coco(dataset, FLAGS.output_dir, split=split_name)
        manifest_path = anno_dir / f"{split_name}.json"
        dump_json(coco, manifest_path)
        manifest_paths.append(manifest_path)

    if len(manifest_paths) == 0:
        raise ValueError("No splits found for dataset.")

    categories = load_json(manifest_paths[0])["categories"]
    cat_ids = [cat["id"] for cat in categories]
    seed_all(FLAGS.seed)
    random.shuffle(cat_ids)

    n_holdout_cats = int(len(cat_ids) * FLAGS.holdout_ratio)
    holdout_cat_ids = set(cat_ids[:n_holdout_cats])
    public_cat_ids = set(cat_ids[n_holdout_cats:])

    logging.info(f"Number of categories: {len(cat_ids)}")
    logging.info(f"Number of holdout categories: {len(holdout_cat_ids)}")
    logging.info(f"Number of public categories: {len(public_cat_ids)}")

    for manifest_path in manifest_paths:
        split_name = manifest_path.stem
        holdout_path = manifest_path.with_name(f"holdout_{split_name}.json")
        public_path = manifest_path.with_name(f"public_{split_name}.json")

        if not holdout_path.exists() or not public_path.exists():
            split_annotations_pub_holdout_categories(
                manifest_path,
                holdout_path,
                public_path,
                holdout_cat_ids,
                public_cat_ids,
            )


if __name__ == "__main__":
    app.run(main)
