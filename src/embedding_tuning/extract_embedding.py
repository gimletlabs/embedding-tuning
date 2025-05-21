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

from pathlib import Path

import torch
from absl import app, flags, logging

from embedding_tuning.model_registry import ModelType
from embedding_tuning.models.zero_shot_classifier import ZeroShotClassifier
from embedding_tuning.utils import load_json

flags.DEFINE_string(
    "anno_path",
    None,
    "Path to annotations file",
    required=True,
)
flags.DEFINE_enum(
    "model_type",
    ModelType.OPEN_CLIP.name,
    [itm.name for itm in ModelType],
    "Features extractor model type to use for angular margin finetuning.",
)
flags.DEFINE_string(
    "model_name",
    "ViT-B-16",
    "Embedding model name",
)
flags.DEFINE_string(
    "embedding_dir",
    None,
    "Path to embeddings output dir",
    required=True,
)
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to pretrained checkpoint",
)
flags.DEFINE_string(
    "device",
    "cuda",
    "Device to run on",
)


FLAGS = flags.FLAGS


def extract_embeddings(
    anno_path: str | Path,
    model_name: str,
    ckpt_path: str | None,
    embedding_dir: str | Path,
    device: str = "cpu",
):
    data = load_json(anno_path)
    image_paths = []
    image_ids = []
    for itm in data["images"]:
        image_paths.append(itm["file_name"])
        image_ids.append(itm["id"])

    model = ZeroShotClassifier(
        model_type=FLAGS.model_type,
        model_weight=FLAGS.model_weight,
        model_name=FLAGS.model_name,
        k=FLAGS.k,
    )
    model.to(device)
    embeddings = model.extract_embeddings(image_paths).cpu()
    emb_path = embedding_dir / f"{anno_path.stem}.pt"

    data = dict(
        embedding=embeddings,
        image_id=image_ids,
        image_path=image_paths,
    )

    torch.save(data, emb_path)
    logging.info(f"Saved embeddings to {emb_path}")


def main(_):
    anno_path = Path(FLAGS.anno_path)
    embedding_dir = Path(FLAGS.embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    extract_embeddings(
        anno_path,
        FLAGS.model_name,
        FLAGS.checkpoint_path,
        embedding_dir,
        FLAGS.device,
    )


if __name__ == "__main__":
    app.run(main)
