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


import torch
from absl import app, flags

from embedding_tuning.models.zero_shot_classifier import (
    ZeroShotClassifier,
)

flags.DEFINE_string("catalog_path", None, "Path to catalog COCO style json file")
flags.DEFINE_string("eval_catalog_path", None, "Path to eval catalog COCO style json file")
flags.DEFINE_string("model_type", None, "Model type")
flags.DEFINE_string("model_weight", None, "Model weight")
flags.DEFINE_string("model_name", None, "Model name")
flags.DEFINE_string("device", "cuda", "Device to run the model on")
flags.DEFINE_integer("k", 1, "use k items per class prototype")

FLAGS = flags.FLAGS


def main(_):
    classifier = ZeroShotClassifier(
        model_type=FLAGS.model_type,
        model_weight=FLAGS.model_weight,
        model_name=FLAGS.model_name,
        k=FLAGS.k,
    )
    classifier.to(FLAGS.device)
    classifier.build_zero_shot_classifier(FLAGS.catalog_path)

    metric = classifier.validate(catalog_path=FLAGS.eval_catalog_path)

    acc = metric.compute()
    mean_acc = acc.mean()

    print(f"Class level acc: {acc}")
    print(f"Mean accuracy: {mean_acc}")
    class_idxs = torch.where(acc != 1)[0]
    print(f"Class indices with acc < 1.0 {class_idxs}")


if __name__ == "__main__":
    app.run(main)
