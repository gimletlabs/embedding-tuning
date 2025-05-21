# Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import orjson
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(f_path):
    with open(f_path, "rb") as f:
        data = orjson.loads(f.read())
    return data


def load_jsonl(f_path):
    with open(f_path, "rb") as f:
        data = [orjson.loads(line) for line in f]
    return data


def dump_json(data, f_path, opts=orjson.OPT_INDENT_2, mode="wb"):
    with open(f_path, mode) as f:
        f.write(orjson.dumps(data, option=opts))
