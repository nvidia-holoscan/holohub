# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import torch
from torchvision.models import ResNet50_Weights, detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

os.environ["TORCH_HOME"] = os.getcwd()

model_file = "frcnn_resnet50_t.pt"
if len(sys.argv) > 1:
    model_file = sys.argv[1]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

det_model = detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    progress=True,
    weights_backbone=ResNet50_Weights.DEFAULT,
).to(DEVICE)

det_model.eval()
det_model_script = torch.jit.script(det_model)
det_model_script.save(model_file)
