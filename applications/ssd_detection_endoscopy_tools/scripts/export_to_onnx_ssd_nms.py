# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import itertools
import os
import sys
from math import sqrt

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn.functional as F
from examples.SSD300_inference import build_predictor

sys.path.append("/workspace/SSD")
sys.path.append("/workspace/SSD/dle")
sys.path.append("/workspace/SSD/examples")
sys.path.append("/workspace/SSD/ssd")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ssd_boxes(device):
    """Get boxes for converting SSD model output from xywh to x0y0x1y1"""
    # dboxes300_coco
    feat_size = [38, 19, 10, 5, 3, 1]
    fig_size = 300
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fk = fig_size / np.array(steps)
    default_boxes = []
    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for alpha in aspect_ratios[idx]:
            w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))
        for w, h in all_sizes:
            for i, j in itertools.product(range(sfeat), repeat=2):
                cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                default_boxes.append((cx, cy, w, h))

    dboxes = torch.tensor(default_boxes, dtype=torch.float, device=device)
    dboxes.clamp_(min=0, max=1)
    dboxes = dboxes.unsqueeze(dim=0)

    return dboxes


class ConvertBoxes(torch.nn.Module):
    def __init__(self, dboxes):
        """Torch module for converting SSD model output from xywh to x0y0x1y1,
        compatible with EfficientNMS_TRT.

        Note, also softmaxes logit scores (important).

        """
        super().__init__()
        self.scale_xy = 0.1
        self.scale_wh = 0.2
        self.dboxes = dboxes

    def forward(self, x):
        x = list(x)

        x[0] = x[0].permute((0, 2, 1))
        x[1] = x[1].permute((0, 2, 1))

        x[0][:, :, :2] = self.scale_xy * x[0][:, :, :2]
        x[0][:, :, 2:] = self.scale_wh * x[0][:, :, 2:]
        x[0][:, :, :2] = x[0][:, :, :2] * self.dboxes[:, :, 2:] + self.dboxes[:, :, :2]
        x[0][:, :, 2:] = x[0][:, :, 2:].exp() * self.dboxes[:, :, 2:]

        x0, y0, x1, y1 = (
            x[0][:, :, 0] - 0.5 * x[0][:, :, 2],
            x[0][:, :, 1] - 0.5 * x[0][:, :, 3],
            x[0][:, :, 0] + 0.5 * x[0][:, :, 2],
            x[0][:, :, 1] + 0.5 * x[0][:, :, 3],
        )

        x[0][:, :, 0] = x0
        x[0][:, :, 1] = y0
        x[0][:, :, 2] = x1
        x[0][:, :, 3] = y1

        x[1] = F.softmax(x[1], dim=-1)

        x = tuple(x)

        return x


def load_model_and_export(modelname, outname, height, width):
    """
    Loading a model by name.
    Args:
        modelname: a whole path name of the model that need to be loaded.
        outname: a name for output onnx model.
        height: input images' height.
        width: input images' width.
    """
    isopen = os.path.exists(modelname)
    if not isopen:
        raise Exception("The specified model to load does not exist!")

    model = build_predictor(modelname)

    # Parameters for box conversion
    dboxes = get_ssd_boxes(device)

    model = torch.nn.Sequential(
        model,
        ConvertBoxes(dboxes),  # Append box conversion module
    )

    model = model.cuda()
    model = model.eval()

    np.random.seed(0)
    x = np.random.random((1, 3, width, height))
    x = torch.tensor(x, dtype=torch.float32)
    x = x.cuda()
    torch_out = model(x)
    input_names = ["INPUT__0"]
    output_names = ["OUTPUT__LOC", "OUTPUT__LABEL"]

    torch.onnx.export(
        model,  # model to save
        x,  # model input
        outname,  # model save path
        export_params=True,
        verbose=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        dynamic_axes={
            "INPUT__0": {0: "batch_size"},
            "OUTPUT__LOC": {0: "batch_size"},
            "OUTPUT__LABEL": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(outname)
    onnx.checker.check_model(onnx_model, full_check=True)
    ort_session = onnxruntime.InferenceSession(outname)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(["OUTPUT__LOC", "OUTPUT__LABEL"], ort_inputs)
    numpy_torch_out_0 = to_numpy(torch_out[0])
    numpy_torch_out_1 = to_numpy(torch_out[1])
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(numpy_torch_out_0, ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(numpy_torch_out_1, ort_outs[1], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the original model for converting.
    parser.add_argument(
        "--model",
        type=str,
        default=r"/workspace/SSD/checkpoints/epoch_64.pt",
        help="Input an existing model weight",
    )

    # path to save the onnx model.
    parser.add_argument(
        "--outpath",
        type=str,
        default=r"/workspace/SSD/epoch_64.onnx",
        help="A path to save the onnx model.",
    )

    parser.add_argument("--width", type=int, default=300, help="Width for exporting onnx model.")

    parser.add_argument("--height", type=int, default=300, help="Height for exporting onnx model.")

    args = parser.parse_args()
    modelname = args.model
    outname = args.outpath
    height = args.height
    width = args.width

    if os.path.exists(outname):
        raise Exception(
            "The specified outpath already exists! Change the outpath to avoid "
            "overwriting your saved model. "
        )
    model = load_model_and_export(modelname, outname, height, width)
