# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle
from pathlib import Path

import blosc
import imageio
import numpy as np
import onnx
import onnxruntime
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.onnx
import torchvision.transforms as transforms


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ModelImage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        return x


class ModelImage(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 100

        ArchitectureClass = getattr(smp, "Unet")
        self.architecture = ArchitectureClass(
            classes=19,
            in_channels=channels,
            **{"encoder_name": "efficientnet-b5", "encoder_weights": "imagenet"},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.architecture(x)

        return x


def decompress_file(path):
    """
    Decompresses a blosc file.

    Args:
        path: File to the blosc data.

    Returns: Decompressed array data. Depending on the file, this will either be directly the numpy array or a dict with all numpy arrays.
    """
    res = {}

    with path.open("rb") as f:
        meta = pickle.load(f)
        if isinstance(meta, tuple):
            shape, dtype = meta
            data = f.read()
            array = np.empty(shape=shape, dtype=dtype)
            blosc.decompress_ptr(data, array.__array_interface__["data"][0])

            res = array
        else:
            for name, (shape, dtype, size) in meta.items():
                data = f.read(size)
                array = np.empty(shape=shape, dtype=dtype)
                blosc.decompress_ptr(data, array.__array_interface__["data"][0])
                res[name] = array

    return res


run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
model = Model()

# Load model from https://github.com/IMSY-DKFZ/htc
dict = torch.load(
    "image@2022-02-03_22-58-44_generated_default_model_comparison/image/2022-02-03_22-58-44_generated_default_model_comparison/fold_P041,P060,P069/epoch=46-dice_metric=0.87.ckpt"
)["state_dict"]
dict = {k: v for k, v in dict.items() if "ce_loss" not in k}

model.load_state_dict(dict)
model.eval()

# Use files from https://www.heiporspectral.org/ to get input/output shapes
img = decompress_file(
    Path("HeiPorSPECTRAL_example/intermediates/preprocessing/L1/P086#2021_04_15_09_22_02.blosc")
)
transform = transforms.ToTensor()

img = transform(img).to(torch.float32)

orig_output = model(img[None])
output = torch.argmax(orig_output, dim=1)

rgb = imageio.imread(
    "HeiPorSPECTRAL_example/intermediates/rgb_crops/P086/P086#2021_04_15_09_22_02.png"
)
rgb = torch.as_tensor(np.moveaxis(rgb, -1, 0))

# Export the model to ONNX
torch.onnx.export(
    model,  # model being run
    img[None],  # model input (or a tuple for multiple inputs)
    "hyperspectral_segmentation.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=11,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
)

onnx_model = onnx.load("hyperspectral_segmentation.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(
    "hyperspectral_segmentation.onnx", providers=["CPUExecutionProvider"]
)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img[None])}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(orig_output), ort_outs[0], rtol=1e-03, atol=1e-04)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
