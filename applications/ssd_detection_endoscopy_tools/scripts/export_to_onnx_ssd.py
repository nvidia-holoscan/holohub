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

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from examples.SSD300_inference import build_predictor

sys.path.append("/workspace/SSD")
sys.path.append("/workspace/SSD/dle")
sys.path.append("/workspace/SSD/examples")
sys.path.append("/workspace/SSD/ssd")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            "overwriting your saved model."
        )
    model = load_model_and_export(modelname, outname, height, width)
