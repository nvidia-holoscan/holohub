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

import glob
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import blosc
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime
import torch
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from PIL import Image

LABEL_COLORMAP = torch.tensor(
    [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
)


class LoadDataOp(Operator):
    def __init__(self, *args, data_path=None, **kwargs):
        self.count = 0
        self.cube_file_list = sorted(glob.glob(os.path.join(data_path, "*.blosc")))
        self.rgb_file_list = sorted(glob.glob(os.path.join(data_path, "*.png")))
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("cube")
        spec.output("rgb")

    def compute(self, op_input, op_output, context):
        cube_path = Path(self.cube_file_list[self.count])
        cube = self.decompress_file(cube_path)

        rgb_path = self.rgb_file_list[self.count]
        rgb = np.array(Image.open(rgb_path))

        cube = torch.as_tensor(cube, dtype=torch.float32)

        self.count += 1
        op_output.emit(cube, "cube")
        op_output.emit(rgb, "rgb")

        if self.count == len(self.cube_file_list):
            self.count = 0

    def decompress_file(self, path):
        """
        Decompresses a blosc file.

        Args:
            path: File to the blosc data.

        Returns: Decompressed array data.
        """
        res = {}

        with path.open("rb") as f:
            meta = pickle.load(f)
            shape, dtype = meta
            data = f.read()
            array = np.empty(shape=shape, dtype=dtype)
            blosc.decompress_ptr(data, array.__array_interface__["data"][0])

            res = array

        return res


class HyperspectralInferenceOp(Operator):
    def __init__(self, *args, model_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        self.ort_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("rgb")
        spec.output("image")
        spec.output("segmentation")

    def compute(self, op_input, op_output, context):
        cube = op_input.receive("cube")
        rgb = op_input.receive("rgb")

        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(cube[None])}
        segmentation = self.ort_session.run(None, ort_inputs)[0]

        op_output.emit(rgb, "image")
        op_output.emit(segmentation, "segmentation")

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class HyperspectralVizOp(Operator):
    def __init__(self, *args, output_folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_folder = output_folder

    def setup(self, spec: OperatorSpec):
        spec.input("segmentation")
        spec.input("rgb")

    def compute(self, op_input, op_output, context):
        seg = op_input.receive("segmentation")
        rgb = op_input.receive("rgb")

        seg = np.moveaxis(np.squeeze(seg, axis=0), 0, -1).argmax(axis=-1)
        rgb_seg = LABEL_COLORMAP[seg]

        plt.figure(figsize=(18, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.gca().set_title("RGB image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(rgb_seg)
        plt.gca().set_title("Segmentation")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(rgb)
        plt.imshow(rgb_seg, alpha=0.5)
        plt.gca().set_title("Overlay")
        plt.axis("off")

        plt.savefig(
            os.path.join(self.output_folder, "result.png"), bbox_inches="tight", pad_inches=0
        )
        plt.close()


class HSApp(Application):
    def __init__(self, data=None, model=None, output_folder=None, count=-1):
        """Hyperspectral segmentation application"""
        super().__init__()
        data = os.environ.get("HOLOHUB_DATA_PATH", "../data") if data is None else data
        model = data if model is None else model

        self.count = count
        self.data_path = data
        self.model_dir = model
        self.model_path = os.path.join(self.model_dir, "hyperspectral_segmentation_nhwc.onnx")
        self.output_folder = self.model_dir if output_folder is None else output_folder

    def compose(self):
        rgb_file_list = glob.glob(os.path.join(self.data_path, "*.png"))
        count = self.count if (self.count != -1) else len(rgb_file_list)
        loader = LoadDataOp(
            self, CountCondition(self, count), data_path=self.data_path, name="data_loader"
        )
        inference = HyperspectralInferenceOp(self, model_path=self.model_path, name="inference")
        viz = HyperspectralVizOp(self, output_folder=self.output_folder, name="viz")

        # Define the workflow
        self.add_flow(loader, inference, {("cube", "cube"), ("rgb", "rgb")})
        self.add_flow(inference, viz, {("segmentation", "segmentation"), ("image", "rgb")})


if __name__ == "__main__":
    parser = ArgumentParser(description="Hyperspectral segmentation application.")

    parser.add_argument(
        "-o",
        "--output_folder",
        default="none",
        help=("Set the model path"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help="Set the data path",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Set the model path",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=-1,
        help=("Number of images to process (by default, loops through dataset once)"),
    )

    args = parser.parse_args()

    app = HSApp(
        data=args.data, model=args.model, output_folder=args.output_folder, count=args.count
    )
    app.run()

    print("Application finished successfully.")
