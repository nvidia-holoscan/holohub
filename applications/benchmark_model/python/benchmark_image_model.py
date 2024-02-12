# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from onnx_helper import thread_cuda_init, thread_cuda_deinit, ONNXClassifierWrapper
from PIL import Image
import numpy as np
import cupy as cp

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.core import (
    ExecutionContext,
    InputContext,
    Operator,
    OperatorSpec,
    OutputContext,
    Tensor,
)
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator
from holoscan.gxf import Entity
import holoscan as hs

myfolder = "../images/"


def main(args):
    app = App(
        args.data,
        args.model_name,
        args.image_folder,
        args.multi_inference,
        args.only_inference,
        args.inference_postprocessing,
    )
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


class ReadImagesOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.ctx = None

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def start(self):
        if not os.path.exists(myfolder):
            print("folder does not exist")
            return False
        self.filenames = os.listdir(myfolder)
        self.index = 0

    def stop(self):
        # if self.ctx:
        #     thread_cuda_deinit(self.ctx)
        pass

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        PRECISION = np.float32

        image_path = os.path.join(
            myfolder, self.filenames[self.index]
        )  # Change this to the path of your image
        if not os.path.exists(image_path):
            print("image path does not exist")
            return
        self.index += 1
        print(image_path)
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        resize = 256
        crop_size = 224

        # Resize
        image = image.resize((resize, resize))

        # Center crop
        left = (resize - crop_size) // 2
        top = (resize - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        image = image.crop((left, top, right, bottom))

        image = np.array(image, dtype=PRECISION) / 255.0  # Normalize the pixel values to [0, 1]
        print(image.shape)

        # Transpose to match ONNX model input format (B, C, H, W)
        image = np.transpose(image, (2, 0, 1))
        print(image.shape)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        print(image.shape)
        # image = np.ascontiguousarray(image.copy())

        # image = np.moveaxis(image, 2, 0)[None]
        imagetensor = cp.asarray(image)
        imagetensor = Tensor.as_tensor(imagetensor)

        print("Shape:", imagetensor.shape)
        out_message = {}
        out_message["source_image"] = imagetensor

        op_output.emit(out_message, "out")
        # op_output.emit(image, "out")


class ImageClassificationOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.ctx = None

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def start(self):
        self.ctx = thread_cuda_init()

    def stop(self):
        if self.ctx:
            thread_cuda_deinit(self.ctx)

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        image = op_input.receive("in")
        # print(image)
        BATCH_SIZE = 1  # We process one image at a time
        PRECISION = np.float32
        N_CLASSES = 1000  # Our ResNet-50 is trained on a 1000 class ImageNet task

        # Load and preprocess the image
        trt_model = ONNXClassifierWrapper(
            "../data/resnet50/resnet_engine.trt",
            [BATCH_SIZE, N_CLASSES],
            target_dtype=PRECISION,
        )
        # Make predictions
        predictions = trt_model.predict(image)

        op_output.emit(predictions, "out")


class PrintTextOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def start(self):
        with open("../data/imagenet_classes.txt", "r") as f:
            self.labels = f.read().splitlines()
            self.labels = [l.strip() for l in self.labels]

    def stop(self):
        pass

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        predictions = op_input.receive("in").get("output")
        predictions = cp.asarray(predictions).get()
        # Get the top 5 probability scores and class labels
        top_5_indices = np.argsort(predictions[0])[::-1][:5]
        print(top_5_indices)

        # Print top 5 probability scores and class labels
        print("Top 5 Predictions:")
        for idx in top_5_indices:
            print(f"Class {idx}: {self.labels[idx]} - Probability {predictions[0][idx]*100:.2f}%")


class App(Application):
    def __init__(
        self,
        datapath,
        model_name,
        image_folder,
        num_inferences,
        only_inference,
        inference_postprocessing,
    ):
        """Initialize the application"""

        super().__init__()

        self.name = "Benchmark Image Model App"

        self.datapath = datapath
        self.model_name = model_name
        self.image_folder = image_folder
        self.num_inferences = num_inferences
        self.only_inference = only_inference
        self.inference_postprocessing = inference_postprocessing

        if not os.path.exists(self.datapath):
            raise ValueError(f"Data path {self.datapath} does not exist.")

        self.model_path = os.path.join(self.datapath, model_name)
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path {self.model_path} does not exist.")

        if not os.path.exists(self.image_folder):
            raise ValueError(f"Image folder {self.image_folder} does not exist.")

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        readimages = ReadImagesOp(self, CountCondition(self, 3), name="source")

        # model_path_map = {"own_model": "../data/resnet50/resnet_engine.trt"}
        model_path_map = {"own_model": "../data/resnet50/model.onnx"}
        pre_processor_map = {"own_model": ["source_image"]}
        inference_map = {"own_model": ["output"]}
        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=model_path_map,
            pre_processor_map=pre_processor_map,
            inference_map=inference_map,
            # is_engine_path=True,
            **self.kwargs("inference"),
        )
        # inference = ImageClassificationOp(self, name="inference")

        textoutput = PrintTextOp(self, name="textoutput")

        self.add_flow(readimages, inference, {("", "receivers")})
        self.add_flow(inference, textoutput, {("transmitter", "in")})
        # self.add_flow(readimages, inference)
        # self.add_flow(inference, textoutput, {("out", "in")})

        # attach count condition to inference


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(
        description="Benchmark Model Application.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    default_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=default_data_path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="identity_model.onnx",
        help="Path to the model directory",
    )
    parser.add_argument(
        "-a", "--image-folder", type=str, default="images", help="Path to the image folder"
    )
    parser.add_argument(
        "-i",
        "--only-inference",
        action="store_true",
        help="Only run inference, no post-processing or visualization",
    )
    parser.add_argument(
        "-p",
        "--inference-postprocessing",
        action="store_true",
        help="Run inference and post-processing, no visualization",
    )
    parser.add_argument(
        "-l",
        "--multi-inference",
        type=int,
        default=1,
        help="Number of inferences to run in parallel",
    )
    # add positional argument CONFIG which is just a string
    config_file = os.path.join(os.path.dirname(__file__), "benchmark_model.yaml")
    parser.add_argument("ConfigPath", nargs="?", default=config_file, help="Path to config file")

    args = parser.parse_args()
    main(args)
