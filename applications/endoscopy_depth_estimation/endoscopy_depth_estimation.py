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

import os
from argparse import ArgumentParser

import cupy as cp
import cv2
import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import UnboundedAllocator


def gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    assert len(arr.shape) in (2, 3), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"
    type_map = {
        cp.dtype("uint8"): cv2.CV_8U,
        cp.dtype("int8"): cv2.CV_8S,
        cp.dtype("uint16"): cv2.CV_16U,
        cp.dtype("int16"): cv2.CV_16S,
        cp.dtype("int32"): cv2.CV_32S,
        cp.dtype("float32"): cv2.CV_32F,
        cp.dtype("float64"): cv2.CV_64F,
    }
    depth = type_map.get(arr.dtype)
    assert depth is not None, "Unsupported CuPy array dtype"
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    mat_type = depth + ((channels - 1) << 3)

    mat = cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__["shape"][1::-1],
        mat_type,
        arr.__cuda_array_interface__["data"][0],
    )
    return mat


def gpumat_to_cupy(gpu_mat: cv2.cuda.GpuMat) -> cp.ndarray:
    w, h = gpu_mat.size()
    size_in_bytes = gpu_mat.step * w
    shapes = (h, w, gpu_mat.channels())
    assert gpu_mat.channels() <= 3, "Unsupported GpuMat channels"
    dtype = None
    if gpu_mat.type() in [cv2.CV_8U, cv2.CV_8UC1, cv2.CV_8UC2, cv2.CV_8UC3]:
        dtype = cp.uint8
    elif gpu_mat.type() == cv2.CV_8S:
        dtype = cp.int8
    elif gpu_mat.type() == cv2.CV_16U:
        dtype = cp.uint16
    elif gpu_mat.type() == cv2.CV_16S:
        dtype = cp.int16
    elif gpu_mat.type() == cv2.CV_32S:
        dtype = cp.int32
    elif gpu_mat.type() == cv2.CV_32F:
        dtype = cp.float32
    elif gpu_mat.type() == cv2.CV_64F:
        dtype = cp.float64

    assert dtype is not None, "Unsupported GpuMat type"

    mem = cp.cuda.UnownedMemory(gpu_mat.cudaPtr(), size_in_bytes, owner=gpu_mat)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    cp_out = cp.ndarray(
        shapes,
        dtype=dtype,
        memptr=memptr,
        strides=(gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
    )

    return cp_out


class DepthPostProcessingOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("depth_in")
        spec.input("video_in")  # .condition(ConditionType.NONE)
        spec.output("out")
        spec.param("in_tensor_name", "inference_output_tensor")
        spec.param("scale_min", -2)
        spec.param("scale_max", 2.44)

    def compute(self, op_input, op_output, context):
        message_depth = op_input.receive("depth_in")
        message_video = op_input.receive("video_in")
        input_depth = message_depth.get(self.in_tensor_name)
        input_video = message_video.get("source_video")

        # Depth processing
        cp_depth = cp.asarray(input_depth)
        cp_depth = cp_depth[:, 0]
        cp_depth = cp.moveaxis(cp_depth, 0, -1)
        cp_depth = (cp_depth * 255.0).astype(cp.uint8)

        # Video processing
        cp_video = cp.asarray(input_video)
        if cp_video.dtype in ["float32", "float64"]:
            # Video has been processed
            cp_video = 255 * (cp_video - self.scale_min) / (self.scale_max - self.scale_min)
        alpha_channel = cp.ones(cp_video.shape[:-1] + (1,), dtype=cp_video.dtype)
        cp_video = cp.concatenate((cp_video, 255 * alpha_channel), axis=-1)
        cp_video = cp_video.astype(cp.uint8)

        out_message = Entity(context)
        out_message.add(hs.as_tensor(cp_video), "color_data")
        out_message.add(hs.as_tensor(cp_depth), "depth_data")
        op_output.emit(out_message, "out")


class CPUCLAHEOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")

        frame = cp.asnumpy(message.get(""))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_frame)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv_frame = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)

        frame = cp.asarray(frame)

        out_message = Entity(context)
        out_message.add(hs.as_tensor(frame), "")
        op_output.emit(out_message, "out")


class CUDACLAHEOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        stream = cv2.cuda_Stream()
        message = op_input.receive("in")

        cp_frame = cp.asarray(message.get(""))  # CuPy array
        cv_frame = gpumat_from_cp_array(cp_frame)  # GPU OpenCV mat

        hsv_frame = cv2.cuda.cvtColor(cv_frame, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.cuda.split(hsv_frame)

        clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v, stream=stream)

        # Merge doesn't return a GpuMat by default, we force it here.
        hsv_merge = cv2.cuda_GpuMat(hsv_frame.size(), hsv_frame.type())
        cv2.cuda.merge([h, s, v], hsv_merge)

        cv_frame = cv2.cuda.cvtColor(hsv_merge, cv2.COLOR_HSV2RGB)

        cp_frame = gpumat_to_cupy(cv_frame)
        cp_frame = cp.ascontiguousarray(cp_frame)

        out_message = Entity(context)
        out_message.add(hs.as_tensor(cp_frame), "")
        op_output.emit(out_message, "out")


class DepthApp(Application):
    def __init__(self, data=None, model=None, apply_clahe=True):
        """Initialize the endoscopy depth estimation application

        Parameters
        ----------
        apply_clahe : bool
            When set to True (the default), adaptive histogram equalization is applied.
            A threshold is applied to the resulting frame and the masked outputs are inpainted.
        """
        super().__init__()

        # set name
        self.name = "Depth App"

        data = os.environ.get("HOLOHUB_DATA_PATH", "../data") if data is None else data
        model = data if model is None else model

        self.data_path = data
        self.model_path = model
        self.model_path_map = {"model": os.path.join(self.model_path, "dispnet_nhwc_fold.onnx")}
        self.apply_clahe = apply_clahe

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")
        video_dir = self.data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

        # Define the replayer and holoviz operators
        replayer = VideoStreamReplayerOp(
            self, name="replayer", directory=video_dir, **self.kwargs("replayer")
        )

        preprocessor_kwargs = self.kwargs("preprocessor")
        if self.apply_clahe:
            clahe = CUDACLAHEOp(self, name="clahe", pool=host_allocator)
            viz_preprocessor = FormatConverterOp(
                self,
                name="viz_preprocessor",
                pool=host_allocator,
                **self.kwargs("viz_preprocessor"),
            )

        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=host_allocator,
            **preprocessor_kwargs,
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=self.model_path_map,
            **self.kwargs("inference"),
        )

        postprocessor = DepthPostProcessingOp(
            self, name="postprocessor", pool=host_allocator, *self.kwargs("postprocessor")
        )

        visualizer = HolovizOp(
            self, name="holoviz", allocator=host_allocator, **self.kwargs("holoviz")
        )

        # Define the workflow
        if self.apply_clahe:
            self.add_flow(replayer, clahe, {("output", "in")})
            self.add_flow(clahe, preprocessor, {("out", "source_video")})
            self.add_flow(preprocessor, inference, {("tensor", "receivers")})

            self.add_flow(replayer, viz_preprocessor, {("output", "source_video")})
            self.add_flow(inference, postprocessor, {("transmitter", "depth_in")})
            self.add_flow(viz_preprocessor, postprocessor, {("tensor", "video_in")})
        else:
            self.add_flow(replayer, preprocessor, {("output", "source_video")})
            self.add_flow(preprocessor, inference, {("tensor", "receivers")})

            self.add_flow(inference, postprocessor, {("transmitter", "depth_in")})
            self.add_flow(preprocessor, postprocessor, {("tensor", "video_in")})

        self.add_flow(postprocessor, visualizer, {("out", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Endoscopy depth estimation demo application.")
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
        "--clahe",
        action="store_true",
        help="Use Adaptive Histogram Equalization (CLAHE)",
    )
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "endoscopy_depth_estimation.yaml")

    app = DepthApp(data=args.data, model=args.model, apply_clahe=args.clahe)
    app.config(config_file)
    app.run()
