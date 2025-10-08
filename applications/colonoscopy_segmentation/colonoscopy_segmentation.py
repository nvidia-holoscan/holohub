# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
from cupyx.scipy import ndimage
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType


class ContourOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structure = cp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.bool_)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def gpu_contour_categorical(self, mask):
        """GPU-accelerated contour detection for categorical masks"""
        num_classes = 1 if mask.size == 0 else int(cp.max(mask)) + 1
        contours = cp.zeros_like(mask)
        class_counter = cp.uint8(1)

        for class_id in range(1, num_classes):
            class_mask = mask == class_id
            eroded = cp.zeros_like(class_mask)

            ndimage.binary_erosion(
                class_mask, structure=self.structure, output=eroded, border_value=0
            )

            contours += (class_mask ^ eroded) * class_counter
            class_counter += 1

        return contours

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # To cupy array
        tensor = cp.asarray(in_message.get("out_tensor"), cp.uint8)
        tensor[:, :, 0] = self.gpu_contour_categorical(tensor[:, :, 0])

        op_output.emit({"out_tensor": tensor}, "out")


class ColonoscopyApp(Application):
    def __init__(self, data, source="replayer", contours=False):
        """Initialize the colonoscopy segmentation application

        Parameters
        ----------
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        contours : bool
            Show segmentation contours instead of mask (default: False).
        """

        super().__init__()

        # set name
        self.name = "Colonoscopy App"
        self.contours = contours

        # Optional parameters affecting the graph created by compose.
        self.source = source

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

        self.model_path_map = {
            "ultrasound_seg": os.path.join(self.sample_data_path, "colon.onnx"),
        }

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        is_aja = self.source.lower() == "aja"
        if is_aja:
            from holohub.aja_source import AJASourceOp

            source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
            drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
            drop_alpha_num_blocks = 2
            drop_alpha_channel = FormatConverterOp(
                self,
                name="drop_alpha_channel",
                pool=BlockMemoryPool(
                    self,
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=drop_alpha_block_size,
                    num_blocks=drop_alpha_num_blocks,
                ),
                cuda_stream_pool=cuda_stream_pool,
                **self.kwargs("drop_alpha_channel"),
            )
        else:
            video_dir = os.path.join(self.sample_data_path)
            if not os.path.exists(video_dir):
                raise ValueError(f"Could not find video data: {video_dir=}")
            source = VideoStreamReplayerOp(
                self, name="replayer", directory=video_dir, **self.kwargs("replayer")
            )

        width_preprocessor = 1350
        height_preprocessor = 1072
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 3
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_preprocessor"),
        )

        n_channels_inference = 2
        width_inference = 512
        height_inference = 512
        bpp_inference = 4
        inference_block_size = (
            width_inference * height_inference * n_channels_inference * bpp_inference
        )
        inference_num_blocks = 2
        segmentation_inference = InferenceOp(
            self,
            name="segmentation_inference_holoinfer",
            backend="trt",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=inference_block_size,
                num_blocks=inference_num_blocks,
            ),
            model_path_map=self.model_path_map,
            pre_processor_map={"ultrasound_seg": ["source_video"]},
            inference_map={"ultrasound_seg": "inference_output_tensor"},
            in_tensor_names=["source_video"],
            out_tensor_names=["inference_output_tensor"],
            enable_fp16=False,
            input_on_cuda=True,
            output_on_cuda=True,
            transmit_on_cuda=True,
        )

        if self.contours:
            contour_op = ContourOp(
                self,
                name="contour_op",
                pool=cuda_stream_pool,
            )

        postprocessor_block_size = width_inference * height_inference
        postprocessor_num_blocks = 2
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=postprocessor_block_size,
                num_blocks=postprocessor_num_blocks,
            ),
            **self.kwargs("segmentation_postprocessor"),
        )

        segmentation_visualizer = HolovizOp(
            self,
            name="segmentation_visualizer",
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_visualizer"),
        )

        if is_aja:
            self.add_flow(source, segmentation_visualizer, {("video_buffer_output", "receivers")})
            self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
            self.add_flow(drop_alpha_channel, segmentation_preprocessor)
        else:
            self.add_flow(source, segmentation_visualizer, {("", "receivers")})
            self.add_flow(source, segmentation_preprocessor)
        self.add_flow(segmentation_preprocessor, segmentation_inference, {("", "receivers")})
        self.add_flow(segmentation_inference, segmentation_postprocessor, {("transmitter", "")})
        if self.contours:
            self.add_flow(segmentation_postprocessor, contour_op, {("", "in")})
            self.add_flow(contour_op, segmentation_visualizer, {("out", "receivers")})
        else:
            self.add_flow(
                segmentation_postprocessor,
                segmentation_visualizer,
                {("", "receivers")},
            )


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Colonoscopy segmentation demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. If 'aja' use an AJA "
            "capture card as the source (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "--contours",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=("Show segmentation contours instead of mask (default: False)"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "colonoscopy_segmentation.yaml")
    else:
        config_file = args.config

    app = ColonoscopyApp(source=args.source, data=args.data, contours=args.contours)
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
