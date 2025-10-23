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

import os
from argparse import ArgumentParser

import numpy as np
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

from holohub.aja_source import AJASourceOp

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    )


class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.
    Following the example of tensor_interop.py and ping.py4

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context, scores_threshold=0.3):
        # Get input message which is a dictionary
        in_message = op_input.receive("in")
        # Convert input to numpy array (using CuPy) via .get()
        bboxes = cp.asarray(
            in_message["inference_output_detection_boxes"]
        ).get()  # (nbatch, nboxes, ncoord)
        scores = cp.asarray(
            in_message["inference_output_detection_scores"]
        ).get()  # (nbatch, nboxes)
        # Threshold scores and prune boxes
        ix = scores.flatten() >= scores_threshold
        if np.all(ix == False):
            bboxes = np.zeros([1, 2, 2], dtype=np.float32)
        else:
            bboxes = bboxes[:, ix, :]
            # Make box shape compatible with Holoviz
            bboxes = np.reshape(bboxes, (1, -1, 2))  # (nbatch, nboxes*2, ncoord/2)
        # Create output message
        out_message = {}
        out_message["rectangles"] = bboxes
        op_output.emit(out_message, "out")


class SSDDetectionApp(Application):
    def __init__(self, source="replayer"):
        """Initialize the ssd detection application

        Parameters
        ----------
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "SSD Detection App"

        # Optional parameters affecting the graph created by compose.
        self.source = source

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel

        is_aja = self.source.lower() == "aja"
        if is_aja:
            width = 1920
            height = 1080
            source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
            drop_alpha_block_size = width * height * n_channels * bpp
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
                **self.kwargs("drop_alpha_channel"),
            )
        else:
            source = VideoStreamReplayerOp(self, name="replayer", **self.kwargs("replayer"))

        width_preprocessor = 300
        height_preprocessor = 300
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            **self.kwargs("detection_preprocessor"),
        )

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=[
                dict(name="", type="color"),
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=0.5,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                ),
            ],
            **self.kwargs("detection_visualizer"),
        )

        if is_aja:
            self.add_flow(source, detection_visualizer, {("video_buffer_output", "receivers")})
            self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
            self.add_flow(drop_alpha_channel, detection_preprocessor)
        else:
            self.add_flow(source, detection_visualizer, {("", "receivers")})
            self.add_flow(source, detection_preprocessor)

        self.add_flow(detection_preprocessor, detection_inference, {("", "receivers")})
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(detection_postprocessor, detection_visualizer, {("out", "receivers")})


def main():
    # Parse args
    parser = ArgumentParser(description="SSD Detection demo application.")
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
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), "ssd_endo_model_with_NMS.yaml")

    app = SSDDetectionApp(source=args.source)
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
