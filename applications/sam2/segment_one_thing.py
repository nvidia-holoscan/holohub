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

import os
from argparse import ArgumentParser

import torch  # noqa: F401
from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator
from sam2operator import FormatInferenceInputOp, PointPublisher, SAM2Operator, SamPostprocessorOp


class SegmentOneThingApp(Application):
    """Segment one thing application segments one object in the current frame of a video stream,
    based on the position of the point query."""

    def __init__(self, source="v4l2", save_intermediate=False, verbose=False):
        super().__init__()
        self.name = "Segment one thing App"
        self.source = source
        self.verbose = verbose
        self.save_intermediate = save_intermediate

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        if self.source == "v4l2":
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
            source_output = "signal"
        else:
            raise ValueError(f"Unknown source type: {self.source}")

        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
            verbose=self.verbose,
        )

        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )

        # create a point publisher, in this case we are using the default point publisher
        # which publishes a single point moving on the perimeter of a circle.
        point_publisher_args = self.kwargs("point_publisher")
        point_publisher = PointPublisher(
            self,
            name="point_publisher",
            allocator=pool,
            **point_publisher_args,
        )

        # create sam2 operator. The SAM2Operator is a wrapper around the SAM2ImagePredictor class.
        # It uses pytorch to load the model and run inference on the input image.
        sam2_args = self.kwargs("sam2")
        sam2 = SAM2Operator(
            self,
            name="sam2",
            allocator=pool,
            **sam2_args,
        )
        # create postprocessor operator.
        # The posrprocessor operator scales the output of the model to the input image size
        # and applies a colormap to the output.
        postprocessor_args = self.kwargs("postprocessor")
        postprocessor = SamPostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            save_intermediate=self.save_intermediate,
            verbose=self.verbose,
            **postprocessor_args,
        )

        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **self.kwargs("holoviz"))

        # Holoviz
        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(point_publisher, sam2, {("out", "point_coords")})
        self.add_flow(point_publisher, holoviz, {("point_viz", "receivers")})
        self.add_flow(format_input, sam2, {("", "image")})
        self.add_flow(sam2, postprocessor, {("out", "in")})
        self.add_flow(postprocessor, holoviz, {("out", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Segment one thing application")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2"],
        default="v4l2",
        help=("If 'v4l2', uses the v4l2 device specified in the yaml file."),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    parser.add_argument(
        "-si", "--save_intermediate", action="store_true", help="Save intermediate tensors to disk"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "segment_one_thing.yaml")
    else:
        config_file = args.config

    app = SegmentOneThingApp(args.source, args.save_intermediate, args.verbose)
    app.config(config_file)
    app.run()
