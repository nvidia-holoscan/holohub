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

from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator

from operators import (
    DecoderConfigurator,
    FormatInferenceInputOp,
    PointPublisher,
    SamPostprocessorOp,
)


class SegmentOneThingApp(Application):
    def __init__(self, source="v4l2", save_intermediate=False, verbose=False):
        """Initialize the body pose estimation application"""

        super().__init__()

        # set name
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

        inference_encoder_args = self.kwargs("inference")
        assert (
            "model_path_map" in inference_encoder_args and inference_encoder_args["model_path_map"]
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            **inference_encoder_args,
        )

        # create a point publisher
        point_publisher_args = self.kwargs("point_publisher")
        point_publisher = PointPublisher(
            self,
            name="point_publisher",
            allocator=pool,
            **point_publisher_args,
        )

        decoder_configurator = DecoderConfigurator(
            self,
            allocator=pool,
            save_intermediate=self.save_intermediate,
            verbose=self.verbose,
            **self.kwargs("decoder_configurator"),
        )

        inference_decoder_args = self.kwargs("inference_decoder")
        assert (
            "model_path_map" in inference_decoder_args and inference_decoder_args["model_path_map"]
        )
        inference_decoder = InferenceOp(
            self,
            name="inference_decoder",
            allocator=pool,
            **inference_decoder_args,
        )

        postprocessor = SamPostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            save_intermediate=self.save_intermediate,
            verbose=self.verbose,
        )

        holoviz = HolovizOp(self, allocator=pool, name="holoviz", **self.kwargs("holoviz"))

        # Holoviz
        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})

        self.add_flow(point_publisher, decoder_configurator, {("out", "point_in")})
        self.add_flow(inference, decoder_configurator, {("transmitter", "in")})
        self.add_flow(decoder_configurator, inference_decoder, {("out", "receivers")})
        # point visualization
        self.add_flow(decoder_configurator, holoviz, {("point", "receivers")})
        self.add_flow(inference_decoder, postprocessor, {("transmitter", "in")})
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
