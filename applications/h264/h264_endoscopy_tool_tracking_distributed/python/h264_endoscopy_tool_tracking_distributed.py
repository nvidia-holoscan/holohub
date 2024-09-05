# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import logging
import os
import sys
from argparse import ArgumentParser, Namespace

from cloud_inference_fragment import CloudInferenceFragment
from holoscan.core import Application
from video_input_fragment import VideoInputFragment
from viz_fragment import VizFragment

logger = logging.getLogger("h264_endoscopy_tool_tracking_distributed")


class EndoscopyApp(Application):
    def __init__(self, data):
        """Initialize the endoscopy tool tracking application"""
        super().__init__()

        # set name
        self.name = "Endoscopy App"

        if (data is None) or (data == "none"):
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.datapath_ = data

    def compose(self):
        width = 854
        height = 480

        video_in = VideoInputFragment(self, "video_in", self.datapath_)
        cloud_inference = CloudInferenceFragment(self, "inference", self.datapath_)
        viz = VizFragment(self, "viz", width, height)

        self.add_flow(
            video_in,
            cloud_inference,
            {("bitstream_reader.output_transmitter", "video_decoder_request.input_frame")},
        )
        self.add_flow(
            video_in, viz, {("decoder_output_format_converter.tensor", "holoviz.receivers")}
        )
        self.add_flow(
            cloud_inference,
            viz,
            {
                ("tool_tracking_postprocessor.out_coords", "holoviz.receivers"),
                ("tool_tracking_postprocessor.out_mask", "holoviz.receivers"),
            },
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Distributed Endoscopy Tool Tracking Application")
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default=os.environ.get("HOLOSCAN_INPUT_PATH", None),
        help="Input dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=os.environ.get(
            "HOLOSCAN_CONFIG_PATH",
            os.path.join(
                os.path.dirname(__file__), "h264_endoscopy_tool_tracking_distributed.yaml"
            ),
        ),
        help="Input dataset.",
    )

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.data is None:
        logger.error(
            "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
        )
        sys.exit(-1)

    app = EndoscopyApp(args.data)

    app.config(args.config)
    app.run()
