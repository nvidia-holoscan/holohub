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

import argparse
import logging
import os
import sys

from cloud_inference_fragment import CloudInferenceFragment
from holoscan.core import Application
from video_input_fragment import VideoInputFragment
from viz_fragment import VizFragment

logger = logging.getLogger("endoscopy_tool_tracking_distributed")


class EndoscopyApp(Application):
    def __init__(self, data):
        """Initialize the endoscopy tool tracking application"""
        super().__init__()

        # set name
        self.name = "Endoscopy App"

        if data == "none":
            data = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")
            if not os.path.exists(data):
                raise ValueError(f"Could not find video data: {self.video_dir=}")

        self.input_path = data

    def compose(self):
        width = 854
        height = 480
        # 4 bytes/channel, 3 channels
        source_block_size = width * height * 3 * 4
        source_num_blocks = 2

        video_in_fragment = VideoInputFragment(self, "video_in", self.input_path)
        inference_fragment = CloudInferenceFragment(
            self,
            "inference",
            self.input_path,
            width,
            height,
            source_block_size,
            source_num_blocks,
        )
        viz_fragment = VizFragment(
            self,
            "viz",
            width,
            height,
        )

        # Flow definition

        self.add_flow(video_in_fragment, inference_fragment, {("replayer", "format_converter")})

        self.add_flow(
            inference_fragment,
            viz_fragment,
            {("tool_tracking_postprocessor.out", "holoviz.receivers")},
        )

        self.add_flow(
            video_in_fragment,
            viz_fragment,
            {("replayer.output", "holoviz.receivers")},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Endoscopy Tool Tracking Application")
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
            os.path.join(os.path.dirname(__file__), "endoscopy_tool_tracking.yaml"),
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
