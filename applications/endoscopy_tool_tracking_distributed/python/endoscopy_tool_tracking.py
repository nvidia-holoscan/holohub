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

from cloud_inference_fragment import CloudInferenceFragment
from holoscan.core import Application
from video_input_fragment import VideoInputFragment
from viz_fragment import VizFragment


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
        video_in_fragment = VideoInputFragment(self, "video_in", self.input_path)
        inference_fragment = CloudInferenceFragment(
            self,
            "inference",
            self.input_path,
            video_in_fragment.width,
            video_in_fragment.height,
            video_in_fragment.source_block_size,
            video_in_fragment.source_num_blocks,
        )
        viz_fragment = VizFragment(
            self,
            "viz",
            video_in_fragment.width,
            video_in_fragment.height,
        )

        # Flow definition

        self.add_flow(video_in_fragment, inference_fragment, {("replayer", "format_converter")})

        self.add_flow(
            inference_fragment,
            viz_fragment,
            {
                ("tool_tracking_postprocessor.out_coords", "holoviz.receivers"),
                ("tool_tracking_postprocessor.out_mask", "holoviz.receivers"),
            },
        )

        self.add_flow(
            video_in_fragment,
            viz_fragment,
            {("replayer.output", "holoviz.receivers")},
        )


if __name__ == "__main__":
    # get the Application's arguments
    app_argv = Application().argv

    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")

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
    args = parser.parse_args(app_argv[1:])

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    app = EndoscopyApp(data=args.data)
    app.config(config_file)
    app.run()
