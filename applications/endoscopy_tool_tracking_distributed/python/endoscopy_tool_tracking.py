# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from video_input_fragment import VideoInputFragment
from cloud_inference_fragment import CloudInferenceFragment
from viz_fragment import VizFragment


class EndoscopyApp(Application):
    def __init__(self, data, source="replayer"):
        """Initialize the endoscopy tool tracking application

        Parameters
        ----------
        source : {"replayer", "aja", "deltacast", "yuan"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA or Yuan
            capture card is used.
        """
        super().__init__()

        # set name
        self.name = "Endoscopy App"

        # Optional parameters affecting the graph created by compose.
        self.source = source

        if data == "none":
            data = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")
            if not os.path.exists(data):
                raise ValueError(f"Could not find video data: {self.video_dir=}")

        self.input_path = data

    def compose(self):
        config_key_name = "format_converter_" + self.source.lower()
        video_in_fragment = VideoInputFragment(self, "video_in", self.source, self.input_path)
        inference_fragment = CloudInferenceFragment(
            self,
            "inference",
            config_key_name,
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
            video_in_fragment.is_overlay_enabled,
        )

        # Flow definition

        if self.source == "replayer":
            self.add_flow(
                video_in_fragment,
                inference_fragment,
                {("replayer", "format_converter")}
            )
        else:
            self.add_flow(
                video_in_fragment,
                inference_fragment,
                {(f"{self.source}.video_buffer_output", "format_converter.source_video")}
            )
        self.add_flow(
            inference_fragment,
            viz_fragment,
            {
                ("tool_tracking_postprocessor.out_coords", "holoviz.receivers"),
                ("tool_tracking_postprocessor.out_mask", "holoviz.receivers"),
            },
        )

        if video_in_fragment.is_overlay_enabled:
            # Overlay buffer flow between AJA source and visualizer
            self.add_flow(video_in_fragment, viz_fragment, {("replayer.overlay_buffer_output", "holoviz.render_buffer_input")})
            self.add_flow(viz_fragment, video_in_fragment, {("holoviz.render_buffer_output", "replayer.overlay_buffer_input")})
        else:
            self.add_flow(
                video_in_fragment,
                viz_fragment,
                {("replayer.video_buffer_output" if self.source != "replayer" else "replayer.output", "holoviz.receivers")},
            )


if __name__ == "__main__":
    # get the Application's arguments
    app_argv = Application().argv

    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja", "yuan"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. Otherwise use a "
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
    args = parser.parse_args(app_argv[1:])

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    app = EndoscopyApp(source=args.source, data=args.data)
    app.config(config_file)
    app.run()
