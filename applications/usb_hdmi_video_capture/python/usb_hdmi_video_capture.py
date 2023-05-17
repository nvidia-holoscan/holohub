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

from holoscan.core import Application
from holoscan.logger import load_env_log_level
from holoscan.operators import HolovizOp
from holoscan.resources import BlockMemoryPool

from holohub.v4l2_video_capture import V4L2VideoCaptureOp


# Now define a simple application using the operators defined above
class App(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - V4L2VideoCaptureOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
    The HolovizOp displays the processed frames.
    """

    def compose(self):

        source_args = self.kwargs("source")
        width = source_args["width"]
        height = source_args["height"]
        n_channels = 4
        block_size = width * height * n_channels

        source = V4L2VideoCaptureOp(
            self,
            name="source",
            allocator=BlockMemoryPool(
                self, 
                name="pool",
                storage_type=0,
                block_size=block_size,
                num_blocks=1
            ),
            **source_args,
        )

        # Set Holoviz width and height from source resolution
        visualizer_args = self.kwargs("visualizer")
        visualizer_args["width"] = width
        visualizer_args["height"] = height
        visualizer = HolovizOp(
            self,
            name="visualizer",
            **visualizer_args,
        )

        self.add_flow(source, visualizer, {("signal", "receivers")})


if __name__ == "__main__":
    load_env_log_level()
    app = App()
    app.config(os.path.join(os.path.dirname(__file__), "usb_hdmi_video_capture.yaml"))
    app.run()
