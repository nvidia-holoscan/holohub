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
from holoscan.resources import UnboundedAllocator

from holohub.v4l2_plus_source import V4L2PlusSourceOp


# Now define a simple application using the operators defined above
class V4L2PlusApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - V4L2SourceOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
    The HolovizOp displays the processed frames.
    """

    def compose(self):
        source = V4L2PlusSourceOp(
            self,
            name="source",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("source"),
        )

        sink = HolovizOp(
            self,
            name="sink",
            **self.kwargs("sink"),
        )

        self.add_flow(source, sink, {("signal", "receivers")})


if __name__ == "__main__":
    load_env_log_level()
    app = V4L2PlusApp()
    app.config(os.path.join(os.path.dirname(__file__), "v4l2_plus.yaml"))
    app.run()
