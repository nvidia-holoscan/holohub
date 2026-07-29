# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os

from holoscan.core import Fragment
from holoscan.operators import VideoStreamReplayerOp


class VideoInputFragment(Fragment):
    def __init__(self, app, name, video_dir):
        super().__init__(app, name)
        self.video_dir = video_dir

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

    def compose(self):
        input_op = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.video_dir,
            **self.kwargs("replayer"),
        )
        try:
            from holoscan.resources import RMMAllocator

            input_op.add_arg(allocator=RMMAllocator(self, name="video_replayer_allocator"))
        except (
            ArithmeticError,
            AssertionError,
            AttributeError,
            EOFError,
            ImportError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            logging.getLogger(__name__).debug("Expected operation failure", exc_info=True)
        self.add_operator(input_op)
