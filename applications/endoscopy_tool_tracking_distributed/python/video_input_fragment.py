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

from holoscan.core import Fragment
from holoscan.operators import VideoStreamReplayerOp


class VideoInputFragment(Fragment):
    @property
    def source_block_size(self):
        print("Getting value")
        return self._source_block_size

    @property
    def source_num_blocks(self):
        return self._source_num_blocks

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __init__(self, app, name, video_dir):
        super().__init__(app, name)
        self.video_dir = video_dir

        self._width = 854
        self._height = 480
        video_dir = self.video_dir
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        self.input_op = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            **self.kwargs("replayer"),
        )
        # 4 bytes/channel, 3 channels
        self._source_block_size = self._width * self._height * 3 * 4
        self._source_num_blocks = 2

    def compose(self):
        self.add_operator(self.input_op)
