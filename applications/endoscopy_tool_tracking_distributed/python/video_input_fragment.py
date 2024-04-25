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
from holoscan.operators import (
    AJASourceOp,
    VideoStreamReplayerOp,
)

# Enable this line for Yuam capture card
# from holohub.qcap_source import QCAPSourceOp

class VideoInputFragment(Fragment):
    @property
    def is_overlay_enabled(self):
        return self._is_overlay_enabled

    @is_overlay_enabled.setter
    def is_overlay_enabled(self, value):
        self._is_overlay_enabled = value

    @property
    def source_block_size(self):
        print("Getting value")
        return self._source_block_size

    @source_block_size.setter
    def source_block_size(self, value):
        self._source_block_size = value

    @property
    def source_num_blocks(self):
        return self._source_num_blocks

    @source_num_blocks.setter
    def source_num_blocks(self, value):
        self._source_num_blocks = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def rdma(self):
        return self._rdma

    @rdma.setter
    def rdma(self, value):
        self._rdma = value

    def __init__(self, app, name, source, video_dir):
        super().__init__(app, name)
        self._is_overlay_enabled = False
        self._rdma = False
        self._width = None
        self._height = None
        self._source_block_size = None
        self._source_num_blocks = None
        self.input_op = source.lower()
        self.video_dir = video_dir

        if source == "aja":
            aja_kwargs = self.kwargs("aja")
            self.input_op = AJASourceOp(self, name="aja", **aja_kwargs)

            # 4 bytes/channel, 4 channels
            self.width = aja_kwargs["width"]
            self.height = aja_kwargs["height"]
            self.rdma = aja_kwargs["rdma"]
            self.is_overlay_enabled = aja_kwargs["enable_overlay"]
            self.source_block_size = self.width * self.height * 4 * 4
            self.source_num_blocks = 3 if self.rdma else 4
        elif source == "yuan":
            yuan_kwargs = self.kwargs("yuan")
            # Uncomment to enable QCap
            # self.input_op = QCAPSourceOp(self, name="yuan", **yuan_kwargs)

            # 4 bytes/channel, 4 channels
            self.width = yuan_kwargs["width"]
            self.height = yuan_kwargs["height"]
            self.rdma = yuan_kwargs["rdma"]
            self.source_block_size = self.width * self.height * 4 * 4
            self.source_num_blocks = 3 if self.rdma else 4
        else:
            self.width = 854
            self.height = 480
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
            self.source_block_size = self.width * self.height * 3 * 4
            self.source_num_blocks = 2

    def compose(self):
        self.add_operator(self.input_op)
