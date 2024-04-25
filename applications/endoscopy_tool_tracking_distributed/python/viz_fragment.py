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


from holoscan.core import Fragment
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)


class VizFragment(Fragment):
    def __init__(self, app, name, width, height, is_overlay_enabled):
        super().__init__(app, name)
        self.width = width
        self.height = height
        self.is_overlay_enabled = is_overlay_enabled

    def compose(self):
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        visualizer = HolovizOp( 
            self,
            name="holoviz",
            width=self.width,
            height=self.height,
            enable_render_buffer_input=self.is_overlay_enabled,
            enable_render_buffer_output=self.is_overlay_enabled,
            # cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("holoviz_overlay" if self.is_overlay_enabled else "holoviz"),
        )
        
        self.add_operator(visualizer)

