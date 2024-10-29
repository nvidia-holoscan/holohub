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

from holoscan.logger import set_log_level, LogLevel
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

from holohub.azure_kinect_camera import AzureKinectCameraOp


class AzureKinectCameraApp(Application):
    def __init__(self):
        """Initialize the azure kinect camera application"""
        super().__init__()
        self.name = "Azure Kinect Camera App"


    def compose(self):
        # cuda_stream_pool = CudaStreamPool(
        #     self,
        #     name="cuda_stream",
        #     dev_id=0,
        #     stream_flags=0,
        #     stream_priority=0,
        #     reserved_size=1,
        #     max_size=5,
        # )
        allocator = UnboundedAllocator(self, name="host_allocator")

        # AzureKinectCameraOp
        camera = AzureKinectCameraOp(
            self,
            name="camera",
            allocator=allocator,
            **self.kwargs("camera")
        )

        # HolovizOp
        color_visualizer = HolovizOp(
            self,
            name="color_visualizer",
            window_title="Color Stream",
            allocator=allocator,
            # cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("color_visualizer"),
        )

        specs = [HolovizOp.InputSpec("", HolovizOp.InputType.COLOR)]
        depth_visualizer = HolovizOp(
            self,
            tensors=specs,
            name="depth_visualizer",
            window_title="Depth Stream",
            allocator=allocator,
            # cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("depth_visualizer"),
        )

        # Build flow
        self.add_flow(camera, color_visualizer, {("color_buffer", "receivers")})
        self.add_flow(camera, depth_visualizer, {("depth_buffer", "receivers")})


if __name__ == "__main__":
    set_log_level(LogLevel.INFO)
    config_file = os.path.join(os.path.dirname(__file__), "narvis_simple_capture.yaml")

    app = AzureKinectCameraApp()
    app.config(config_file)
    app.run()
