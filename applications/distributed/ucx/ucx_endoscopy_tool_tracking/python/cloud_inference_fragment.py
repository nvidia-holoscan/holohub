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
from holoscan.operators import FormatConverterOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp


class CloudInferenceFragment(Fragment):
    def __init__(
        self,
        app,
        name,
        model_dir,
        width,
        height,
        source_block_size,
        source_num_blocks,
    ):
        super().__init__(app, name)
        self.model_dir = model_dir
        self.width = width
        self.height = height
        self.source_block_size = source_block_size
        self.source_num_blocks = source_num_blocks

    def compose(self):
        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=self.source_block_size,
            num_blocks=self.source_num_blocks,
        )

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("format_converter"),
        )

        lstm_inferer_block_size = 107 * 60 * 7 * 4
        lstm_inferer_num_blocks = 2 + 5 * 2
        model_file_path = os.path.join(self.model_dir, "tool_loc_convlstm.onnx")
        engine_cache_dir = os.path.join(self.model_dir, "engines")
        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            pool=BlockMemoryPool(
                self,
                name="device_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=lstm_inferer_block_size,
                num_blocks=lstm_inferer_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            **self.kwargs("lstm_inference"),
        )

        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=UnboundedAllocator(
                self,
                name="device_allocator",
            ),
        )

        self.add_flow(format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
