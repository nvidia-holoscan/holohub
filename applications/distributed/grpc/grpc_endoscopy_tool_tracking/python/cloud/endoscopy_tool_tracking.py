# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from queue import Queue

from holoscan.operators import FormatConverterOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp
from operators.grpc_operators.python.server.grpc_application import HoloscanGrpcApplication


class EndoscopyToolTrackingPipeline(HoloscanGrpcApplication):
    def __init__(self, incoming_request_queue: Queue, outgoing_response_queue: Queue):
        super().__init__(incoming_request_queue, outgoing_response_queue)

    def compose(self):
        # Call base class compose to initialize the queues.
        super().compose()

        # Create the Endoscopy Tool Tracking (ETT) Pipeline similar to the regular ETT application.
        width = 854
        height = 480

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=width * height * 3 * 4,
            num_blocks=2,
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
        model_file_path = os.path.join(self.data_path, "tool_loc_convlstm.onnx")
        engine_cache_dir = os.path.join(self.data_path, "engines")

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

        bytes_per_float32 = 4
        tool_tracking_postprocessor_block_size = max(
            107 * 60 * 7 * 4 * bytes_per_float32, 7 * 3 * bytes_per_float32
        )
        tool_tracking_postprocessor_num_blocks = 2 * 2
        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            cuda_stream_pool=cuda_stream_pool,
            device_allocator=BlockMemoryPool(
                self,
                name="device_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=tool_tracking_postprocessor_block_size,
                num_blocks=tool_tracking_postprocessor_num_blocks,
            ),
        )

        self.add_flow(self.grpc_request_op, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
        self.add_flow(tool_tracking_postprocessor, self.grpc_response_op, {("out", "input")})

        self.composed = True
