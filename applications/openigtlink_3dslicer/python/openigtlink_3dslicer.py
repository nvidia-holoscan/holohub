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

from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

from holohub.openigtlink_rx import OpenIGTLinkRxOp
from holohub.openigtlink_tx import OpenIGTLinkTxOp


class OpenIGTLinkApp(Application):
    def __init__(self):
        """Initialize the endoscopy tool tracking application"""
        super().__init__()

        self.name = "Endoscopy App"

        self.sample_data_path = os.environ.get(
            "HOLOHUB_DATA_PATH", "../data/colonoscopy_segmentation"
        )

        self.model_path_map = {
            "colon_seg": os.path.join(self.sample_data_path, "colon.onnx"),
        }

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

        # VideoStreamReplayerOp
        video_dir = os.path.join(self.sample_data_path)
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        replayer = VideoStreamReplayerOp(
            self, name="replayer", directory=video_dir, **self.kwargs("replayer")
        )

        # OpenIGTLinkTxOp
        openigtlink_tx_slicer_img = OpenIGTLinkTxOp(
            self, name="openigtlink_tx_slicer_img", **self.kwargs("openigtlink_tx_slicer_img")
        )

        # OpenIGTLinkRxOp
        openigtlink_rx_slicer_img = OpenIGTLinkRxOp(
            self,
            name="openigtlink_rx_slicer_img",
            allocator=UnboundedAllocator(self, name="host_allocator"),
            **self.kwargs("openigtlink_rx_slicer_img"),
        )

        # FormatConverterOp
        n_channels = 4  # RGBA
        width_preprocessor = 256
        height_preprocessor = 256
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * 1
        preprocessor_num_blocks = 2
        uint8_preprocessor = FormatConverterOp(
            self,
            name="uint8_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("uint8_preprocessor"),
        )

        # FormatConverterOp
        width_preprocessor = 720
        height_preprocessor = 576
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * 4
        preprocessor_num_blocks = 2
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_preprocessor"),
        )

        # InferenceOp
        n_channels_inference = 2
        width_inference = 512
        height_inference = 512
        bpp_inference = 4
        inference_block_size = (
            width_inference * height_inference * n_channels_inference * bpp_inference
        )
        inference_num_blocks = 2
        segmentation_inference = InferenceOp(
            self,
            name="segmentation_inference_holoinfer",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=inference_block_size,
                num_blocks=inference_num_blocks,
            ),
            model_path_map=self.model_path_map,
            **self.kwargs("segmentation_inference_holoinfer"),
        )

        # SegmentationPostprocessorOp
        postprocessor_block_size = width_inference * height_inference * 1
        postprocessor_num_blocks = 2
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=postprocessor_block_size,
                num_blocks=postprocessor_num_blocks,
            ),
            **self.kwargs("segmentation_postprocessor"),
        )

        # HolovizOp
        segmentation_visualizer = HolovizOp(
            self,
            name="segmentation_visualizer",
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_visualizer"),
        )

        # OpenIGTLinkTxOp
        openigtlink_tx_slicer_holoscan = OpenIGTLinkTxOp(
            self,
            name="openigtlink_tx_slicer_holoscan",
            **self.kwargs("openigtlink_tx_slicer_holoscan"),
        )

        # Build flow
        self.add_flow(replayer, uint8_preprocessor, {("", "source_video")})
        self.add_flow(uint8_preprocessor, openigtlink_tx_slicer_img, {("tensor", "receivers")})
        self.add_flow(
            openigtlink_rx_slicer_img, segmentation_visualizer, {("out_tensor", "receivers")}
        )
        self.add_flow(
            openigtlink_rx_slicer_img, segmentation_preprocessor, {("out_tensor", "source_video")}
        )
        self.add_flow(segmentation_preprocessor, segmentation_inference, {("tensor", "receivers")})
        self.add_flow(segmentation_inference, segmentation_postprocessor, {("transmitter", "")})
        self.add_flow(segmentation_postprocessor, segmentation_visualizer, {("", "receivers")})
        self.add_flow(
            segmentation_visualizer,
            openigtlink_tx_slicer_holoscan,
            {("render_buffer_output", "receivers")},
        )


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "openigtlink_3dslicer.yaml")

    app = OpenIGTLinkApp()
    app.config(config_file)
    app.run()
