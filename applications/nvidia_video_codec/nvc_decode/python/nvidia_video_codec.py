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
from argparse import ArgumentParser

import numpy as np
from holoscan import as_tensor
from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator, Tracker
from holoscan.gxf import Entity, load_extensions
from holoscan.operators import HolovizOp, GXFCodeletOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    RMMAllocator,
    UnboundedAllocator,
)
from holohub.nv_video_decoder import NvVideoDecoderOp
from holohub.nv_video_encoder import NvVideoEncoderOp
from holohub.nv_video_reader import NvVideoReaderOp

class StatsOp(Operator):
    def __init__(self, app, *args, **kwargs):
        self.encode_latency = []
        self.decode_latency = []
        self.jitter_time = []
        self.fps = []
        self.first_frame_ignored = False
        super().__init__(app, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        if not self.first_frame_ignored:
            self.first_frame_ignored = True
            return

        # Check if metadata exists before accessing it
        if hasattr(self, 'metadata'):
            self.encode_latency.append(self.metadata.get("video_encoder_encode_latency_ms", 0))
            self.decode_latency.append(self.metadata.get("video_decoder_decode_latency_ms", 0))
            self.jitter_time.append(self.metadata.get("jitter_time", 0))
            self.fps.append(self.metadata.get("fps", 0))

            print(f"Encode Latency (min, max, avg): {min(self.encode_latency):.3f}, {max(self.encode_latency):.3f}, {sum(self.encode_latency) / len(self.encode_latency):.3f}")
            print(f"Decode Latency (min, max, avg): {min(self.decode_latency):.3f}, {max(self.decode_latency):.3f}, {sum(self.decode_latency) / len(self.decode_latency):.3f}")
            print(f"Jitter Time (min, max, avg): {min(self.jitter_time):.3f}, {max(self.jitter_time):.3f}, {sum(self.jitter_time) / len(self.jitter_time):.3f}")
            print(f"FPS (min, max, avg): {min(self.fps):.3f}, {max(self.fps):.3f}, {sum(self.fps) / len(self.fps):.3f}")


class NVIDIAVideoCodecApp(Application):
    def __init__(self, data):
        """Initialize the NVIDIA Video Codec application

        Parameters
        ----------
            data: str, optional
                The path to the data directory. If not provided, the data directory will be
                set to the HOLOSCAN_INPUT_PATH environment variable.
        """
        super().__init__()

        # set name
        self.name = "NVIDIA Video Codec App"

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):        
        # Create H.264 file reader without PeriodicCondition - let it run continuously
        h264_file_reader = NvVideoReaderOp(
            self,
            CountCondition(self, 683),
            PeriodicCondition(self, name="periodic_condition", recess_period=1/60),
            name="h264_file_reader", 
            directory=self.sample_data_path,
            allocator=UnboundedAllocator(
                self,
                name="video_reader_pool"
            ),
            **self.kwargs("reader"),
        )

        decoder = NvVideoDecoderOp(
            self,
            name="nv_decoder",
            allocator=UnboundedAllocator(
                self,
                name="video_decoder_pool"
            ),
            **self.kwargs("decoder"),
        )

        visualizer = HolovizOp(
            self,
            name="visualizer",
            allocator=CudaStreamPool(
                self,
                name="cuda_stream",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
            ),
            **self.kwargs("holoviz"),
        )

        stats = StatsOp(self, name="stats")

        self.add_flow(h264_file_reader, decoder, {("output", "input")})
        self.add_flow(decoder, visualizer, {("output", "receivers")})
        self.add_flow(decoder, stats, {("output", "input")})


if __name__ == "__main__":
    default_data_path = f"{os.getcwd()}/data/endoscopy"
    # Parse args
    parser = ArgumentParser(description="NVIDIA Video Codec demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default=os.environ.get("HOLOSCAN_INPUT_PATH", default_data_path),
        help=("Set the data path (default: %(default)s)."),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "nvidia_video_codec.yaml")
    else:
        config_file = args.config

    # handle case where HOLOSCAN_INPUT_PATH is set with no value
    if len(args.data) == 0:
        args.data = default_data_path

    if not os.path.isdir(args.data):
        raise ValueError(
            f"Data path '{args.data}' does not exist. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
        )

    app = NVIDIAVideoCodecApp(data=args.data)
    app.config(config_file)

    context = app.executor.context_uint64
    exts = [
        "libgxf_videodecoderio.so",
    ]
    load_extensions(context, exts)
    with Tracker(app) as tracker:
        app.run()
        tracker.print()

