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

import logging
import os
from argparse import ArgumentParser

from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, Tracker
from holoscan.operators import FormatConverterOp, HolovizOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType, RMMAllocator

from holohub.nv_video_decoder import NvVideoDecoderOp
from holohub.nv_video_encoder import NvVideoEncoderOp


class StatsOp(Operator):
    """Print common streaming statistics"""

    def __init__(self, app, *args, **kwargs):
        self.encode_latency = []
        self.decode_latency = []
        self.jitter_time = []
        self.fps = []
        self.first_frame_ignored = False
        self._logger = logging.getLogger(__name__)
        super().__init__(app, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        if not self.first_frame_ignored:
            self.first_frame_ignored = True
            return

        self.encode_latency.append(self.metadata["video_encoder_encode_latency_ms"])
        self.decode_latency.append(self.metadata["video_decoder_decode_latency_ms"])
        self.jitter_time.append(self.metadata["jitter_time"])
        self.fps.append(self.metadata["fps"])

    def stop(self):
        if self.encode_latency:
            self._logger.info(
                f"Encode Latency (ms) (min, max, avg): {min(self.encode_latency):.3f}, {max(self.encode_latency):.3f}, {sum(self.encode_latency) / len(self.encode_latency):.3f}"
            )
            self._logger.info(
                f"Decode Latency (ms) (min, max, avg): {min(self.decode_latency):.3f}, {max(self.decode_latency):.3f}, {sum(self.decode_latency) / len(self.decode_latency):.3f}"
            )
            self._logger.info(
                f"Jitter Time (ms) (min, max, avg): {min(self.jitter_time):.3f}, {max(self.jitter_time):.3f}, {sum(self.jitter_time) / len(self.jitter_time):.3f}"
            )
            self._logger.info(
                f"FPS (min, max, avg): {min(self.fps):.3f}, {max(self.fps):.3f}, {sum(self.fps) / len(self.fps):.3f}"
            )


class NVIDIAVideoCodecApp(Application):
    def __init__(self, data=None):
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

        if data is None:
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        width = self.kwargs("holoviz")["width"]
        height = self.kwargs("holoviz")["height"]
        fps = self.kwargs("holoviz")["framerate"]
        source_block_size = width * height * 3 * 4
        source_num_blocks = 2

        video_dir = self.sample_data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            PeriodicCondition(self, name="periodic-condition", recess_period=1 / fps),
            name="replayer",
            directory=video_dir,
            allocator=RMMAllocator(
                self,
                name="video_replayer_allocator",
            ),
            **self.kwargs("replayer"),
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            **self.kwargs("format_converter"),
        )

        encoder = NvVideoEncoderOp(
            self,
            name="nv_encoder",
            width=width,
            height=height,
            allocator=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.HOST,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            **self.kwargs("encoder"),
        )

        decoder = NvVideoDecoderOp(
            self,
            name="nv_decoder",
            allocator=RMMAllocator(self, name="video_decoder_allocator"),
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

        self.add_flow(source, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, encoder, {("tensor", "input")})
        self.add_flow(encoder, decoder, {("output", "input")})
        self.add_flow(decoder, visualizer, {("output", "receivers")})
        self.add_flow(decoder, stats, {("output", "input")})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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
        config_file = os.path.join(os.path.dirname(__file__), "nvc_encode_decode.yaml")
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
    with Tracker(app) as tracker:
        app.run()
        tracker.print()
