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

from holoscan.core import Application, Tracker
from holoscan.operators import FormatConverterOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, RMMAllocator

from holohub.nv_video_encoder import NvVideoEncoderOp
from holohub.tensor_to_file import TensorToFileOp


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
        width = self.kwargs("video")["width"]
        height = self.kwargs("video")["height"]
        source_block_size = width * height * 3 * 4
        source_num_blocks = 2

        video_dir = self.sample_data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
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

        writer = TensorToFileOp(
            self,
            name="nv_writer",
            allocator=RMMAllocator(self, name="video_writer_allocator"),
            **self.kwargs("writer"),
        )

        self.add_flow(source, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, encoder, {("tensor", "input")})
        self.add_flow(encoder, writer, {("output", "input")})


def main():
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
        config_file = os.path.join(os.path.dirname(__file__), "nvc_encode_writer.yaml")
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


if __name__ == "__main__":
    main()
